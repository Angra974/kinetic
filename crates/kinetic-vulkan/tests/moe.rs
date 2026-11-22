//! One token through the whole routed feed-forward on the device, against
//! the same arithmetic on the host.

use std::rc::Rc;

use half::f16;
use kinetic_quant::{dequantize, Matrix, TensorType};
use kinetic_vulkan::{
    Batch, Commands, Device, DeviceBuffer, DeviceExperts, DeviceMatrix, Error, GemvInput, Instance,
    Kernels, Out, Route, Rows, SharedMix,
};

struct Gpu {
    device: Rc<Device>,
    kernels: Kernels,
}

fn gpus() -> Vec<Gpu> {
    let physical = match Instance::new().and_then(|i| i.physical_devices()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipped: {e}");
            return Vec::new();
        }
    };
    physical
        .iter()
        .filter_map(|p| match Device::new(p) {
            Ok(device) => Some(Gpu {
                kernels: Kernels::load(&device).unwrap(),
                device,
            }),
            Err(e @ Error::DeviceUnsupported(_)) => {
                eprintln!("skipped {}: {e}", p.info().name);
                None
            }
            Err(e) => panic!("{}: {e}", p.info().name),
        })
        .collect()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn f32(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Random blocks of `ty` with sane f16 scales at their usual offset: first
/// in a block, except Q6_K's, which is last; Q4_K and Q5_K keep a small f16
/// minimum after it.
fn random_matrix(ty: TensorType, rows: usize, cols: usize, rng: &mut Lcg) -> Vec<u8> {
    let size = ty.block_bytes();
    let blocks = rows * cols / ty.block_len();
    let mut data: Vec<u8> = (0..blocks * size).map(|_| rng.next() as u8).collect();
    for blk in data.chunks_exact_mut(size) {
        // A K-quant scale covers a super-block of 256 values, so its f16 is
        // a thousandth of the others' to keep the products the same size.
        let unit = if matches!(ty, TensorType::Q4K | TensorType::Q5K | TensorType::Q6K) {
            0.001
        } else {
            1.0
        };
        let at = if ty == TensorType::Q6K { size - 2 } else { 0 };
        let v = f16::from_f32(unit * (0.001 + rng.f32().abs() * 0.01));
        blk[at..at + 2].copy_from_slice(&v.to_le_bytes());
        if matches!(ty, TensorType::Q4K | TensorType::Q5K) {
            let min = f16::from_f32(0.0005 * unit);
            blk[2..4].copy_from_slice(&min.to_le_bytes());
        }
    }
    data
}

/// The rows of `data` times `x`, dequantized.
fn matvec(ty: TensorType, data: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let m = Matrix::new(ty, data, rows, cols);
    let mut row = vec![0.0; cols];
    (0..rows)
        .map(|r| {
            dequantize(ty, m.row(r), &mut row).unwrap();
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// The `k` experts `v` picks, largest first, weighted by their share of the
/// `k` probabilities, as the host's gating does: a softmax, or a sigmoid per
/// expert when `sigmoid` says so, chosen by probability plus `bias`.
fn picks(v: &mut [f32], bias: &[f32], sigmoid: bool, k: usize) -> Vec<(u32, f32)> {
    if sigmoid {
        for p in v.iter_mut() {
            *p = 1.0 / (1.0 + (-*p).exp());
        }
    } else {
        let max = v.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        for p in v.iter_mut() {
            *p = (*p - max).exp();
            sum += *p;
        }
        for p in v.iter_mut() {
            *p /= sum;
        }
    }
    let score = |e: u32| v[e as usize] + bias[e as usize];
    let mut order: Vec<u32> = (0..v.len() as u32).collect();
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    order.truncate(k);
    let share: f32 = order.iter().map(|&e| v[e as usize]).sum();
    order
        .into_iter()
        .map(|e| (e, v[e as usize] / share.max(6.1035156e-5)))
        .collect()
}

fn check(ty: TensorType, sigmoid: bool, mix: SharedMix) {
    for gpu in gpus() {
        let mut rng = Lcg(ty as u64 + 3);
        let (rows, cols, count, k, tokens) = (256, 2 * 256, 4, 2, 3);
        let gate: Vec<Vec<u8>> = (0..count)
            .map(|_| random_matrix(ty, rows, cols, &mut rng))
            .collect();
        let up: Vec<Vec<u8>> = (0..count)
            .map(|_| random_matrix(ty, rows, cols, &mut rng))
            .collect();
        let down: Vec<Vec<u8>> = (0..count)
            .map(|_| random_matrix(ty, cols, rows, &mut rng))
            .collect();
        let router: Vec<f32> = (0..count * cols).map(|_| rng.f32()).collect();
        let x: Vec<f32> = (0..tokens * cols).map(|_| rng.f32() * 3.0).collect();
        let bias: Vec<f32> = (0..count)
            .map(|_| if sigmoid { rng.f32() * 0.5 } else { 0.0 })
            .collect();
        let shexp_gate: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let shared: Vec<f32> = (0..tokens * cols).map(|_| rng.f32()).collect();

        let dev = &gpu.device;
        let kernels = &gpu.kernels;
        let router_bytes = bytemuck(&router);
        let router = DeviceMatrix::upload(
            dev,
            &Matrix::new(TensorType::F32, router_bytes, count, cols),
        )
        .unwrap();
        let flat = |e: &Vec<Vec<u8>>| e.concat();
        let gate_d = DeviceExperts::upload(dev, ty, &flat(&gate), rows, cols, count).unwrap();
        let up_d = DeviceExperts::upload(dev, ty, &flat(&up), rows, cols, count).unwrap();
        let down_d = DeviceExperts::upload(dev, ty, &flat(&down), cols, rows, count).unwrap();
        let logits = DeviceBuffer::new(dev, tokens * count).unwrap();
        let bias_d = DeviceBuffer::from_slice(dev, &bias).unwrap();
        let shexp = DeviceMatrix::upload(
            dev,
            &Matrix::new(TensorType::F32, bytemuck(&shexp_gate), 1, cols),
        )
        .unwrap();
        let shexp_d = DeviceBuffer::new(dev, tokens).unwrap();
        let y = DeviceBuffer::from_slice(dev, &shared).unwrap();
        let gu = DeviceBuffer::new(dev, tokens * k * 2 * rows).unwrap();
        let out = DeviceBuffer::new(dev, tokens * k * cols).unwrap();
        let mut act = GemvInput::batch(dev, rows, tokens * k).unwrap();
        let mut input = GemvInput::batch(dev, cols, tokens).unwrap();
        let route = Route::new(dev, tokens * k).unwrap();
        let mut batch = Batch::new(dev, tokens, 0, 16, 0).unwrap();
        batch.set(tokens, 0, 0..0, None).unwrap();

        let mut c = Commands::new(dev).unwrap();
        batch.record_upload(&mut c).unwrap();
        input.upload(kernels, &x, &mut c).unwrap();
        kernels
            .gemv_rows(
                &router,
                &input,
                &logits,
                Out {
                    stride: count,
                    offset: 0,
                },
                Rows::Tokens,
                &batch,
                &mut c,
            )
            .unwrap();
        kernels
            .gemv_rows(
                &shexp,
                &input,
                &shexp_d,
                Out {
                    stride: 1,
                    offset: 0,
                },
                Rows::Tokens,
                &batch,
                &mut c,
            )
            .unwrap();
        kernels
            .moe_route(
                &batch, &logits, &gate_d, &bias_d, sigmoid, k, &route, &mut c,
            )
            .unwrap();
        for (m, offset) in [(&gate_d, 0), (&up_d, rows)] {
            let o = Out {
                stride: 2 * rows,
                offset,
            };
            kernels
                .moe_gemv(m, &input, &gu, o, k, &route, &mut c)
                .unwrap();
        }
        kernels
            .moe_swiglu(&route, &gu, &act, false, &mut c)
            .unwrap();
        kernels
            .moe_gemv(
                &down_d,
                &act,
                &out,
                Out {
                    stride: cols,
                    offset: 0,
                },
                1,
                &route,
                &mut c,
            )
            .unwrap();
        kernels
            .moe_combine(&route, &out, &y, cols, k, &shexp_d, mix, &mut c)
            .unwrap();
        c.submit().unwrap();

        let bases = route.slots.to_vec().unwrap();
        let host_weights = route.weights.to_vec().unwrap();
        assert_eq!(bases[0], (tokens * k) as u32);

        // The host: the router's picks, then each picked expert, on the
        // input and activations the device quantized.
        let xe = quantized(&input);
        let act_q = act.quantized().to_vec().unwrap();
        let act_d = act.block_scales().to_vec().unwrap();
        let act_x: Vec<f32> = act_q
            .iter()
            .enumerate()
            .map(|(i, &q)| q as f32 * act_d[2 * (i / 32)])
            .collect();
        let mut expected = match mix {
            SharedMix::None => vec![0.0; tokens * cols],
            _ => shared.clone(),
        };
        for t in 0..tokens {
            let row = &x[t * cols..(t + 1) * cols];
            if mix == SharedMix::Gated {
                let logit: f32 = shexp_gate.iter().zip(row).map(|(a, b)| a * b).sum();
                let scale = 1.0 / (1.0 + (-logit).exp());
                expected[t * cols..][..cols]
                    .iter_mut()
                    .for_each(|v| *v *= scale);
            }
            let mut probs = matvec(TensorType::F32, router_bytes, count, cols, row);
            let picks = picks(&mut probs, &bias, sigmoid, k);
            for (j, (expert, weight)) in picks.iter().enumerate() {
                let slot = t * k + j;
                let e = *expert as usize;
                assert_eq!(
                    bases[1 + slot],
                    e as u32,
                    "slot {slot} names another expert"
                );
                assert!(
                    (host_weights[slot] - weight).abs() < 1e-4,
                    "slot {slot}: weight {} against {weight}",
                    host_weights[slot]
                );
                let d = matvec(
                    ty,
                    &down[e],
                    cols,
                    rows,
                    &act_x[slot * rows..(slot + 1) * rows],
                );
                let out = &mut expected[t * cols..(t + 1) * cols];
                for (o, d) in out.iter_mut().zip(d) {
                    *o += weight * d;
                }
            }
        }
        let got = y.to_vec().unwrap();
        let scale: f64 = expected.iter().map(|v| v.abs() as f64).sum();
        let error: f64 = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum();
        assert!(scale > 0.0);
        assert!(
            error <= 2e-5 * scale + 1e-4,
            "{}: {ty} error {error} on a scale of {scale}",
            dev.info().name
        );
    }
}

fn bytemuck(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

/// The input the kernels multiply with: the int8 copy they quantized.
fn quantized(input: &GemvInput) -> Vec<f32> {
    let qs = input.quantized().to_vec().unwrap();
    let ds = input.block_scales().to_vec().unwrap();
    qs.iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * ds[2 * (i / 32)])
        .collect()
}

#[test]
fn routed_feed_forward_matches_the_host() {
    check(TensorType::IQ3XXS, false, SharedMix::None);
}

#[test]
fn sigmoid_routing_scales_the_shared_expert() {
    check(TensorType::IQ3XXS, true, SharedMix::Gated);
}

#[test]
fn sigmoid_routing_adds_an_ungated_shared_expert() {
    check(TensorType::IQ3XXS, true, SharedMix::Plain);
}

#[test]
fn routed_q4_k_experts_match_the_host() {
    check(TensorType::Q4K, false, SharedMix::None);
}

#[test]
fn sigmoid_routing_scales_q4_k_experts() {
    check(TensorType::Q4K, true, SharedMix::Gated);
}

#[test]
fn routed_q6_k_experts_match_the_host() {
    check(TensorType::Q6K, false, SharedMix::None);
}
