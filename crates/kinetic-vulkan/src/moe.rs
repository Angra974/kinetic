use std::rc::Rc;

use kinetic_quant::{Matrix, TensorType};

use crate::batch::Batch;
use crate::device::Device;
use crate::commands::Commands;
use crate::error::{Error, Result};
use crate::gemv::{block_len, GemvInput, Out};
use crate::kernels::{Kernels, MAX_GROUPS};
use crate::layout::device_layout;
use crate::memory::DeviceBuffer;

/// Invocations per workgroup of the GEMV kernels, four rows of 32.
const ROWS_PER_GROUP: usize = 4;
/// Elements per block, and per slot, of the quantization of a routed input.
const Q8_BLOCK: usize = 32;

/// The routed expert matrices of a mixture-of-experts layer, one after the
/// other in one buffer, each laid out for the GEMV kernels.
pub struct DeviceExperts {
    ty: TensorType,
    rows: usize,
    cols: usize,
    count: usize,
    /// 16-byte words of one expert's matrix, and of the whole buffer.
    words: usize,
    total_words: usize,
    data: DeviceBuffer<u8>,
}

impl DeviceExperts {
    /// Copies `data` to the device, one expert at a time so the host never
    /// holds the planar copy of the whole mixture. Fails for types without
    /// a GEMV kernel and for row lengths that are not a multiple of 32.
    pub fn upload(
        device: &Rc<Device>,
        ty: TensorType,
        data: &[u8],
        rows: usize,
        cols: usize,
        count: usize,
    ) -> Result<DeviceExperts> {
        let Some(block) = block_len(ty) else {
            return Err(Error::Unsupported(ty));
        };
        let bytes = rows * cols / block * ty.block_bytes().max(1);
        if count == 0 || !cols.is_multiple_of(Q8_BLOCK) || data.len() != bytes * count {
            return Err(Error::Shape(format!(
                "{count} experts of {rows}x{cols}: {} bytes",
                data.len()
            )));
        }
        let mut words = 0;
        let mut buffer = DeviceBuffer::<u8>::new(device, 16)?;
        for e in 0..count {
            let m = Matrix::new(ty, &data[e * bytes..(e + 1) * bytes], rows, cols);
            let mut chunk = device_layout(&m).into_owned();
            // Kernels read whole 16-byte words, also of the last plane.
            chunk.resize(chunk.len().next_multiple_of(16), 0);
            if e == 0 {
                words = chunk.len() / 16;
                buffer = DeviceBuffer::new(device, words * 16 * count)?;
            }
            buffer.copy_range_from(e * words * 16, &chunk)?;
        }
        Ok(DeviceExperts {
            ty,
            rows,
            cols,
            count,
            words,
            total_words: words * count,
            data: buffer,
        })
    }

    pub fn ty(&self) -> TensorType {
        self.ty
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// 16-byte words of one expert's matrix, the stride between experts.
    pub fn words(&self) -> usize {
        self.words
    }

    pub fn size_bytes(&self) -> usize {
        self.total_words * 16
    }
}

/// What the router writes for each slot: the slot count, then each slot's
/// expert matrix as a word offset, and each slot's weight.
pub struct Route {
    /// The slot count, then each slot's expert matrix as a word offset.
    pub slots: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
}

impl Route {
    /// Room for `slots` of up to `experts` experts each.
    pub fn new(device: &Rc<Device>, slots: usize) -> Result<Route> {
        Ok(Route {
            slots: DeviceBuffer::new(device, slots + 1)?,
            weights: DeviceBuffer::new(device, slots.max(1))?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        4 * self.slots.len() + 4 * self.weights.len()
    }
}

impl Kernels {
    /// Records the router's choices for the pass's tokens: softmax over the
    /// experts, the `k` likeliest per token, their normalized weights and
    /// the word each slot's expert starts at. `logits` holds one row of
    /// `experts` values per token.
    pub fn moe_route(
        &self,
        batch: &Batch,
        logits: &DeviceBuffer<f32>,
        experts: &DeviceExperts,
        k: usize,
        route: &Route,
        commands: &mut Commands,
    ) -> Result<()> {
        let rows = batch.rows();
        if logits.len() < rows * experts.count()
            || route.weights.len() < rows * k
        {
            return Err(Error::Shape(format!(
                "routing {rows} tokens over {} experts into {} slots",
                experts.count(),
                route.weights.len()
            )));
        }
        // SAFETY: matches moe_route.wgsl: the batch, the logits, the slots
        // and the weights at bindings 0..4, then the expert count, the
        // picks per token and one expert's stride. The buffers hold the
        // rows the batch can name, and rows past the pass return.
        unsafe {
            commands.dispatch(
                &self.moe_route,
                &[
                    batch.device().allocation(),
                    logits.allocation(),
                    route.slots.allocation(),
                    route.weights.allocation(),
                ],
                &[experts.count() as u32, k as u32],
                [rows as u32, 1, 1],
            )
        };
        Ok(())
    }

    /// Records `y = w x` for the routed experts: row `r` of slot `s` of the
    /// output takes its weights from the expert the router picked for `s`,
    /// and its input from row `s / share` of `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gemv(
        &self,
        experts: &DeviceExperts,
        x: &GemvInput,
        y: &DeviceBuffer<f32>,
        out: Out,
        share: usize,
        route: &Route,
        commands: &mut Commands,
    ) -> Result<()> {
        let kernel = match experts.ty() {
            TensorType::IQ3XXS => &self.moe_gemv_iq3_xxs,
            TensorType::IQ3S => &self.moe_gemv_iq3_s,
            TensorType::IQ4XS => &self.moe_gemv_iq4_xs,
            other => return Err(Error::Unsupported(other)),
        };
        let block = block_len(experts.ty()).ok_or(Error::Unsupported(experts.ty()))?;
        let slots = route.weights.len();
        let rows = slots * experts.rows();
        let work = rows.div_ceil(ROWS_PER_GROUP);
        let wide = work.min(MAX_GROUPS as usize);
        let tall = work.div_ceil(wide.max(1));
        // `x` holds either the pass's tokens or one row per slot, and both
        // are what the share asks the kernel to read.
        let inputs = match share {
            1 => slots,
            _ => slots / share.max(1),
        };
        if x.width() != experts.cols()
            || x.rows() < inputs
            || y.len() < slots * out.stride
            || u32::try_from(rows).is_err()
        {
            return Err(Error::Shape(format!(
                "grouped GEMV of {}x{} on {} slots into {} outputs",
                experts.rows(),
                experts.cols(),
                slots,
                y.len()
            )));
        }
        // SAFETY: matches moe_gemv.wgsl: the weights, the input's three
        // buffers, the output and the slots at bindings 0..6, then the rows
        // of one expert, its blocks per row, the output stride and offset
        // and the share. The input holds the rows the share names and the
        // output every slot row, checked above; workgroups past the last
        // row return.
        unsafe {
            commands.dispatch(
                kernel,
                &[
                    experts.data.allocation(),
                    x.values().allocation(),
                    x.quantized().allocation(),
                    x.block_scales().allocation(),
                    y.allocation(),
                    route.slots.allocation(),
                ],
                &[
                    experts.rows() as u32,
                    (experts.cols() / block) as u32,
                    out.stride as u32,
                    out.offset as u32,
                    share as u32,
                    experts.words() as u32,
                ],
                [wide as u32, tall as u32, 1],
            )
        };
        Ok(())
    }

    /// Records the routed activation of every slot: `out = act(gate) * up`
    /// from the `2 * len` values of each slot's row of `gu`, written as a
    /// quantized GEMV input.
    pub fn moe_swiglu(
        &self,
        route: &Route,
        gu: &DeviceBuffer<f32>,
        out: &GemvInput,
        gelu: bool,
        commands: &mut Commands,
    ) -> Result<()> {
        let (len, slots) = (out.width(), route.weights.len());
        if gu.len() < 2 * slots * len || out.rows() < slots {
            return Err(Error::Shape(format!(
                "expert SwiGLU of {slots} slots of {len} from {} values",
                gu.len()
            )));
        }
        let [ox, oq, od] = out.outputs();
        // SAFETY: matches moe_swiglu.wgsl: the slots, gu and the three
        // output buffers at bindings 0..5, then the length and the
        // activation; the buffers hold the slots, checked above, and
        // invocations past them return.
        unsafe {
            commands.dispatch(
                &self.moe_swiglu,
                &[route.slots.allocation(), gu.allocation(), ox, oq, od],
                &[len as u32, gelu as u32],
                [len.div_ceil(256) as u32, slots as u32, 1],
            )
        };
        Ok(())
    }

    /// Records the mix of the slots' outputs into the block's output: row
    /// `t` of `y` is the weighted sum of its `k` rows of `out`, which holds
    /// one row of `d` values per slot.
    pub fn moe_combine(
        &self,
        route: &Route,
        out: &DeviceBuffer<f32>,
        y: &DeviceBuffer<f32>,
        d: usize,
        k: usize,
        commands: &mut Commands,
    ) -> Result<()> {
        let slots = route.weights.len();
        if out.len() < slots * d || y.len() < slots / k * d {
            return Err(Error::Shape(format!(
                "combining {slots} slots into {} values of {d}",
                y.len()
            )));
        }
        // SAFETY: matches moe_combine.wgsl: the slots, the weights, the
        // routed outputs and the block's output at bindings 0..4, then the
        // width and the picks per token; the buffers hold the rows the
        // kernel can ask for, checked above.
        unsafe {
            commands.dispatch(
                &self.moe_combine,
                &[
                    route.slots.allocation(),
                    route.weights.allocation(),
                    out.allocation(),
                    y.allocation(),
                ],
                &[d as u32, k as u32],
                [d.div_ceil(256) as u32, (slots / k) as u32, 1],
            )
        };
        Ok(())
    }
}
