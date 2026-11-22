use kinetic_quant::{Matrix, TensorType};
use kinetic_vulkan::{Commands, DeviceBuffer, DeviceExperts, DeviceMatrix, Out, Rows, SharedMix};

use super::layer::R;
use super::work::Work;
use super::Gpu;
use crate::blocks::{Gating, Moe};
use crate::LoadError;

/// The routed experts of a mixture-of-experts feed-forward on the device;
/// see [`Moe`].
pub(super) struct VulkanMoe {
    /// One row of logits per expert.
    router: DeviceMatrix,
    gate: DeviceExperts,
    up: DeviceExperts,
    down: DeviceExperts,
    /// The shared expert's gate as a single row, dotted with the block's
    /// input to scale that expert's output.
    shared_gate: Option<DeviceMatrix>,
    /// One value per expert, added to a sigmoid's probabilities to pick the
    /// experts; zero for a softmax.
    bias: DeviceBuffer<f32>,
    /// Whether the router takes a sigmoid per expert.
    sigmoid: bool,
    /// Experts each token runs through.
    n_used: usize,
    n_ff: usize,
}

impl VulkanMoe {
    pub fn upload(gpu: &Gpu, m: &Moe) -> Result<Self, LoadError> {
        let (bias, sigmoid) = match &m.gating {
            // The router reads a bias per expert either way.
            Gating::Softmax => (gpu.vector(&vec![0.0; m.n_experts()])?, false),
            Gating::Sigmoid { bias } => (gpu.vector(bias)?, true),
        };
        let shared_gate = match m.shared_gate.as_deref() {
            Some(g) => {
                let bytes: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
                Some(gpu.matrix(&Matrix::new(TensorType::F32, &bytes, 1, g.len()))?)
            }
            None => None,
        };
        Ok(VulkanMoe {
            router: gpu.matrix(&m.router)?,
            gate: gpu.experts(&m.gate)?,
            up: gpu.experts(&m.up)?,
            down: gpu.experts(&m.down)?,
            shared_gate,
            bias,
            sigmoid,
            n_used: m.n_used,
            n_ff: m.gate.rows,
        })
    }

    /// How the experts mix with the layer's own feed-forward, when the layer
    /// has one: behind the shared expert's gate, or as it stands.
    pub fn mix(&self) -> SharedMix {
        match self.shared_gate {
            Some(_) => SharedMix::Gated,
            None => SharedMix::Plain,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes() + 4 * self.bias.len()
    }

    /// The experts it routes over, and how many run per token.
    pub fn shape(&self) -> (usize, usize) {
        (self.router.rows(), self.n_used)
    }

    /// The width of one expert's activations.
    pub fn n_ff(&self) -> usize {
        self.n_ff
    }

    /// Reads `w.h` and leaves the mixture's output in `w.y`, mixed by the
    /// router's weights and, when `mix` asks for it, added to the shared
    /// expert's output `w.y` already holds.
    pub fn record(&self, gpu: &Gpu, w: &Work, mix: SharedMix, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let route = w.route();
        let k_used = self.n_used;
        c.independent(|c| {
            k.gemv_rows(
                &self.router,
                &w.h,
                w.moe_logits(),
                Out {
                    stride: self.router.rows(),
                    offset: 0,
                },
                Rows::Tokens,
                &w.batch,
                c,
            )
        })?;
        if let (Some(gate), SharedMix::Gated) = (&self.shared_gate, mix) {
            let out = Out {
                stride: 1,
                offset: 0,
            };
            k.gemv_rows(gate, &w.h, w.shexp_gate(), out, Rows::Tokens, &w.batch, c)?;
        }
        k.moe_route(
            &w.batch,
            w.moe_logits(),
            &self.gate,
            &self.bias,
            self.sigmoid,
            k_used,
            route,
            c,
        )?;
        c.independent(|c| {
            for (m, offset) in [(&self.gate, 0), (&self.up, self.n_ff)] {
                let out = Out {
                    stride: 2 * self.n_ff,
                    offset,
                };
                k.moe_gemv(m, &w.h, w.gu(), out, k_used, route, c)?;
            }
            Ok(())
        })?;
        let act = w.activations(self.n_ff);
        k.moe_swiglu(route, w.gu(), act, false, c)?;
        let out = Out {
            stride: w.d,
            offset: 0,
        };
        k.moe_gemv(&self.down, act, w.moe_out(), out, 1, route, c)?;
        k.moe_combine(
            route,
            w.moe_out(),
            &w.y,
            w.d,
            k_used,
            w.shexp_gate(),
            mix,
            c,
        )
    }
}

impl Gpu {
    /// Copies a routed expert matrix to the device.
    pub(super) fn experts(&self, e: &crate::blocks::Experts) -> Result<DeviceExperts, LoadError> {
        DeviceExperts::upload(&self.device, e.ty, e.data, e.rows, e.cols, e.count)
            .map_err(super::vulkan_error)
    }
}
