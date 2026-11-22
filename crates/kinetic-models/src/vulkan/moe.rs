use kinetic_vulkan::{Commands, DeviceExperts, DeviceMatrix, Out, Rows};

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
    /// Experts each token runs through.
    n_used: usize,
    n_ff: usize,
}

impl VulkanMoe {
    pub fn upload(gpu: &Gpu, m: &Moe) -> Result<Self, LoadError> {
        // The routing kernel spreads a softmax over the experts; the shared
        // expert and the sigmoid gate of the other mixtures wait for their
        // own kernels.
        if !matches!(m.gating, Gating::Softmax) {
            return Err(super::layer::unsupported("sigmoid-routed experts"));
        }
        if m.shared_gate.is_some() {
            return Err(super::layer::unsupported("shared experts"));
        }
        Ok(VulkanMoe {
            router: gpu.matrix(&m.router)?,
            gate: gpu.experts(&m.gate)?,
            up: gpu.experts(&m.up)?,
            down: gpu.experts(&m.down)?,
            n_used: m.n_used,
            n_ff: m.gate.rows,
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes()
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
    /// router's weights.
    pub fn record(&self, gpu: &Gpu, w: &Work, c: &mut Commands) -> R {
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
        k.moe_route(&w.batch, w.moe_logits(), &self.gate, k_used, route, c)?;
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
        k.moe_combine(route, w.moe_out(), &w.y, w.d, k_used, c)
    }
}

impl Gpu {
    /// Copies a routed expert matrix to the device.
    pub(super) fn experts(
        &self,
        e: &crate::blocks::Experts,
    ) -> Result<DeviceExperts, LoadError> {
        DeviceExperts::upload(&self.device, e.ty, e.data, e.rows, e.cols, e.count)
            .map_err(super::vulkan_error)
    }
}
