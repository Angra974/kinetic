use kinetic_vulkan::{Commands, DeviceMatrix, Out, Rows};

use super::layer::R;
use super::work::Work;
use super::Gpu;
use crate::blocks::{Activation, SwiGlu};
use crate::LoadError;

/// The gated feed-forward on the device; see [`SwiGlu`].
pub(super) struct VulkanFfn {
    pub gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
    /// Whether the gate goes through GELU rather than SiLU.
    gelu: bool,
}

impl VulkanFfn {
    pub fn upload(gpu: &Gpu, f: &SwiGlu) -> Result<Self, LoadError> {
        let gelu = match f.activation {
            Activation::Silu => false,
            Activation::Gelu => true,
        };
        Ok(VulkanFfn {
            gate: gpu.matrix(&f.gate)?,
            up: gpu.matrix(&f.up)?,
            down: gpu.matrix(&f.down)?,
            gelu,
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes()
    }

    /// Reads `w.h` and leaves the output in `w.y`.
    pub fn record(&self, gpu: &Gpu, w: &Work, c: &mut Commands) -> R {
        let k = &gpu.kernels;
        let n_ff = self.gate.rows();
        c.independent(|c| {
            for (m, offset) in [(&self.gate, 0), (&self.up, n_ff)] {
                let out = Out {
                    stride: 2 * n_ff,
                    offset,
                };
                k.gemv_rows(m, &w.h, &w.gu, out, Rows::Tokens, &w.batch, c)?;
            }
            Ok(())
        })?;
        let act = w.activations(n_ff);
        k.swiglu(&w.batch, &w.gu, act, self.gelu, c)?;
        let out = Out {
            stride: w.d,
            offset: 0,
        };
        k.gemv_rows(&self.down, act, &w.y, out, Rows::Tokens, &w.batch, c)
    }
}
