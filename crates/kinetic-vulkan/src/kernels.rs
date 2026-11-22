use std::rc::Rc;

use crate::commands::Commands;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::memory::DeviceBuffer;
use crate::pipeline::Kernel;

/// A kernel as build.rs wrote it for the driver: SPIR-V for Vulkan, Metal
/// shader source for Metal.
#[cfg(not(metal_driver))]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}
#[cfg(metal_driver)]
macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metal"))
    };
}

const READ_SUM_THREADS: u64 = 256;
/// 16-byte words each read_sum invocation reads, at least; more workgroups
/// would only add scheduling overhead.
const READ_SUM_WORDS_PER_THREAD: u64 = 16;
/// The dispatch size every device supports in each dimension.
pub(crate) const MAX_GROUPS: u32 = 65535;

/// The crate's kernels, built for one device.
pub struct Kernels {
    pub(crate) gemv_f32: Rc<Kernel>,
    pub(crate) gemv_f16: Rc<Kernel>,
    pub(crate) gemv_q8_0: Rc<Kernel>,
    pub(crate) gemv_q4_k: Rc<Kernel>,
    /// The same sums for a pass of many tokens; see `shaders/gemv_many.wgsl`.
    pub(crate) gemv_q4_k_many: Rc<Kernel>,
    pub(crate) gemv_q5_k_many: Rc<Kernel>,
    pub(crate) gemv_q6_k_many: Rc<Kernel>,
    pub(crate) gemv_q5_k: Rc<Kernel>,
    pub(crate) gemv_q6_k: Rc<Kernel>,
    pub(crate) gemv_iq3_xxs: Rc<Kernel>,
    pub(crate) gemv_iq3_s: Rc<Kernel>,
    pub(crate) gemv_iq4_xs: Rc<Kernel>,
    pub(crate) gemv_q4_0: Rc<Kernel>,
    pub(crate) gemv_iq3_xxs_many: Rc<Kernel>,
    pub(crate) gemv_iq4_xs_many: Rc<Kernel>,
    pub(crate) gemv_q4_0_many: Rc<Kernel>,
    pub(crate) moe_gemv_iq3_xxs: Rc<Kernel>,
    pub(crate) moe_gemv_iq3_s: Rc<Kernel>,
    pub(crate) moe_gemv_iq4_xs: Rc<Kernel>,
    pub(crate) moe_gemv_q4_k: Rc<Kernel>,
    pub(crate) moe_gemv_q6_k: Rc<Kernel>,
    pub(crate) moe_route: Rc<Kernel>,
    pub(crate) moe_swiglu: Rc<Kernel>,
    pub(crate) moe_combine: Rc<Kernel>,
    pub(crate) quantize_q8: Rc<Kernel>,
    pub(crate) norm: Rc<Kernel>,
    pub(crate) swiglu: Rc<Kernel>,
    pub(crate) argmax: Rc<Kernel>,
    pub(crate) rope: Rc<Kernel>,
    pub(crate) short_conv: Rc<Kernel>,
    pub(crate) delta_conv: Rc<Kernel>,
    pub(crate) delta_rule: Rc<Kernel>,
    pub(crate) gated_norm: Rc<Kernel>,
    pub(crate) attend_64: Rc<Kernel>,
    pub(crate) attend_128: Rc<Kernel>,
    pub(crate) attend_256: Rc<Kernel>,
    pub(crate) attend_512: Rc<Kernel>,
    read_sum: Rc<Kernel>,
    /// The batch words of a pass of one token, for GEMVs outside a model.
    pub(crate) single: DeviceBuffer<u32>,
}

impl Kernels {
    /// Builds the pipelines for the kernels embedded in this crate.
    pub fn load(device: &Rc<Device>) -> Result<Kernels> {
        let gemv = |name, spirv| Kernel::new(device, name, spirv, 6, 5);
        let grouped = |name, spirv| Kernel::new(device, name, spirv, 6, 6);
        let attend = |name, spirv| Kernel::new(device, name, spirv, 7, 9);
        Ok(Kernels {
            gemv_f32: gemv("gemv_f32", spirv!("gemv_f32"))?,
            gemv_f16: gemv("gemv_f16", spirv!("gemv_f16"))?,
            gemv_q8_0: gemv("gemv_q8_0", spirv!("gemv_q8_0"))?,
            gemv_q4_k: gemv("gemv_q4_k", spirv!("gemv_q4_k"))?,
            #[cfg(not(metal_driver))]
            gemv_q4_k_many: gemv("gemv_q4_k_many", spirv!("gemv_q4_k_many"))?,
            #[cfg(metal_driver)]
            gemv_q4_k_many: gemv(
                "gemv_q4_k_many",
                include_bytes!("../shaders/metal/gemv_q4_k_many.metal"),
            )?,
            #[cfg(not(metal_driver))]
            gemv_q5_k_many: gemv("gemv_q5_k_many", spirv!("gemv_q5_k_many"))?,
            #[cfg(not(metal_driver))]
            gemv_q6_k_many: gemv("gemv_q6_k_many", spirv!("gemv_q6_k_many"))?,
            #[cfg(metal_driver)]
            gemv_q5_k_many: gemv(
                "gemv_q5_k_many",
                include_bytes!("../shaders/metal/gemv_q5_k_many.metal"),
            )?,
            #[cfg(metal_driver)]
            gemv_q6_k_many: gemv(
                "gemv_q6_k_many",
                include_bytes!("../shaders/metal/gemv_q6_k_many.metal"),
            )?,
            gemv_q5_k: gemv("gemv_q5_k", spirv!("gemv_q5_k"))?,
            gemv_q6_k: gemv("gemv_q6_k", spirv!("gemv_q6_k"))?,
            gemv_iq3_xxs: gemv("gemv_iq3_xxs", spirv!("gemv_iq3_xxs"))?,
            gemv_iq3_s: gemv("gemv_iq3_s", spirv!("gemv_iq3_s"))?,
            gemv_iq4_xs: gemv("gemv_iq4_xs", spirv!("gemv_iq4_xs"))?,
            gemv_q4_0: gemv("gemv_q4_0", spirv!("gemv_q4_0"))?,
            gemv_iq3_xxs_many: gemv("gemv_iq3_xxs_many", spirv!("gemv_iq3_xxs_many"))?,
            gemv_iq4_xs_many: gemv("gemv_iq4_xs_many", spirv!("gemv_iq4_xs_many"))?,
            gemv_q4_0_many: gemv("gemv_q4_0_many", spirv!("gemv_q4_0_many"))?,
            moe_gemv_iq3_xxs: grouped("moe_gemv_iq3_xxs", spirv!("moe_gemv_iq3_xxs"))?,
            moe_gemv_iq3_s: grouped("moe_gemv_iq3_s", spirv!("moe_gemv_iq3_s"))?,
            moe_gemv_iq4_xs: grouped("moe_gemv_iq4_xs", spirv!("moe_gemv_iq4_xs"))?,
            moe_gemv_q4_k: grouped("moe_gemv_q4_k", spirv!("moe_gemv_q4_k"))?,
            moe_gemv_q6_k: grouped("moe_gemv_q6_k", spirv!("moe_gemv_q6_k"))?,
            moe_route: Kernel::new(device, "moe_route", spirv!("moe_route"), 5, 3)?,
            moe_swiglu: Kernel::new(device, "moe_swiglu", spirv!("moe_swiglu"), 5, 2)?,
            moe_combine: Kernel::new(device, "moe_combine", spirv!("moe_combine"), 5, 4)?,
            quantize_q8: Kernel::new(device, "quantize_q8", spirv!("quantize_q8"), 3, 1)?,
            norm: Kernel::new(device, "norm", spirv!("norm"), 8, 5)?,
            swiglu: Kernel::new(device, "swiglu", spirv!("swiglu"), 5, 1)?,
            argmax: Kernel::new(device, "argmax", spirv!("argmax"), 3, 1)?,
            rope: Kernel::new(device, "rope", spirv!("rope"), 5, 13)?,
            short_conv: Kernel::new(device, "short_conv", spirv!("short_conv"), 7, 3)?,
            delta_conv: Kernel::new(device, "delta_conv", spirv!("delta_conv"), 6, 7)?,
            delta_rule: Kernel::new(device, "delta_rule", spirv!("delta_rule"), 4, 7)?,
            gated_norm: Kernel::new(device, "gated_norm", spirv!("gated_norm"), 7, 5)?,
            attend_64: attend("attend_64", spirv!("attend_64"))?,
            attend_128: attend("attend_128", spirv!("attend_128"))?,
            attend_256: attend("attend_256", spirv!("attend_256"))?,
            attend_512: attend("attend_512", spirv!("attend_512"))?,
            read_sum: Kernel::new(device, "read_sum", spirv!("read_sum"), 2, 1)?,
            single: DeviceBuffer::from_slice(device, &[1, 0, 0, 1])?,
        })
    }

    /// Bytes per second the device reads its memory at, measured by
    /// [`Kernels::read_sum`] on a buffer far larger than any cache.
    ///
    /// This is the ceiling for memory-bound kernels, and the only bandwidth
    /// a Vulkan driver reports: it has no nominal figure to ask for.
    pub fn read_bandwidth(&self, device: &Rc<Device>) -> Result<f64> {
        // far larger than any GPU cache, so every read reaches DRAM
        const BYTES: usize = 512 << 20;
        const REPS: usize = 10;
        // the fastest run counts: the others were slowed by whatever else
        // used the memory meanwhile
        const TRIALS: usize = 3;
        let data = DeviceBuffer::<u8>::new(device, BYTES)?;
        let sum = DeviceBuffer::<u32>::new(device, 1)?;
        let mut best = f64::INFINITY;
        for _ in 0..TRIALS {
            let mut commands = Commands::new(device)?;
            let start = commands.timestamp()?;
            for _ in 0..REPS {
                self.read_sum(&data, &sum, &mut commands)?;
            }
            let end = commands.timestamp()?;
            best = best.min(commands.submit()?.seconds(start, end) / REPS as f64);
        }
        Ok(BYTES as f64 / best)
    }

    /// Reads all of `data` once and adds its sum as 32-bit words to `out[0]`.
    ///
    /// The sum is meaningless; the kernel's throughput is the achievable
    /// read bandwidth of device memory, the ceiling for memory-bound kernels.
    /// `data.len()` must be a multiple of 16.
    pub fn read_sum(
        &self,
        data: &DeviceBuffer<u8>,
        out: &DeviceBuffer<u32>,
        commands: &mut Commands,
    ) -> Result<()> {
        if !data.len().is_multiple_of(16) || out.is_empty() {
            return Err(Error::Shape(format!(
                "read_sum needs a multiple of 16 bytes and one output, got {} and {}",
                data.len(),
                out.len()
            )));
        }
        let words = (data.len() / 16) as u64;
        let groups = words
            .div_ceil(READ_SUM_THREADS * READ_SUM_WORDS_PER_THREAD)
            .clamp(1, MAX_GROUPS.into()) as u32;
        // SAFETY: matches read_sum.wgsl: `data` and `out` at bindings 0 and
        // 1 and the word count, which fits in 32 bits because buffers are
        // capped at the device's storage buffer range.
        unsafe {
            commands.dispatch(
                &self.read_sum,
                &[data.allocation(), out.allocation()],
                &[words as u32],
                [groups, 1, 1],
            )
        };
        Ok(())
    }
}
