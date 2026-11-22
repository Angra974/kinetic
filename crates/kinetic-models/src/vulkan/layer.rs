use std::rc::Rc;

use kinetic_vulkan::{
    Commands, ConvRing, DeltaState, Device, DeviceBuffer, KvCache, Residual, SharedMix,
};

use super::attention::VulkanAttention;
use super::delta_net::VulkanDeltaNet;
use super::ffn::VulkanFfn;
use super::moe::VulkanMoe;
use super::rope::RopeTables;
use super::short_conv::VulkanShortConv;
use super::work::{Needs, Work};
use super::Gpu;
use crate::blocks::Mixer;
use crate::model::Layer;
use crate::LoadError;

pub(super) type R = kinetic_vulkan::Result<()>;

pub(super) fn unsupported(what: &str) -> LoadError {
    LoadError(format!("{what} are not supported on Vulkan yet"))
}

/// A token mixer on the device.
enum VulkanMixer {
    Attention(VulkanAttention),
    ShortConv(VulkanShortConv),
    DeltaNet(VulkanDeltaNet),
}

/// What a sequence keeps of one layer.
pub(super) enum LayerState {
    Attention(KvCache),
    ShortConv(ConvRing),
    DeltaNet(DeltaState),
}

impl LayerState {
    pub fn size_bytes(&self) -> usize {
        match self {
            LayerState::Attention(c) => c.size_bytes(),
            LayerState::ShortConv(r) => r.size_bytes(),
            LayerState::DeltaNet(s) => s.size_bytes(),
        }
    }
}

/// The feed-forward of a layer: a dense one, or the routed experts of a
/// mixture, which a layer may have on top of a dense feed-forward, the
/// shared expert every token runs.
enum VulkanFeed {
    Dense(VulkanFfn),
    Moe(VulkanMoe),
    Shared(VulkanFfn, VulkanMoe),
}

impl VulkanFeed {
    fn record(&self, gpu: &Gpu, w: &Work, c: &mut Commands) -> R {
        match self {
            VulkanFeed::Dense(f) => f.record(gpu, w, c),
            VulkanFeed::Moe(m) => m.record(gpu, w, SharedMix::None, c),
            VulkanFeed::Shared(f, m) => {
                f.record(gpu, w, c)?;
                m.record(gpu, w, m.mix(), c)
            }
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            VulkanFeed::Dense(f) => f.size_bytes(),
            VulkanFeed::Moe(m) => m.size_bytes(),
            VulkanFeed::Shared(f, m) => f.size_bytes() + m.size_bytes(),
        }
    }

    /// The widths of the activations this feed-forward reads: its own, and
    /// the routed experts' when it has a shared expert too.
    fn widths(&self) -> impl Iterator<Item = usize> + '_ {
        match self {
            VulkanFeed::Dense(f) => [Some(f.gate.rows()), None],
            VulkanFeed::Moe(m) => [None, Some(m.n_ff())],
            VulkanFeed::Shared(f, m) => [Some(f.gate.rows()), Some(m.n_ff())],
        }
        .into_iter()
        .flatten()
    }
}

/// A decoder layer on the device; see [`Layer`].
pub(super) struct VulkanLayer {
    pub mixer_norm: DeviceBuffer<f32>,
    mixer: VulkanMixer,
    mixer_out_norm: Option<DeviceBuffer<f32>>,
    ffn_norm: DeviceBuffer<f32>,
    ffn: VulkanFeed,
    ffn_out_norm: Option<DeviceBuffer<f32>>,
    residual_scale: f32,
    out_scale: f32,
}

impl VulkanLayer {
    pub fn upload(gpu: &Gpu, l: &Layer, ropes: &mut RopeTables) -> Result<VulkanLayer, LoadError> {
        let optional = |v: &Option<Vec<f32>>| v.as_deref().map(|v| gpu.vector(v)).transpose();
        let mixer = match &l.mixer {
            Mixer::Attention(a) => VulkanMixer::Attention(VulkanAttention::upload(gpu, a, ropes)?),
            Mixer::ShortConv(s) => VulkanMixer::ShortConv(VulkanShortConv::upload(gpu, s)?),
            Mixer::DeltaNet(d) => VulkanMixer::DeltaNet(VulkanDeltaNet::upload(gpu, d)?),
            Mixer::Mamba2(_) | Mixer::Parallel(_) => {
                return Err(unsupported("Mamba-2 and parallel token mixers"))
            }
        };
        let (Some(mixer_norm), Some(ffn_norm)) = (&l.mixer_norm, &l.ffn_norm) else {
            return Err(unsupported("layers without input norms"));
        };
        let ffn = match (&l.ffn, &l.experts) {
            (Some(f), None) => VulkanFeed::Dense(VulkanFfn::upload(gpu, f)?),
            (None, Some(m)) => VulkanFeed::Moe(VulkanMoe::upload(gpu, m)?),
            (Some(f), Some(m)) => {
                VulkanFeed::Shared(VulkanFfn::upload(gpu, f)?, VulkanMoe::upload(gpu, m)?)
            }
            (None, None) => return Err(unsupported("layers without a feed-forward")),
        };
        Ok(VulkanLayer {
            mixer_norm: gpu.vector(mixer_norm)?,
            mixer,
            mixer_out_norm: optional(&l.mixer_out_norm)?,
            ffn_norm: gpu.vector(ffn_norm)?,
            ffn,
            ffn_out_norm: optional(&l.ffn_out_norm)?,
            residual_scale: l.residual_scale.unwrap_or(1.0),
            out_scale: l.out_scale.unwrap_or(1.0),
        })
    }

    pub fn size_bytes(&self) -> usize {
        let mixer = match &self.mixer {
            VulkanMixer::Attention(a) => a.size_bytes(),
            VulkanMixer::ShortConv(s) => s.size_bytes(),
            VulkanMixer::DeltaNet(d) => d.size_bytes(),
        };
        mixer + self.ffn.size_bytes()
    }

    /// What the mixer needs of the work buffers.
    pub fn needs(&self) -> Needs {
        match &self.mixer {
            VulkanMixer::Attention(a) => a.needs(),
            VulkanMixer::ShortConv(s) => s.needs(),
            VulkanMixer::DeltaNet(d) => d.needs(),
        }
    }

    /// The widths of the activations the feed-forward reads.
    pub fn ffn_widths(&self) -> impl Iterator<Item = usize> + '_ {
        self.ffn.widths()
    }

    /// The experts a mixture layer routes over, and how many run per token.
    pub fn moe_shape(&self) -> Option<(usize, usize)> {
        match &self.ffn {
            VulkanFeed::Dense(_) => None,
            VulkanFeed::Moe(m) | VulkanFeed::Shared(_, m) => Some(m.shape()),
        }
    }

    /// Past positions an attention mixer sees, when not all.
    pub fn window(&self) -> Option<usize> {
        match &self.mixer {
            VulkanMixer::Attention(a) => a.shape.window,
            _ => None,
        }
    }

    /// Whether a rollback must replay the layer's kept tokens.
    pub fn replays(&self) -> bool {
        matches!(self.mixer, VulkanMixer::DeltaNet(_))
    }

    /// A fresh state for `positions` positions run in passes of up to
    /// `rows` tokens.
    pub fn new_state(
        &self,
        dev: &Rc<Device>,
        positions: usize,
        rows: usize,
    ) -> kinetic_vulkan::Result<LayerState> {
        Ok(match &self.mixer {
            VulkanMixer::Attention(a) => {
                LayerState::Attention(KvCache::new(dev, &a.shape, positions, rows)?)
            }
            VulkanMixer::ShortConv(s) => {
                LayerState::ShortConv(ConvRing::new(dev, s.channels(), s.taps, rows)?)
            }
            VulkanMixer::DeltaNet(d) => LayerState::DeltaNet(DeltaState::new(dev, &d.shape, rows)?),
        })
    }

    /// Empties `state` for a new sequence. Caches and rings need nothing:
    /// every slot a pass reads was written before it.
    pub fn clear(&self, state: &mut LayerState) -> R {
        match state {
            LayerState::DeltaNet(s) => s.clear(),
            LayerState::Attention(_) | LayerState::ShortConv(_) => Ok(()),
        }
    }

    /// Records the layer over the batch, whose input `w.h` holds the stream
    /// `w.x` after this layer's input norm. Leaves `w.h` holding `x` after
    /// the layer, normalized by `next`, the following layer's input norm or
    /// the output norm.
    pub fn record<'a>(
        &'a self,
        gpu: &Gpu,
        w: &'a Work,
        state: &LayerState,
        next: &DeviceBuffer<f32>,
        eps: f32,
        c: &mut Commands,
    ) -> R {
        match (&self.mixer, state) {
            (VulkanMixer::Attention(a), LayerState::Attention(cache)) => a.record(gpu, w, cache, c),
            (VulkanMixer::ShortConv(s), LayerState::ShortConv(ring)) => s.record(gpu, w, ring, c),
            (VulkanMixer::DeltaNet(d), LayerState::DeltaNet(st)) => d.record(gpu, w, st, c),
            _ => unreachable!("state does not belong to this layer"),
        }?;
        let k = &gpu.kernels;
        let add = |post: Option<&'a DeviceBuffer<f32>>, out_scale: f32| Residual {
            y: &w.y,
            post,
            scale: self.residual_scale,
            out_scale,
        };
        let mixed = add(self.mixer_out_norm.as_ref(), 1.0);
        k.norm(&w.batch, &w.x, Some(mixed), &self.ffn_norm, eps, &w.h, c)?;
        self.ffn.record(gpu, w, c)?;
        let fed = add(self.ffn_out_norm.as_ref(), self.out_scale);
        k.norm(&w.batch, &w.x, Some(fed), next, eps, &w.h, c)
    }

    /// Records what a rollback runs for this layer, if anything.
    pub fn record_replay(&self, gpu: &Gpu, w: &Work, state: &LayerState, c: &mut Commands) -> R {
        match (&self.mixer, state) {
            (VulkanMixer::DeltaNet(d), LayerState::DeltaNet(st)) => d.record_replay(gpu, w, st, c),
            _ => Ok(()),
        }
    }
}
