// The gated feed-forward activation of each token row, written as a GEMV
// input: out = act(gate) * up, where a row of `gu` holds the `len` gate
// values followed by the `len` up values. `act` is SiLU, or GELU in its tanh
// approximation when `gelu` is set, which makes the block a GeGLU.

struct Shape {
    len: u32,
    gelu: u32,
}

// GELU in its tanh approximation, as the CPU kernels compute it.
fn activate(g: f32) -> f32 {
    if shape.gelu == 0u {
        return g / (1.0 + exp(-g));
    }
    return 0.5 * g * (1.0 + tanh(0.7978846 * (g + 0.044715 * g * g * g)));
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> batch: array<u32>;
@group(0) @binding(1) var<storage, read> gu: array<f32>;
@group(0) @binding(2) var<storage, read_write> ox: array<f32>;
@group(0) @binding(3) var<storage, read_write> oq: array<u32>;
@group(0) @binding(4) var<storage, read_write> od: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let i = group.x * 256u + tid;
    let row = group.y;
    // `len` is a multiple of 32, so subgroups leave whole.
    if row >= batch[0] || i >= shape.len {
        return;
    }
    let g = gu[2u * row * shape.len + i];
    let u = gu[(2u * row + 1u) * shape.len + i];
    put_q8(row * shape.len + i, activate(g) * u, lane);
}
