// Shared part of the grouped GEMV of a mixture of experts; build.rs appends
// one weight format to it, which defines `LANES` and `block_dot`.
//
// One row of the pass is a (token, pick) slot: the weights of a slot's row
// come from the expert the router picked for it, `moe[1 + slot]`, whose
// matrix starts that many expert strides into the weights. `moe[0]` is the
// number of slots. A slot reads input row `slot / share`, which is the
// token for the first projections and the slot itself for the one after
// the activation.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    stride: u32,
    offset: u32,
    share: u32,
    /// 16-byte words of one expert's matrix, the stride between experts.
    words: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
// The slot count, then the word its expert's matrix starts at.
@group(0) @binding(5) var<storage, read> moe: array<u32>;

const ROWS_PER_GROUP: u32 = 4u;
const NIBBLES: u32 = 0x0F0F0F0Fu;

// Four signed int8 dot products of 16 values.
fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

// Element `i` of an f16 plane that starts at word `base` of the weights.
fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(subgroup_id) lane_group: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let line = (group.y * groups.x + group.x) * ROWS_PER_GROUP + lane_group;
    let slots = moe[0];
    if line >= slots * shape.rows {
        return;
    }
    let slot = line / shape.rows;
    let row = line % shape.rows;
    let base = moe[1u + slot] * shape.words;
    let src = slot / shape.share;

    let part = lane % LANES;
    let n = shape.blocks_per_row;
    let total = shape.rows * n;
    let first = row * n;
    let xb = src * n;
    let step = 32u / LANES;
    var sum = 0.0;
    var b = lane / LANES;
    // Two blocks per step keep two weight loads in flight.
    for (; b + step < n; b += 2u * step) {
        sum += block_dot(base, total, first + b, xb + b, part)
            + block_dot(base, total, first + b + step, xb + b + step, part);
    }
    if b < n {
        sum += block_dot(base, total, first + b, xb + b, part);
    }
    sum = subgroupAdd(sum);
    if lane == 0u {
        y[slot * shape.stride + shape.offset + row] = sum;
    }
}
