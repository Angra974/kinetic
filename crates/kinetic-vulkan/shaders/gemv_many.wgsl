// Shared part of the GEMV kernels for a pass of many tokens; a weight
// format appends `unpack` and `share`.
//
// y = W x for every row x of the batch, with one subgroup of 32 invocations
// per row of W. It runs the same sums as `gemv.wgsl`, with the tokens of the
// pass in groups the driver can keep in flight.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    stride: u32,
    offset: u32,
    head: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// The input quantized to int8, four per word, in blocks of 32 elements.
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
// Per block of 32: (d, d * sum of its int8 values).
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;

const ROWS_PER_GROUP: u32 = 4u;
const TOKENS: u32 = 4u;
const NIBBLES: u32 = 0x0F0F0F0Fu;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = (group.y * groups.x + group.x) * ROWS_PER_GROUP + slot;
    if row >= shape.rows {
        return;
    }
    let part = lane % LANES;
    let total = shape.rows * shape.blocks_per_row;
    let first = row * shape.blocks_per_row;
    var start = 0u;
    var count = batch[0];
    if shape.head != 0u {
        start = batch[2];
        count = batch[3];
    }

    let n = shape.blocks_per_row;
    let step = 32u / LANES;
    let out = shape.offset + row;
    for (var t0 = 0u; t0 < count; t0 += TOKENS) {
        let live = min(TOKENS, count - t0);
        var s0 = 0.0;
        var s1 = 0.0;
        var s2 = 0.0;
        var s3 = 0.0;
        var b = lane / LANES;
        for (; b < n; b += step) {
            let u = unpack(first + b, part);
            s0 += share(u, (start + t0) * n + b, part);
            if live > 1u {
                s1 += share(u, (start + t0 + 1u) * n + b, part);
            }
            if live > 2u {
                s2 += share(u, (start + t0 + 2u) * n + b, part);
            }
            if live > 3u {
                s3 += share(u, (start + t0 + 3u) * n + b, part);
            }
        }
        let r0 = subgroupAdd(s0);
        let r1 = subgroupAdd(s1);
        let r2 = subgroupAdd(s2);
        let r3 = subgroupAdd(s3);
        if lane == 0u {
            y[t0 * shape.stride + out] = r0;
            if live > 1u {
                y[(t0 + 1u) * shape.stride + out] = r1;
            }
            if live > 2u {
                y[(t0 + 2u) * shape.stride + out] = r2;
            }
            if live > 3u {
                y[(t0 + 3u) * shape.stride + out] = r3;
            }
        }
    }
}
