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
const TOKENS: u32 = 16u;
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
        var s4 = 0.0;
        var s5 = 0.0;
        var s6 = 0.0;
        var s7 = 0.0;
        var s8 = 0.0;
        var s9 = 0.0;
        var s10 = 0.0;
        var s11 = 0.0;
        var s12 = 0.0;
        var s13 = 0.0;
        var s14 = 0.0;
        var s15 = 0.0;
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
            if live > 4u {
                s4 += share(u, (start + t0 + 4u) * n + b, part);
            }
            if live > 5u {
                s5 += share(u, (start + t0 + 5u) * n + b, part);
            }
            if live > 6u {
                s6 += share(u, (start + t0 + 6u) * n + b, part);
            }
            if live > 7u {
                s7 += share(u, (start + t0 + 7u) * n + b, part);
            }
            if live > 8u {
                s8 += share(u, (start + t0 + 8u) * n + b, part);
            }
            if live > 9u {
                s9 += share(u, (start + t0 + 9u) * n + b, part);
            }
            if live > 10u {
                s10 += share(u, (start + t0 + 10u) * n + b, part);
            }
            if live > 11u {
                s11 += share(u, (start + t0 + 11u) * n + b, part);
            }
            if live > 12u {
                s12 += share(u, (start + t0 + 12u) * n + b, part);
            }
            if live > 13u {
                s13 += share(u, (start + t0 + 13u) * n + b, part);
            }
            if live > 14u {
                s14 += share(u, (start + t0 + 14u) * n + b, part);
            }
            if live > 15u {
                s15 += share(u, (start + t0 + 15u) * n + b, part);
            }
        }
        let r0 = subgroupAdd(s0);
        let r1 = subgroupAdd(s1);
        let r2 = subgroupAdd(s2);
        let r3 = subgroupAdd(s3);
        let r4 = subgroupAdd(s4);
        let r5 = subgroupAdd(s5);
        let r6 = subgroupAdd(s6);
        let r7 = subgroupAdd(s7);
        let r8 = subgroupAdd(s8);
        let r9 = subgroupAdd(s9);
        let r10 = subgroupAdd(s10);
        let r11 = subgroupAdd(s11);
        let r12 = subgroupAdd(s12);
        let r13 = subgroupAdd(s13);
        let r14 = subgroupAdd(s14);
        let r15 = subgroupAdd(s15);
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
            if live > 4u {
                y[(t0 + 4u) * shape.stride + out] = r4;
            }
            if live > 5u {
                y[(t0 + 5u) * shape.stride + out] = r5;
            }
            if live > 6u {
                y[(t0 + 6u) * shape.stride + out] = r6;
            }
            if live > 7u {
                y[(t0 + 7u) * shape.stride + out] = r7;
            }
            if live > 8u {
                y[(t0 + 8u) * shape.stride + out] = r8;
            }
            if live > 9u {
                y[(t0 + 9u) * shape.stride + out] = r9;
            }
            if live > 10u {
                y[(t0 + 10u) * shape.stride + out] = r10;
            }
            if live > 11u {
                y[(t0 + 11u) * shape.stride + out] = r11;
            }
            if live > 12u {
                y[(t0 + 12u) * shape.stride + out] = r12;
            }
            if live > 13u {
                y[(t0 + 13u) * shape.stride + out] = r13;
            }
            if live > 14u {
                y[(t0 + 14u) * shape.stride + out] = r14;
            }
            if live > 15u {
                y[(t0 + 15u) * shape.stride + out] = r15;
            }
        }
    }
}
