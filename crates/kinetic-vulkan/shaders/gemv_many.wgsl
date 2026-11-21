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
const TOKENS: u32 = 32u;
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
        var s16 = 0.0;
        var s17 = 0.0;
        var s18 = 0.0;
        var s19 = 0.0;
        var s20 = 0.0;
        var s21 = 0.0;
        var s22 = 0.0;
        var s23 = 0.0;
        var s24 = 0.0;
        var s25 = 0.0;
        var s26 = 0.0;
        var s27 = 0.0;
        var s28 = 0.0;
        var s29 = 0.0;
        var s30 = 0.0;
        var s31 = 0.0;
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
            if live > 16u {
                s16 += share(u, (start + t0 + 16u) * n + b, part);
            }
            if live > 17u {
                s17 += share(u, (start + t0 + 17u) * n + b, part);
            }
            if live > 18u {
                s18 += share(u, (start + t0 + 18u) * n + b, part);
            }
            if live > 19u {
                s19 += share(u, (start + t0 + 19u) * n + b, part);
            }
            if live > 20u {
                s20 += share(u, (start + t0 + 20u) * n + b, part);
            }
            if live > 21u {
                s21 += share(u, (start + t0 + 21u) * n + b, part);
            }
            if live > 22u {
                s22 += share(u, (start + t0 + 22u) * n + b, part);
            }
            if live > 23u {
                s23 += share(u, (start + t0 + 23u) * n + b, part);
            }
            if live > 24u {
                s24 += share(u, (start + t0 + 24u) * n + b, part);
            }
            if live > 25u {
                s25 += share(u, (start + t0 + 25u) * n + b, part);
            }
            if live > 26u {
                s26 += share(u, (start + t0 + 26u) * n + b, part);
            }
            if live > 27u {
                s27 += share(u, (start + t0 + 27u) * n + b, part);
            }
            if live > 28u {
                s28 += share(u, (start + t0 + 28u) * n + b, part);
            }
            if live > 29u {
                s29 += share(u, (start + t0 + 29u) * n + b, part);
            }
            if live > 30u {
                s30 += share(u, (start + t0 + 30u) * n + b, part);
            }
            if live > 31u {
                s31 += share(u, (start + t0 + 31u) * n + b, part);
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
        let r16 = subgroupAdd(s16);
        let r17 = subgroupAdd(s17);
        let r18 = subgroupAdd(s18);
        let r19 = subgroupAdd(s19);
        let r20 = subgroupAdd(s20);
        let r21 = subgroupAdd(s21);
        let r22 = subgroupAdd(s22);
        let r23 = subgroupAdd(s23);
        let r24 = subgroupAdd(s24);
        let r25 = subgroupAdd(s25);
        let r26 = subgroupAdd(s26);
        let r27 = subgroupAdd(s27);
        let r28 = subgroupAdd(s28);
        let r29 = subgroupAdd(s29);
        let r30 = subgroupAdd(s30);
        let r31 = subgroupAdd(s31);
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
            if live > 16u {
                y[(t0 + 16u) * shape.stride + out] = r16;
            }
            if live > 17u {
                y[(t0 + 17u) * shape.stride + out] = r17;
            }
            if live > 18u {
                y[(t0 + 18u) * shape.stride + out] = r18;
            }
            if live > 19u {
                y[(t0 + 19u) * shape.stride + out] = r19;
            }
            if live > 20u {
                y[(t0 + 20u) * shape.stride + out] = r20;
            }
            if live > 21u {
                y[(t0 + 21u) * shape.stride + out] = r21;
            }
            if live > 22u {
                y[(t0 + 22u) * shape.stride + out] = r22;
            }
            if live > 23u {
                y[(t0 + 23u) * shape.stride + out] = r23;
            }
            if live > 24u {
                y[(t0 + 24u) * shape.stride + out] = r24;
            }
            if live > 25u {
                y[(t0 + 25u) * shape.stride + out] = r25;
            }
            if live > 26u {
                y[(t0 + 26u) * shape.stride + out] = r26;
            }
            if live > 27u {
                y[(t0 + 27u) * shape.stride + out] = r27;
            }
            if live > 28u {
                y[(t0 + 28u) * shape.stride + out] = r28;
            }
            if live > 29u {
                y[(t0 + 29u) * shape.stride + out] = r29;
            }
            if live > 30u {
                y[(t0 + 30u) * shape.stride + out] = r30;
            }
            if live > 31u {
                y[(t0 + 31u) * shape.stride + out] = r31;
            }
        }
    }
}
