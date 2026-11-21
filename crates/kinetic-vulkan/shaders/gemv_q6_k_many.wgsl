// block_q6_K is 210 bytes, so blocks are only 2-byte aligned. The device copy
// is planar: all ql[128], then all qh[64], all scales[16], all f16 d.
//
// A block is two halves of 128 elements; in half n, element 32g + l (g < 4,
// l < 32) takes its low four bits from ql[64n + l + 32(g & 1)] (low nibble
// for g < 2, high for g >= 2) and its top two from bits 2g..2g+1 of
// qh[32n + l]. Each run of 16 elements has its own int8 scale.
//
// Eight invocations share a block; invocation j = 4n + k loads ql bytes
// 64n + 16k, which hold elements l = 16(k & 1) .. +16 of group g = k / 2
// (low nibbles) and of group g + 2 (high nibbles).
// `unpack` decodes once what `block_dot` redoes for every token.

const LANES: u32 = 8u;

struct Unpacked {
    lo: vec4<u32>,
    hi: vec4<u32>,
    d: f32,
    a: f32,
    c: f32,
}

// Four signed int8 dot products of 16 values, as in `gemv.wgsl`.
fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

// Signed byte `i` of a word.
fn sbyte(v: vec4<u32>, i: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(v[i / 4u]), 8u * (i % 4u), 8u));
}

fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

// The planes of a planar layout start after the blocks of every row.
fn blocks_total() -> u32 {
    return shape.rows * shape.blocks_per_row;
}

fn unpack(blk: u32, j: u32) -> Unpacked {
    let total = blocks_total();
    let n = j / 4u;
    let k = j % 4u;
    let g = k / 2u;
    let lq = w[blk * 8u + j];
    let hq = w[total * 8u + blk * 4u + 2u * n + (k & 1u)];
    let low2 = vec4<u32>(0x03030303u);
    let lo = (lq & vec4<u32>(NIBBLES)) | (((hq >> vec4<u32>(2u * g)) & low2) << vec4<u32>(4u));
    let hi = ((lq >> vec4<u32>(4u)) & vec4<u32>(NIBBLES))
        | (((hq >> vec4<u32>(2u * g + 4u)) & low2) << vec4<u32>(4u));
    let scales = w[total * 12u + blk];
    let sc = 8u * n + (k & 1u) + 2u * g;
    return Unpacked(lo, hi, f16_at(total * 13u, blk), sbyte(scales, sc), sbyte(scales, sc + 4u));
}

// The same term, in the same order, as `block_dot`.
fn share(u: Unpacked, b: u32, j: u32) -> f32 {
    let e = b * 256u + 128u * (j / 4u) + 32u * ((j % 4u) / 2u) + 16u * (j & 1u);
    let x0 = xq[e / 16u];
    let x1 = xq[e / 16u + 4u];
    // Quants are stored as q + 32; dotting 32 with the input takes it back out.
    let bias = vec4<u32>(0x20202020u);
    let s0 = dot16(u.lo, x0) - dot16(bias, x0);
    let s1 = dot16(u.hi, x1) - dot16(bias, x1);
    let r = u.a * xd[e / 32u].x * f32(s0) + u.c * xd[e / 32u + 2u].x * f32(s1);
    return u.d * r;
}
