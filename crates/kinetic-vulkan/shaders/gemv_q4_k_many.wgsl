// block_q4_K: d, dmin, scales[12], qs[128]; 144 bytes, nine words.
// `unpack` decodes once what `block_dot` redoes for every token.

struct Unpacked {
    lo: vec4<u32>,
    hi: vec4<u32>,
    d: f32,
    dmin: f32,
    a: vec2<f32>,
    c: vec2<f32>,
}

// Four signed int8 dot products of 16 values, as in `gemv.wgsl`.
fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

fn unpack(blk: u32, j: u32) -> Unpacked {
    let head = w[blk * 9u];
    let q = w[blk * 9u + 1u + j];
    let s = 2u * (j / 2u);
    let dm = unpack2x16float(head.x);
    return Unpacked(
        q & vec4<u32>(NIBBLES),
        (q >> vec4<u32>(4u)) & vec4<u32>(NIBBLES),
        dm.x,
        dm.y,
        k4_scale_min(s, head.y, head.z, head.w),
        k4_scale_min(s + 1u, head.y, head.z, head.w),
    );
}

// The same term, in the same order, as `k4_combine`.
fn share(u: Unpacked, b: u32, j: u32) -> f32 {
    let s = 2u * (j / 2u);
    let xi = k4_input(b, j);
    let lo = f32(dot16(u.lo, xq[xi]));
    let hi = f32(dot16(u.hi, xq[xi + 2u]));
    let x0 = xd[b * 8u + s];
    let x1 = xd[b * 8u + s + 1u];
    var r = u.d * (u.a.x * x0.x * lo + u.c.x * x1.x * hi);
    if j % 2u == 0u {
        r -= u.dmin * (u.a.y * x0.y + u.c.y * x1.y);
    }
    return r;
}
