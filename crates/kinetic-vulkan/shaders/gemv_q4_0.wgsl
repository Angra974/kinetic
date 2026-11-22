// block_q4_0: f16 d then 16 bytes of nibbles, planar: every block's 16
// nibble bytes, then every block's f16. Each byte holds values j and j + 16,
// less eight; four invocations take the first sixteen values and four the
// next, four values each. The kernel's block is the format's block of 32.

const LANES: u32 = 8u;

fn block_dot(base: u32, total: u32, blk: u32, b: u32, part: u32) -> f32 {
    // The nibble plane is one 16-byte word a block, so the f16 plane starts
    // at word `total`.
    let d = f16_at(total, blk);
    let low = part < 4u;
    let at = part % 4u;
    let q = w[base + blk][at];
    var packed = 0u;
    for (var t = 0u; t < 4u; t++) {
        let byte = (q >> (8u * t)) & 0xFFu;
        let nibble = select(byte >> 4u, byte & 0x0Fu, low);
        packed |= ((nibble - 8u) & 0xFFu) << (8u * t);
    }
    let e = xq[b * 2u + part / 4u];
    return d * xd[b].x * f32(dot4I8Packed(packed, e[at]));
}
