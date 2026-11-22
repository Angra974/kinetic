// block_q4_0 for a pass of many tokens; `unpack` decodes once what
// `block_dot` redoes for every token.

const LANES: u32 = 8u;

struct Unpacked {
    q: u32,
    d: f32,
}

fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

fn blocks_total() -> u32 {
    return shape.rows * shape.blocks_per_row;
}

fn unpack(blk: u32, j: u32) -> Unpacked {
    let low = j < 4u;
    let word = w[blk][j % 4u];
    var q = 0u;
    for (var t = 0u; t < 4u; t++) {
        let byte = (word >> (8u * t)) & 0xFFu;
        let nibble = select(byte >> 4u, byte & 0x0Fu, low);
        q |= ((nibble - 8u) & 0xFFu) << (8u * t);
    }
    // The nibble plane is one 16-byte word a block, so the f16 plane starts
    // at word `blocks_total`.
    return Unpacked(q, f16_at(blocks_total(), blk));
}

// The same term, in the same order, as `block_dot`.
fn share(u: Unpacked, b: u32, j: u32) -> f32 {
    let e = xq[b * 2u + j / 4u];
    return u.d * xd[b].x * f32(dot4I8Packed(u.q, e[j % 4u]));
}
