// block_iq4_xs for a pass of many tokens; `unpack` decodes once what
// `block_dot` redoes for every token.

const LANES: u32 = 8u;

/// The sixteen values a code stands for, as int8 in a word.
const IQ4NL: array<u32, 16> = array<u32, 16>(
    0x00000081u, 0x00000098u, 0x000000ADu, 0x000000BFu,
    0x000000CFu, 0x000000DDu, 0x000000EAu, 0x000000F6u,
    0x00000001u, 0x0000000Du, 0x00000019u, 0x00000026u,
    0x00000035u, 0x00000045u, 0x00000059u, 0x00000071u,
);

/// Element `i` of a u16 plane that starts at word `base`.
fn u16_at(base: u32, i: u32) -> u32 {
    let word = w[base + i / 8u][(i / 2u) % 4u];
    return select(word & 0xFFFFu, word >> 16u, (i & 1u) == 1u);
}

fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}

struct Unpacked {
    lo: array<u32, 4>,
    hi: array<u32, 4>,
    dl: f32,
}

fn blocks_total() -> u32 {
    return shape.rows * shape.blocks_per_row;
}

fn unpack(blk: u32, j: u32) -> Unpacked {
    // The planes: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128].
    let total = blocks_total();
    let short = (total + 7u) / 8u;
    let l_base = 2u * short;
    let qs_base = l_base + (total + 3u) / 4u;
    let hw = u16_at(short, blk) >> (2u * j) & 3u;
    let l = w[l_base + blk / 4u];
    let lw = (l[blk % 4u] >> (8u * (j / 2u) + 4u * (j % 2u))) & 0x0Fu;
    let qs = w[qs_base + 8u * blk + j];
    var u: Unpacked;
    u.dl = f16_at(0u, blk) * f32(i32(lw | (hw << 4u)) - 32);
    for (var k = 0u; k < 4u; k++) {
        let word = qs[k];
        var lo = 0u;
        var hi = 0u;
        for (var t = 0u; t < 4u; t++) {
            let byte = (word >> (8u * t)) & 0xFFu;
            lo |= IQ4NL[byte & 0x0Fu] << (8u * t);
            hi |= IQ4NL[byte >> 4u] << (8u * t);
        }
        u.lo[k] = lo;
        u.hi[k] = hi;
    }
    return u;
}

// The same term, in the same order, as `block_dot`.
fn share(u: Unpacked, b: u32, j: u32) -> f32 {
    let e0 = xq[b * 16u + 2u * j];
    let e1 = xq[b * 16u + 2u * j + 1u];
    var sum = 0i;
    for (var k = 0u; k < 4u; k++) {
        sum += dot4I8Packed(u.lo[k], e0[k]) + dot4I8Packed(u.hi[k], e1[k]);
    }
    return u.dl * xd[b * 8u + j].x * f32(sum);
}
