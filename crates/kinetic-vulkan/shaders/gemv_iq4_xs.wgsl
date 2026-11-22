// block_iq4_xs: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128], planar:
// each of the four fields of every block in turn, the short ones padded to
// whole 16-byte words. Every 32 values have a 6-bit scale, taken from
// scales_l and scales_h, and are read from 16 bytes of qs: the low nibbles
// are the first 16 values, the high nibbles the next 16, each a code of
// IQ4NL.

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

fn block_dot(base: u32, total: u32, blk: u32, b: u32, part: u32) -> f32 {
    // The planes: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128].
    let short = (total + 7u) / 8u;
    let l_base = 2u * short;
    let qs_base = l_base + (total + 3u) / 4u;
    let d = f16_at(base + 0u, blk);
    let h = u16_at(base + short, blk);
    let l = w[base + l_base + blk / 4u];
    let hw = bitcast<u32>(h) >> (2u * part) & 3u;
    let lw = (l[blk % 4u] >> (8u * (part / 2u) + 4u * (part % 2u))) & 0x0Fu;
    let dl = d * f32(i32(lw | (hw << 4u)) - 32);
    let qs = w[base + qs_base + 8u * blk + part];
    let e0 = xq[b * 16u + 2u * part];
    let e1 = xq[b * 16u + 2u * part + 1u];
    var sum = 0i;
    for (var k = 0u; k < 4u; k++) {
        let word = qs[k];
        var lo = 0u;
        var hi = 0u;
        for (var t = 0u; t < 4u; t++) {
            let byte = (word >> (8u * t)) & 0xFFu;
            lo |= IQ4NL[byte & 0x0Fu] << (8u * t);
            hi |= IQ4NL[byte >> 4u] << (8u * t);
        }
        sum += dot4I8Packed(lo, e0[k]) + dot4I8Packed(hi, e1[k]);
    }
    return dl * xd[b * 8u + part].x * f32(sum);
}
