//! Device layout of weight matrices.
//!
//! Q6_K (210-byte), Q8_0 (34-byte), IQ3_S (110-byte) and IQ3_XXS (98-byte)
//! blocks are not 16-byte aligned, so kernels could not read them as 16-byte words. On the
//! device their fields are stored planar instead: each field of every block
//! in turn, so every plane starts aligned. Only IQ3_XXS needs its first
//! plane padded, its field being two bytes long.

use std::borrow::Cow;

use kinetic_quant::{Matrix, TensorType};

/// (offset, length) of each field in a block, in plane order.
const Q6_K_FIELDS: &[(usize, usize)] = &[(0, 128), (128, 64), (192, 16), (208, 2)];
const Q8_0_FIELDS: &[(usize, usize)] = &[(2, 32), (0, 2)];
const IQ3_XXS_FIELDS: &[(usize, usize)] = &[(0, 2), (2, 64), (66, 32)];
const IQ4_XS_FIELDS: &[(usize, usize)] = &[(0, 2), (2, 2), (4, 4), (8, 128)];
const IQ3_S_FIELDS: &[(usize, usize)] = &[(0, 2), (2, 64), (66, 8), (74, 32), (106, 4)];
const Q4_0_FIELDS: &[(usize, usize)] = &[(2, 16), (0, 2)];

/// The bytes of `m` as the GEMV kernels expect them.
pub(crate) fn device_layout<'a>(m: &Matrix<'a>) -> Cow<'a, [u8]> {
    let (fields, pad) = match m.ty {
        TensorType::Q6K => (Q6_K_FIELDS, false),
        TensorType::Q8_0 => (Q8_0_FIELDS, false),
        TensorType::IQ3XXS => (IQ3_XXS_FIELDS, true),
        TensorType::IQ4XS => (IQ4_XS_FIELDS, true),
        TensorType::IQ3S => (IQ3_S_FIELDS, true),
        TensorType::Q4_0 => (Q4_0_FIELDS, false),
        _ => return Cow::Borrowed(m.data),
    };
    Cow::Owned(planar(m.data, m.ty.block_bytes(), fields, pad))
}

/// The fields of every block in turn; with `pad`, each field of all blocks
/// is followed by zeroes up to a whole 16-byte word.
fn planar(data: &[u8], block_bytes: usize, fields: &[(usize, usize)], pad: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    for &(start, len) in fields {
        for block in data.chunks_exact(block_bytes) {
            out.extend_from_slice(&block[start..start + len]);
        }
        if pad {
            out.resize(out.len().next_multiple_of(16), 0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_planes_hold_quants_then_scales() {
        let blocks: Vec<u8> = (0..68).collect();
        let m = Matrix::new(TensorType::Q8_0, &blocks, 1, 64);
        let out = device_layout(&m);
        let expected: Vec<u8> = (2..34).chain(36..68).chain([0, 1, 34, 35]).collect();
        assert_eq!(*out, expected[..]);
    }

    #[test]
    fn iq3_xxs_planes_hold_d_then_indices_then_scales() {
        let blocks: Vec<u8> = (0..196).collect();
        let m = Matrix::new(TensorType::IQ3XXS, &blocks, 1, 512);
        let out = device_layout(&m);
        let mut expected: Vec<u8> = (0..2).chain(98..100).collect();
        expected.resize(16, 0);
        expected.extend(2..66);
        expected.extend(100..164);
        expected.extend(66..98);
        expected.extend(164..196);
        assert_eq!(*out, expected[..]);
    }

    #[test]
    fn aligned_formats_are_borrowed() {
        let blocks = [0u8; 144];
        let m = Matrix::new(TensorType::Q4K, &blocks, 1, 256);
        assert!(matches!(device_layout(&m), Cow::Borrowed(_)));
    }
}
