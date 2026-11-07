//! Hand-computed known vectors for the GGML block formats.
//!
//! These are the only tests that can catch a misread spec: every input byte is
//! chosen here, the expected float is derived on paper below, and the test
//! asserts the exact value. If the 6-bit unpacking, the nibble order or the qh
//! bit planes are wrong, these fail before any GPU is involved.

use hipml_core::ops;
use hipml_core::quant::{
    dequant_q4_k, dequant_q6_k, dequant_q8_0, f16_to_f32, gemv_q4_k, gemv_q6_k, TensorType,
    Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K,
};

#[test]
fn f16_matches_known_bit_patterns() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x8000), -0.0);
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0xBC00), -1.0);
    assert_eq!(f16_to_f32(0x3800), 0.5);
    assert_eq!(f16_to_f32(0x4000), 2.0);
    assert_eq!(f16_to_f32(0x1400), 2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x1000), 2.0f32.powi(-11));
    // Largest subnormal half is 1023 * 2^-24; smallest normal is 2^-14.
    assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
    assert!(f16_to_f32(0x7C00).is_infinite());
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    assert!(f16_to_f32(0x7E00).is_nan());
}

/// Q4_K block built by hand.
///
/// Header:
///   d      = 1.0    (fp16 0x3C00)
///   dmin   = 0.5    (fp16 0x3800)
///   scales = [65, 130, 3, 196, 5, 70, 135, 8, 41, 58, 75, 92]
///
/// Unpacking the 6-bit fields (q4_k_scale_min):
///   j:      0    1    2    3    4    5    6    7
///   scale:  1    2    3    4   25   42   11   60
///   min:    5    6    7    8    2   19   36    5
/// and with d=1, dmin=0.5 the effective min is min/2.
///
/// Quants are four 32-byte groups: 0x21, 0x43, 0x65, 0x87, so sub-block j
/// uses q = (1,2), (3,4), (5,6), (7,8) for its low/high nibbles.
///
///   j=0:  1*1  - 0.5*5  = -1.5
///   j=1:  2*2  - 0.5*6  =  1.0
///   j=2:  3*3  - 0.5*7  =  5.5
///   j=3:  4*4  - 0.5*8  = 12.0
///   j=4: 25*5  - 0.5*2  = 124.0
///   j=5: 42*6  - 0.5*19 = 242.5
///   j=6: 11*7  - 0.5*36 = 59.0
///   j=7: 60*8  - 0.5*5  = 477.5
fn q4_k_hand_block() -> ([u8; Q4_K_BYTES], [f32; 8]) {
    let mut b = [0u8; Q4_K_BYTES];
    b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    b[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
    b[4..16].copy_from_slice(&[65, 130, 3, 196, 5, 70, 135, 8, 41, 58, 75, 92]);
    for (g, byte) in [0x21u8, 0x43, 0x65, 0x87].iter().enumerate() {
        for i in 0..32 {
            b[16 + g * 32 + i] = *byte;
        }
    }
    (b, [-1.5, 1.0, 5.5, 12.0, 124.0, 242.5, 59.0, 477.5])
}

#[test]
fn q4_k_hand_computed_vector() {
    let (block, expected) = q4_k_hand_block();
    let mut out = [0.0f32; QK_K];
    dequant_q4_k(&block, &mut out);
    for j in 0..8 {
        for l in 0..32 {
            assert_eq!(
                out[j * 32 + l],
                expected[j],
                "sub-block {j} value {l}: got {}, want {}",
                out[j * 32 + l],
                expected[j]
            );
        }
    }
}

/// Q6_K block built by hand.
///
///   d      = 0.5 (fp16 0x3800)
///   ql     = 0x21 everywhere -> low nibble 1, high nibble 2
///   qh     = 0x1B everywhere -> 2-bit planes 3, 2, 1, 0
///   scales = [1..16]
///
/// Group 0/1 read the low nibble, groups 2/3 the high nibble; qh planes are
/// shifted by 0, 2, 4, 6 bits. So q-32 is 17, 1, -14, -30 for groups 0..3.
/// Sub-block j covers positions j*16..j*16+15 and uses scale[j]:
///
///   j=0:  0.5*1*17  =   8.5     j=8:  0.5*9*17  =  76.5
///   j=1:  0.5*2*17  =  17.0     j=9:  0.5*10*17 =  85.0
///   j=2:  0.5*3*1   =   1.5     j=10: 0.5*11*1  =   5.5
///   j=3:  0.5*4*1   =   2.0     j=11: 0.5*12*1  =   6.0
///   j=4:  0.5*5*-14 = -35.0     j=12: 0.5*13*-14= -91.0
///   j=5:  0.5*6*-14 = -42.0     j=13: 0.5*14*-14= -98.0
///   j=6:  0.5*7*-30 = -105.0    j=14: 0.5*15*-30= -225.0
///   j=7:  0.5*8*-30 = -120.0    j=15: 0.5*16*-30= -240.0
fn q6_k_hand_block() -> ([u8; Q6_K_BYTES], [f32; 16]) {
    let mut b = [0u8; Q6_K_BYTES];
    b[0..128].fill(0x21);
    b[128..192].fill(0x1B);
    for j in 0..16 {
        b[192 + j] = (j + 1) as u8;
    }
    b[208..210].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    (
        b,
        [
            8.5, 17.0, 1.5, 2.0, -35.0, -42.0, -105.0, -120.0, 76.5, 85.0, 5.5, 6.0, -91.0, -98.0,
            -225.0, -240.0,
        ],
    )
}

#[test]
fn q6_k_hand_computed_vector() {
    let (block, expected) = q6_k_hand_block();
    let mut out = [0.0f32; QK_K];
    dequant_q6_k(&block, &mut out);
    for j in 0..16 {
        for l in 0..16 {
            assert_eq!(
                out[j * 16 + l],
                expected[j],
                "sub-block {j} value {l}: got {}, want {}",
                out[j * 16 + l],
                expected[j]
            );
        }
    }
}

#[test]
fn q8_0_hand_computed_vector() {
    let mut b = [0u8; Q8_0_BYTES];
    b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    for i in 0..32 {
        b[2 + i] = (i as i8 + 1) as u8;
    }
    let mut out = [0.0f32; 32];
    dequant_q8_0(&b, &mut out);
    for i in 0..32 {
        assert_eq!(out[i], 0.5 * (i as f32 + 1.0));
    }
}

#[test]
fn q4_k_gemv_reduces_the_dequantised_block() {
    let (block, expected) = q4_k_hand_block();
    let x = vec![1.0f32; QK_K];
    let y = gemv_q4_k(&block, &x, 1, QK_K);
    let want: f32 = expected.iter().sum::<f32>() * 32.0;
    assert_eq!(y.len(), 1);
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );

    // A one-hot input selects one column of the dequantised row.
    let mut hot = vec![0.0f32; QK_K];
    hot[7] = 1.0;
    assert_eq!(gemv_q4_k(&block, &hot, 1, QK_K), vec![expected[0]]);
}

#[test]
fn q6_k_gemv_reduces_the_dequantised_block() {
    let (block, expected) = q6_k_hand_block();
    let x = vec![1.0f32; QK_K];
    let y = gemv_q6_k(&block, &x, 1, QK_K);
    let want: f32 = expected.iter().sum::<f32>() * 16.0;
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );

    let mut hot = vec![0.0f32; QK_K];
    hot[250] = 1.0; // last sub-block (j=15), offset 250-240=10
    assert_eq!(gemv_q6_k(&block, &hot, 1, QK_K), vec![expected[15]]);
}

#[test]
fn tensor_type_sizes_match_the_ggml_blocks() {
    assert_eq!(TensorType::Q4K.nbytes(256), Some(144));
    assert_eq!(TensorType::Q6K.nbytes(256), Some(210));
    assert_eq!(TensorType::Q5K.nbytes(256), Some(176));
    assert_eq!(TensorType::Q8_0.nbytes(32), Some(34));
    assert_eq!(TensorType::F32.nbytes(10), Some(40));
    assert_eq!(TensorType::Q4K.nbytes(255), None);
    assert_eq!(TensorType::Unsupported(99).nbytes(256), None);
    assert_eq!(TensorType::from_id(12), TensorType::Q4K);
    assert_eq!(TensorType::from_id(14), TensorType::Q6K);
}

#[test]
fn f32_gemv_still_works_for_the_parity_baseline() {
    // Guard against a quant module refactor touching the f32 path.
    let w = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![1.0, 1.0];
    assert_eq!(ops::gemv_f32(&w, &x, 2, 2), vec![3.0, 7.0]);
}
