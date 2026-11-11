// SPDX-License-Identifier: Apache-2.0
//! Hand-computed known vectors for the GGML block formats.
//!
//! These are the only tests that can catch a misread spec: every input byte is
//! chosen here, the expected float is derived on paper below, and the test
//! asserts the exact value. If the 6-bit unpacking, the nibble order or the qh
//! bit planes are wrong, these fail before any GPU is involved.

use hipml_core::ops;
use hipml_core::quant::{
    dequant_iq3_s, dequant_iq4_nl, dequant_iq4_xs, dequant_q3_k, dequant_q4_k, dequant_q6_k,
    dequant_q8_0, f16_to_f32, gemv_iq3_s, gemv_iq4_nl, gemv_iq4_xs, gemv_q3_k, gemv_q4_k,
    gemv_q6_k, TensorType, IQ3S_GRID, IQ3_S_BYTES, IQ4_NL_BYTES, IQ4_NL_VALUES, IQ4_XS_BYTES,
    Q3_K_BYTES, Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K,
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

/// IQ4_NL block built by hand:
///   d  = 0.5 (fp16 0x3800)
///   qs = ((j*7+3) & 0xf) | (((j*5+9) & 0xf) << 4) for j in 0..16
///
/// The expected f32 bytes below do not come from the Rust: they are an
/// independent Python transcription of `dequantize_row_iq4_nl`
/// (docs/reference/ggml-quants.c:2741) and the `kvalues_iq4nl` table from
/// upstream ggml-common.h. Both nibble halves are exercised.
fn iq4_nl_hand_block() -> [u8; IQ4_NL_BYTES] {
    let mut b = [0u8; IQ4_NL_BYTES];
    b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    for j in 0..16 {
        b[2 + j] = ((j as u8 * 7 + 3) & 0xF) | (((j as u8 * 5 + 9) & 0xF) << 4);
    }
    b
}

#[test]
fn iq4_nl_hand_computed_vector() {
    let expected: [u32; 32] = [
        0xC2020000, 0x41480000, 0xC2500000, 0x3F000000, 0x42620000, 0xC1300000, 0x420A0000,
        0xC1C40000, 0x41980000, 0xC2260000, 0x40D00000, 0xC27E0000, 0xC0A00000, 0x42320000,
        0xC18C0000, 0x41D40000, 0x40D00000, 0x42320000, 0xC2020000, 0x3F000000, 0x420A0000,
        0xC2260000, 0xC0A00000, 0x41D40000, 0xC2500000, 0xC1300000, 0x41980000, 0xC27E0000,
        0xC18C0000, 0x41480000, 0x42620000, 0xC1C40000,
    ];
    let mut out = [0.0f32; IQ4_NL_VALUES];
    dequant_iq4_nl(&iq4_nl_hand_block(), &mut out);
    for j in 0..IQ4_NL_VALUES {
        assert_eq!(
            out[j].to_bits(),
            expected[j],
            "value {j}: got {}, want {}",
            out[j],
            f32::from_bits(expected[j])
        );
    }
}

/// IQ4_XS super-block built by hand:
///   d        = 0.25 (fp16 0x3400)
///   scales_h = 0xFA50, scales_l = [0x50, 0xF0, 0xF0, 0xF0]
///   qs[k]    = ((k*13+5) & 0xf) | (((k*11+7) & 0xf) << 4)
///
/// The six-bit `ls` per 32-value sub-block comes out as
///   0, 5, 16, 31, 32, 47, 48, 63
/// so `dl = d * (ls - 32)` is -8, -6.75, -4, -0.25, 0, 3.75, 4, 7.75 —
/// negative scales, an exactly zero sub-block, and positive scales in one
/// block. Expected bytes again come from the Python transcription.
fn iq4_xs_hand_block() -> [u8; IQ4_XS_BYTES] {
    let mut b = [0u8; IQ4_XS_BYTES];
    b[0..2].copy_from_slice(&0x3400u16.to_le_bytes()); // d = 0.25
    b[2..4].copy_from_slice(&0xFA50u16.to_le_bytes()); // scales_h
    b[4..8].copy_from_slice(&[0x50, 0xF0, 0xF0, 0xF0]); // scales_l
    for k in 0..128 {
        let k = k as u16;
        b[8 + k as usize] = (((k * 13 + 5) & 0xF) | (((k * 11 + 7) & 0xF) << 4)) as u8;
    }
    b
}

#[test]
fn iq4_xs_hand_computed_vector() {
    let expected: [u32; QK_K] = [
        0x438C0000, 0x44260000, 0xC4620000, 0xC3D40000, 0xC2D00000, 0x43300000, 0x44020000,
        0x447E0000, 0xC40A0000, 0xC3480000, 0x42A00000, 0x43C40000, 0x44500000, 0xC4320000,
        0xC3980000, 0xC1000000, 0x42A00000, 0x44260000, 0xC40A0000, 0xC1000000, 0x44020000,
        0xC4320000, 0xC2D00000, 0x43C40000, 0xC4620000, 0xC3480000, 0x438C0000, 0x447E0000,
        0xC3980000, 0x43300000, 0x44500000, 0xC3D40000, 0x436C4000, 0x440C1000, 0xC43EB000,
        0xC3B2E000, 0xC2AF8000, 0x43148000, 0x43DB6000, 0x44565000, 0xC3E8E000, 0xC328C000,
        0x42870000, 0x43A56000, 0x442F8000, 0xC4163000, 0xC3804000, 0xC0D80000, 0x42870000,
        0x440C1000, 0xC3E8E000, 0xC0D80000, 0x43DB6000, 0xC4163000, 0xC2AF8000, 0x43A56000,
        0xC43EB000, 0xC328C000, 0x436C4000, 0x44565000, 0xC3804000, 0x43148000, 0x442F8000,
        0xC3B2E000, 0x430C0000, 0x43A60000, 0xC3E20000, 0xC3540000, 0xC2500000, 0x42B00000,
        0x43820000, 0x43FE0000, 0xC38A0000, 0xC2C80000, 0x42200000, 0x43440000, 0x43D00000,
        0xC3B20000, 0xC3180000, 0xC0800000, 0x42200000, 0x43A60000, 0xC38A0000, 0xC0800000,
        0x43820000, 0xC3B20000, 0xC2500000, 0x43440000, 0xC3E20000, 0xC2C80000, 0x430C0000,
        0x43FE0000, 0xC3180000, 0x42B00000, 0x43D00000, 0xC3540000, 0x410C0000, 0x41A60000,
        0xC1E20000, 0xC1540000, 0xC0500000, 0x40B00000, 0x41820000, 0x41FE0000, 0xC18A0000,
        0xC0C80000, 0x40200000, 0x41440000, 0x41D00000, 0xC1B20000, 0xC1180000, 0xBE800000,
        0x40200000, 0x41A60000, 0xC18A0000, 0xBE800000, 0x41820000, 0xC1B20000, 0xC0500000,
        0x41440000, 0xC1E20000, 0xC0C80000, 0x410C0000, 0x41FE0000, 0xC1180000, 0x40B00000,
        0x41D00000, 0xC1540000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000,
        0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000,
        0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000,
        0x00000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x00000000, 0x00000000,
        0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0xC3034000,
        0xC39BA000, 0x43D3E000, 0x4346C000, 0x42430000, 0xC2A50000, 0xC373C000, 0xC3EE2000,
        0x43816000, 0x42BB8000, 0xC2160000, 0xC337C000, 0xC3C30000, 0x43A6E000, 0x430E8000,
        0x40700000, 0xC2160000, 0xC39BA000, 0x43816000, 0x40700000, 0xC373C000, 0x43A6E000,
        0x42430000, 0xC337C000, 0x43D3E000, 0x42BB8000, 0xC3034000, 0xC3EE2000, 0x430E8000,
        0xC2A50000, 0xC3C30000, 0x4346C000, 0xC30C0000, 0xC3A60000, 0x43E20000, 0x43540000,
        0x42500000, 0xC2B00000, 0xC3820000, 0xC3FE0000, 0x438A0000, 0x42C80000, 0xC2200000,
        0xC3440000, 0xC3D00000, 0x43B20000, 0x43180000, 0x40800000, 0xC2200000, 0xC3A60000,
        0x438A0000, 0x40800000, 0xC3820000, 0x43B20000, 0x42500000, 0xC3440000, 0x43E20000,
        0x42C80000, 0xC30C0000, 0xC3FE0000, 0x43180000, 0xC2B00000, 0xC3D00000, 0x43540000,
        0xC387A000, 0xC420D000, 0x445AF000, 0x43CD6000, 0x42C98000, 0xC32A8000, 0xC3FBE000,
        0xC4761000, 0x4405B000, 0x4341C000, 0xC29B0000, 0xC3BDE000, 0xC4498000, 0x442C7000,
        0x43934000, 0x40F80000, 0xC29B0000, 0xC420D000, 0x4405B000, 0x40F80000, 0xC3FBE000,
        0x442C7000, 0x42C98000, 0xC3BDE000, 0x445AF000, 0x4341C000, 0xC387A000, 0xC4761000,
        0x43934000, 0xC32A8000, 0xC4498000, 0x43CD6000,
    ];
    let mut out = [0.0f32; QK_K];
    dequant_iq4_xs(&iq4_xs_hand_block(), &mut out);
    for j in 0..QK_K {
        assert_eq!(
            out[j].to_bits(),
            expected[j],
            "value {j}: got {}, want {}",
            out[j],
            f32::from_bits(expected[j])
        );
    }
    // The 32 values of sub-block 4 all have dl = 0, signed zero included.
    assert!(out[128..160].iter().all(|v| *v == 0.0));
}

#[test]
fn iq4_gemv_reduces_the_dequantised_block() {
    let nl = iq4_nl_hand_block();
    let mut nl_out = [0.0f32; IQ4_NL_VALUES];
    dequant_iq4_nl(&nl, &mut nl_out);
    let y = gemv_iq4_nl(&nl, &vec![1.0f32; IQ4_NL_VALUES], 1, IQ4_NL_VALUES);
    let want: f32 = nl_out.iter().sum();
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );
    let mut hot = vec![0.0f32; IQ4_NL_VALUES];
    hot[17] = 1.0;
    assert_eq!(gemv_iq4_nl(&nl, &hot, 1, IQ4_NL_VALUES), vec![nl_out[17]]);

    let xs = iq4_xs_hand_block();
    let mut xs_out = [0.0f32; QK_K];
    dequant_iq4_xs(&xs, &mut xs_out);
    let y = gemv_iq4_xs(&xs, &vec![1.0f32; QK_K], 1, QK_K);
    let want: f32 = xs_out.iter().sum();
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );
    let mut hot = vec![0.0f32; QK_K];
    hot[150] = 1.0; // sub-block 4 (dl = 0) offset 22
    assert_eq!(gemv_iq4_xs(&xs, &hot, 1, QK_K), vec![xs_out[150]]);
}

/// Q3_K super-block built by hand.
///
///   d      = 0.5 (fp16 0x3800)
///   scales = 12 bytes that unpack to the 16 six-bit scales
///            [32, 35, 28, 40, 16, 48, 0, 63, 33, 2, 61, 20, 44, 7, 55, 30]
///   hmask  = (i*7+3) & 0xff
///   qs     = (i*5+1) & 0xff
///
/// so `scale - 32` is [0, 3, -4, 8, -16, 16, -32, 31, 1, -30, 29, -12, 12,
/// -25, 23, -2]: zero, negative and positive scales in one block. The two
/// 128-value halves take `hmask` bits 1..8, so both halves and both the
/// `-4` and the signed-quant branch are exercised. Expected bytes come from an
/// independent Python transcription of `dequantize_row_q3_K`
/// (docs/reference/ggml-quants.c:1315).
fn q3_k_hand_block() -> [u8; Q3_K_BYTES] {
    let mut b = [0u8; Q3_K_BYTES];
    for i in 0..32 {
        b[i] = ((i as u16 * 7 + 3) & 0xFF) as u8;
    }
    for i in 0..64 {
        b[32 + i] = ((i as u16 * 5 + 1) & 0xFF) as u8;
    }
    b[96..108].copy_from_slice(&[16, 35, 220, 72, 192, 112, 112, 239, 166, 14, 241, 94]);
    b[108..110].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    b
}

#[test]
fn q3_k_hand_computed_vector() {
    let expected: [u32; QK_K] = [
        0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000,
        0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
        0x00000000, 0x80000000, 0x3FC00000, 0xC0400000, 0x40900000, 0xC0C00000, 0x3FC00000,
        0xC0400000, 0x40900000, 0xC0C00000, 0x3FC00000, 0xC0400000, 0x40900000, 0xC0C00000,
        0x3FC00000, 0xC0400000, 0x40900000, 0xC0C00000, 0x80000000, 0xC0000000, 0x40800000,
        0x41000000, 0xC0000000, 0xC0800000, 0x40000000, 0x40C00000, 0xC0800000, 0xC0C00000,
        0x41000000, 0x40800000, 0xC0C00000, 0x80000000, 0x40C00000, 0x40000000, 0x00000000,
        0x40800000, 0xC1000000, 0xC1800000, 0x40800000, 0x41000000, 0xC0800000, 0xC1400000,
        0x41000000, 0x41400000, 0xC1800000, 0xC1000000, 0x41400000, 0x00000000, 0xC1400000,
        0xC0800000, 0x42000000, 0x42000000, 0x42000000, 0x41C00000, 0xC1000000, 0xC1000000,
        0xC1000000, 0xC1800000, 0x41800000, 0x41800000, 0x41000000, 0x41000000, 0xC1C00000,
        0x80000000, 0x80000000, 0x80000000, 0xC1C00000, 0xC1C00000, 0xC1C00000, 0xC1800000,
        0x41800000, 0x41800000, 0x41800000, 0x41C00000, 0xC1000000, 0xC1000000, 0xC2000000,
        0xC2000000, 0x00000000, 0x41000000, 0x41000000, 0x41000000, 0x42800000, 0x80000000,
        0x42800000, 0x80000000, 0x80000000, 0x42800000, 0x80000000, 0x42800000, 0x80000000,
        0x42800000, 0x80000000, 0x42800000, 0x42800000, 0xC1800000, 0x42400000, 0xC1800000,
        0xC23A0000, 0x41780000, 0xC23A0000, 0x41780000, 0x41780000, 0xC23A0000, 0x41780000,
        0xC23A0000, 0x41780000, 0xC23A0000, 0x41F80000, 0xC1F80000, 0xC1F80000, 0x41F80000,
        0xC1F80000, 0x41F80000, 0xBFC00000, 0xBF800000, 0x3FC00000, 0x00000000, 0x3F000000,
        0xBF800000, 0xBF000000, 0x00000000, 0x3F000000, 0xBF800000, 0xBF000000, 0x00000000,
        0x3F000000, 0x3F800000, 0xBF000000, 0xC0000000, 0xC1700000, 0xC1F00000, 0x41700000,
        0x42700000, 0x42340000, 0xC1F00000, 0xC2340000, 0x42700000, 0x42340000, 0xC1F00000,
        0xC2340000, 0x42700000, 0x42340000, 0x41F00000, 0xC2340000, 0x80000000, 0xC2680000,
        0xC22E0000, 0xC1E80000, 0xC2680000, 0xC22E0000, 0x41E80000, 0x422E0000, 0x41680000,
        0x41E80000, 0xC1680000, 0xC2680000, 0xC1E80000, 0xC1680000, 0xC2680000, 0x41680000,
        0x422E0000, 0x80000000, 0xC0C00000, 0x41400000, 0x41C00000, 0x41900000, 0x41400000,
        0x40C00000, 0xC0C00000, 0xC1400000, 0xC1900000, 0x80000000, 0x41400000, 0x40C00000,
        0x41C00000, 0x41900000, 0x40C00000, 0xC1400000, 0xC1400000, 0xC1400000, 0xC0C00000,
        0xC0C00000, 0xC0C00000, 0xC0C00000, 0xC1C00000, 0xC1C00000, 0x00000000, 0x40C00000,
        0x40C00000, 0x40C00000, 0x41400000, 0x41400000, 0x41400000, 0xC2160000, 0xC2160000,
        0x41480000, 0x42480000, 0x42480000, 0x42480000, 0x42480000, 0x42160000, 0x42160000,
        0x42160000, 0x41C80000, 0xC1C80000, 0xC1C80000, 0xC2160000, 0xC2160000, 0xC2160000,
        0xC1B80000, 0xC1B80000, 0xC1B80000, 0xC1B80000, 0xC1B80000, 0xC1B80000, 0xC1B80000,
        0xC1380000, 0xC1380000, 0xC1380000, 0xC1380000, 0xC1380000, 0xC1380000, 0xC1380000,
        0xC1380000, 0xC1380000, 0x3F800000, 0x3F800000, 0xC0400000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0x80000000, 0x80000000,
    ];
    let mut out = [0.0f32; QK_K];
    dequant_q3_k(&q3_k_hand_block(), &mut out);
    for j in 0..QK_K {
        assert_eq!(
            out[j].to_bits(),
            expected[j],
            "value {j}: got {}, want {}",
            out[j],
            f32::from_bits(expected[j])
        );
    }
    // The first 16 values are the zero-scale sub-block, signs included.
    assert!(out[..16].iter().all(|v| *v == 0.0));
    assert!(out.iter().any(|v| *v < 0.0) && out.iter().any(|v| *v > 0.0));
}

/// IQ3_S super-block built by hand:
///   d      = 0.5 (fp16 0x3800)
///   qs     = (i*13+5) & 0xff          (grid index low 8 bits)
///   qh     = [0xA5, 0x3C, 0xFF, 0x01, 0x80, 0x55, 0x0F, 0xF0]  (ninth bit)
///   signs  = (i*37+11) & 0xff         (one bit per value)
///   scales = [0x3A, 0xC5, 0x71, 0x0D] (low/high nibble per 32-value pair)
///
/// Every `l` takes its first four values from grid entry `qs[2l]` and its last
/// four from `qs[2l+1]` — both halves of each byte pair — and every value's
/// sign comes from a different bit of `signs`. Expected bytes come from an
/// independent Python transcription of `dequantize_row_iq3_s`
/// (docs/reference/ggml-quants.c:2623) and the `iq3s_grid` table.
fn iq3_s_hand_block() -> [u8; IQ3_S_BYTES] {
    let mut b = [0u8; IQ3_S_BYTES];
    b[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    for i in 0..64 {
        b[2 + i] = ((i as u16 * 13 + 5) & 0xFF) as u8;
    }
    b[66..74].copy_from_slice(&[0xA5, 0x3C, 0xFF, 0x01, 0x80, 0x55, 0x0F, 0xF0]);
    for i in 0..32 {
        b[74 + i] = ((i as u16 * 37 + 11) & 0xFF) as u8;
    }
    b[106..110].copy_from_slice(&[0x3A, 0xC5, 0x71, 0x0D]);
    b
}

#[test]
fn iq3_s_hand_computed_vector() {
    let expected: [u32; QK_K] = [
        0xC2E70000, 0xC31D8000, 0x42BD0000, 0xC2520000, 0x41FC0000, 0x42E70000, 0x41280000,
        0x41280000, 0x41FC0000, 0x42930000, 0x41280000, 0x42930000, 0xC31D8000, 0xC1280000,
        0x42520000, 0x41280000, 0xC1280000, 0x431D8000, 0xC2520000, 0x41280000, 0xC3088000,
        0x41FC0000, 0xC2BD0000, 0x42930000, 0x42520000, 0xC1280000, 0x42E70000, 0xC1280000,
        0xC31D8000, 0xC2520000, 0xC1280000, 0x42BD0000, 0xC0600000, 0xC0600000, 0xC0600000,
        0xC1280000, 0xC2360000, 0x41C40000, 0x40600000, 0xC1280000, 0x42520000, 0x40600000,
        0xC21A0000, 0x41FC0000, 0x41FC0000, 0x41280000, 0xC0600000, 0xC21A0000, 0xC18C0000,
        0x41C40000, 0x41280000, 0xC21A0000, 0x41280000, 0xC21A0000, 0xC1C40000, 0xC21A0000,
        0x41280000, 0xC0600000, 0xC2520000, 0xC1280000, 0x42520000, 0x40600000, 0x40600000,
        0x418C0000, 0xC0B00000, 0xC0B00000, 0x42720000, 0x428F0000, 0xC28F0000, 0xC21A0000,
        0x40B00000, 0x42A50000, 0x41840000, 0x41DC0000, 0x41DC0000, 0xC2A50000, 0xC1DC0000,
        0x42460000, 0xC2720000, 0x42A50000, 0xC2A50000, 0x421A0000, 0xC2720000, 0xC1DC0000,
        0xC0B00000, 0xC21A0000, 0xC2A50000, 0x41DC0000, 0x42460000, 0xC2460000, 0x40B00000,
        0x421A0000, 0x41DC0000, 0xC2A50000, 0x41840000, 0xC21A0000, 0xC33B8000, 0xC27A0000,
        0xC2AF0000, 0x42AF0000, 0x42AF0000, 0x42160000, 0xC2E10000, 0xC1480000, 0x43228000,
        0x427A0000, 0xC3098000, 0xC1480000, 0x41480000, 0xC1480000, 0xC33B8000, 0xC1480000,
        0xC1480000, 0x42160000, 0x41480000, 0x42160000, 0xC27A0000, 0x433B8000, 0x41480000,
        0x42160000, 0x41480000, 0xC33B8000, 0xC2160000, 0x42160000, 0xC27A0000, 0xC33B8000,
        0x427A0000, 0x42160000, 0xC1280000, 0xBFC00000, 0x41580000, 0xC0900000, 0xC0900000,
        0x40F00000, 0xC1840000, 0x40900000, 0x40900000, 0x40F00000, 0x41B40000, 0x40900000,
        0x419C0000, 0x40900000, 0x3FC00000, 0xC0F00000, 0xC1280000, 0x41B40000, 0xBFC00000,
        0x40F00000, 0x40900000, 0xBFC00000, 0x40F00000, 0xC0F00000, 0x40F00000, 0xBFC00000,
        0x41280000, 0xC0F00000, 0x41280000, 0x40F00000, 0xC1580000, 0xC0F00000, 0xC0F00000,
        0xC0F00000, 0xC2C30000, 0xC2160000, 0x41B40000, 0xC0F00000, 0xC1B40000, 0xC0F00000,
        0x41B40000, 0x42E10000, 0xC0F00000, 0x42520000, 0xC2160000, 0x42870000, 0x42160000,
        0x40F00000, 0xC2E10000, 0x42870000, 0x42520000, 0xC2520000, 0xC2160000, 0xC2520000,
        0x42870000, 0x40F00000, 0x40F00000, 0xC0F00000, 0xC0F00000, 0xC2870000, 0xC2160000,
        0x42160000, 0xC2E10000, 0x40F00000, 0xC1580000, 0xC2F30000, 0x42870000, 0x42F30000,
        0x42F30000, 0x42870000, 0x42F30000, 0xC2F30000, 0x41580000, 0x42BD0000, 0x434A8000,
        0xC2F30000, 0x42220000, 0xC1580000, 0x42220000, 0xC3148000, 0xC2220000, 0x42BD0000,
        0xC2F30000, 0xC2220000, 0x41580000, 0x42870000, 0xC32F8000, 0xC2220000, 0x41580000,
        0xC34A8000, 0x434A8000, 0x42220000, 0xC1580000, 0xC2BD0000, 0xC1580000, 0xC2870000,
        0xBF000000, 0xBFC00000, 0xBFC00000, 0x40200000, 0xC0600000, 0x40200000, 0x40200000,
        0x40200000, 0x40900000, 0x40200000, 0xC0600000, 0xC0200000, 0xC0F00000, 0xBF000000,
        0x3F000000, 0x3F000000, 0xC0B00000, 0x40B00000, 0x40D00000, 0x40200000, 0x40200000,
        0xC0200000, 0xBF000000, 0x40600000, 0x40900000, 0xBFC00000, 0xBFC00000, 0x40600000,
        0x3FC00000, 0x3F000000, 0x40600000, 0xC0600000,
    ];
    let mut out = [0.0f32; QK_K];
    dequant_iq3_s(&iq3_s_hand_block(), &mut out);
    for j in 0..QK_K {
        assert_eq!(
            out[j].to_bits(),
            expected[j],
            "value {j}: got {}, want {}",
            out[j],
            f32::from_bits(expected[j])
        );
    }
    assert!(out.iter().any(|v| *v < 0.0) && out.iter().any(|v| *v > 0.0));
    // A sign bit flips exactly one value: bit 0 of signs[0] is value 0.
    let mut flipped = iq3_s_hand_block();
    flipped[74] ^= 1;
    let mut other = [0.0f32; QK_K];
    dequant_iq3_s(&flipped, &mut other);
    assert_eq!(other[0], -out[0]);
    for j in 1..QK_K {
        assert_eq!(other[j].to_bits(), out[j].to_bits(), "value {j} moved");
    }
}

#[test]
fn iq3_s_grid_table_matches_the_vendored_c() {
    // Spot values from ggml-common.h `iq3s_grid`, plus the fact that every
    // entry is four 1..15 bytes: a truncated or reordered table cannot pass
    // both.
    assert_eq!(IQ3S_GRID[0], 0x0101_0101);
    assert_eq!(IQ3S_GRID[1], 0x0101_0103);
    assert_eq!(IQ3S_GRID[255], 0x0509_0501);
    assert_eq!(IQ3S_GRID[256], 0x0509_0507);
    assert_eq!(IQ3S_GRID[511], 0x0F0F_0101);
    assert!(IQ3S_GRID
        .iter()
        .all(|g| g.to_le_bytes().iter().all(|b| (1..=15).contains(b))));
}

#[test]
fn q3_iq3_gemv_reduces_the_dequantised_block() {
    let q3 = q3_k_hand_block();
    let mut q3_out = [0.0f32; QK_K];
    dequant_q3_k(&q3, &mut q3_out);
    let y = gemv_q3_k(&q3, &vec![1.0f32; QK_K], 1, QK_K);
    let want: f32 = q3_out.iter().sum();
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );
    let mut hot = vec![0.0f32; QK_K];
    hot[200] = 1.0;
    assert_eq!(gemv_q3_k(&q3, &hot, 1, QK_K), vec![q3_out[200]]);

    let iq3 = iq3_s_hand_block();
    let mut iq3_out = [0.0f32; QK_K];
    dequant_iq3_s(&iq3, &mut iq3_out);
    let y = gemv_iq3_s(&iq3, &vec![1.0f32; QK_K], 1, QK_K);
    let want: f32 = iq3_out.iter().sum();
    assert!(
        (y[0] - want).abs() <= want.abs() * 1e-6,
        "got {} want {want}",
        y[0]
    );
    let mut hot = vec![0.0f32; QK_K];
    hot[77] = 1.0;
    assert_eq!(gemv_iq3_s(&iq3, &hot, 1, QK_K), vec![iq3_out[77]]);
}

#[test]
fn tensor_type_sizes_match_the_ggml_blocks() {
    assert_eq!(TensorType::Q4K.nbytes(256), Some(144));
    assert_eq!(TensorType::Q6K.nbytes(256), Some(210));
    assert_eq!(TensorType::Q5K.nbytes(256), Some(176));
    assert_eq!(TensorType::Q8_0.nbytes(32), Some(34));
    assert_eq!(TensorType::IQ4NL.nbytes(32), Some(18));
    assert_eq!(TensorType::IQ4XS.nbytes(256), Some(136));
    assert_eq!(TensorType::IQ4XS.nbytes(5120 * 17408), Some(47_349_760));
    assert_eq!(TensorType::Q3K.nbytes(256), Some(110));
    assert_eq!(TensorType::IQ3S.nbytes(256), Some(110));
    assert_eq!(TensorType::Q3K.nbytes(5120 * 17408), Some(38_297_600));
    assert_eq!(TensorType::IQ3S.nbytes(5120 * 17408), Some(38_297_600));
    assert_eq!(TensorType::F32.nbytes(10), Some(40));
    assert_eq!(TensorType::Q4K.nbytes(255), None);
    assert_eq!(TensorType::Unsupported(99).nbytes(256), None);
    assert_eq!(TensorType::from_id(12), TensorType::Q4K);
    assert_eq!(TensorType::from_id(14), TensorType::Q6K);
    assert_eq!(TensorType::from_id(11), TensorType::Q3K);
    assert_eq!(TensorType::from_id(21), TensorType::IQ3S);
    assert_eq!(TensorType::IQ3S.id(), 21);
}

#[test]
fn f32_gemv_still_works_for_the_parity_baseline() {
    // Guard against a quant module refactor touching the f32 path.
    let w = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![1.0, 1.0];
    assert_eq!(ops::gemv_f32(&w, &x, 2, 2), vec![3.0, 7.0]);
}
