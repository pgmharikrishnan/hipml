// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! Quantized block formats from GGML, and their reference dequantisation.
//!
//! Only the formats hipml's models store are here: Q4_K, Q5_K, Q6_K, Q8_0, the
//! IQ4 pair Unsloth's dynamic quants use, plus Q3_K and IQ3_S. The layout below
//! is the on-disk ggml layout, byte for byte — no padding, no reordering.
//! Getting the 6-bit unpacking in Q4_K and the qh bit planes in Q6_K right is
//! the whole point of this module, so the hand-computed vectors live in
//! `tests/quant_reference.rs`.
//!
//! Reference GEMV is dequantise-then-dot, block by block: it never materialises
//! the full dequantised matrix, so a [17408, 5120] parity case is 350 MB of
//! quantized bytes instead of 356 MB of f32.

use crate::rng::Rng;

/// Values per K-quant super-block.
pub const QK_K: usize = 256;

/// Bytes per Q4_K super-block: d(2) + dmin(2) + scales(12) + qs(128).
pub const Q4_K_BYTES: usize = 144;
/// Bytes per Q5_K super-block: d(2) + dmin(2) + scales(12) + qh(32) + qs(128).
pub const Q5_K_BYTES: usize = 176;
/// Bytes per Q6_K super-block: ql(128) + qh(64) + scales(16) + d(2).
pub const Q6_K_BYTES: usize = 210;
/// Values per Q8_0 block.
pub const Q8_0_VALUES: usize = 32;
/// Bytes per Q8_0 block: d(2) + qs(32).
pub const Q8_0_BYTES: usize = 34;
/// Bytes per BF16 value. BF16 is not a block format: one value, two bytes,
/// the top half of an f32.
pub const BF16_BYTES: usize = 2;
/// Values per IQ4_NL block.
pub const IQ4_NL_VALUES: usize = 32;
/// Bytes per IQ4_NL block: d(2) + qs(16).
pub const IQ4_NL_BYTES: usize = 18;
/// Bytes per IQ4_XS super-block: d(2) + scales_h(2) + scales_l(4) + qs(128).
pub const IQ4_XS_BYTES: usize = 136;
/// Bytes per Q3_K super-block: hmask(32) + qs(64) + scales(12) + d(2).
pub const Q3_K_BYTES: usize = 110;
/// Bytes per IQ3_S super-block: d(2) + qs(64) + qh(8) + signs(32) + scales(4).
pub const IQ3_S_BYTES: usize = 110;

/// The 16-entry signed codebook both IQ4 formats index, from ggml-common.h
/// (`kvalues_iq4nl`). It is not in `docs/reference/`; taken verbatim from
/// upstream llama.cpp `ggml/src/ggml-common.h`.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// The 512-entry 3-bit codebook `iq3s_grid`: four little-endian bytes per
/// grid point. Transcribed from `iq3s_grid` in upstream llama.cpp
/// `ggml/src/ggml-common.h` (the copy vendored at
/// `app/mlx-serve/lib/ds4/cuda/mmq/ggml-common.h`); IQ3_S indexes it with a
/// 9-bit index, the low 8 bits from `qs` and the ninth from `qh`.
pub const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010B, 0x0101010F, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030D, 0x01010501, 0x01010503, 0x0101050B, 0x01010707, 0x01010901, 0x01010905,
    0x0101090B, 0x0101090F, 0x01010B03, 0x01010B07, 0x01010D01, 0x01010D05, 0x01010F03, 0x01010F09,
    0x01010F0F, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030B,
    0x01030501, 0x01030507, 0x0103050F, 0x01030703, 0x0103070B, 0x01030909, 0x01030D03, 0x01030D0B,
    0x01030F05, 0x01050101, 0x01050103, 0x0105010B, 0x0105010F, 0x01050301, 0x01050307, 0x0105030D,
    0x01050503, 0x0105050B, 0x01050701, 0x01050709, 0x01050905, 0x0105090B, 0x0105090F, 0x01050B03,
    0x01050B07, 0x01050F01, 0x01050F07, 0x01070107, 0x01070303, 0x0107030B, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070D, 0x01070909, 0x01070B01, 0x01070B05, 0x01070D0F, 0x01070F03,
    0x01070F0B, 0x01090101, 0x01090307, 0x0109030F, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090B03, 0x01090F01, 0x010B0105, 0x010B0109, 0x010B0501, 0x010B0505, 0x010B050D,
    0x010B0707, 0x010B0903, 0x010B090B, 0x010B090F, 0x010B0D0D, 0x010B0F07, 0x010D010D, 0x010D0303,
    0x010D0307, 0x010D0703, 0x010D0B05, 0x010D0F03, 0x010F0101, 0x010F0105, 0x010F0109, 0x010F0501,
    0x010F0505, 0x010F050D, 0x010F0707, 0x010F0B01, 0x010F0B09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030B, 0x0301030F, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070D, 0x03010B09, 0x03010B0D, 0x03010D03, 0x03010F05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010D, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030B01, 0x03030B05, 0x03030F01, 0x03030F0D, 0x03050101, 0x03050305, 0x0305030B,
    0x0305030F, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050B0B, 0x03050D01,
    0x03050F05, 0x03070103, 0x03070109, 0x0307010F, 0x03070301, 0x03070307, 0x03070503, 0x0307050F,
    0x03070701, 0x03070709, 0x03070903, 0x03070D05, 0x03070F01, 0x03090107, 0x0309010B, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090D, 0x03090B01, 0x03090B09, 0x030B0103,
    0x030B0301, 0x030B0307, 0x030B0503, 0x030B0701, 0x030B0705, 0x030B0B03, 0x030D0501, 0x030D0509,
    0x030D050F, 0x030D0909, 0x030D090D, 0x030F0103, 0x030F0107, 0x030F0301, 0x030F0305, 0x030F0503,
    0x030F070B, 0x030F0903, 0x030F0D05, 0x030F0F01, 0x05010101, 0x05010103, 0x05010107, 0x0501010B,
    0x0501010F, 0x05010301, 0x05010305, 0x05010309, 0x0501030D, 0x05010503, 0x05010507, 0x0501050F,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090B, 0x05010B01, 0x05010B05, 0x05010D0F,
    0x05010F01, 0x05010F07, 0x05010F0B, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030F,
    0x05030505, 0x0503050B, 0x05030703, 0x05030709, 0x05030905, 0x05030B03, 0x05050103, 0x05050109,
    0x0505010F, 0x05050503, 0x05050507, 0x05050701, 0x0505070F, 0x05050903, 0x05050B07, 0x05050B0F,
    0x05050F03, 0x05050F09, 0x05070101, 0x05070105, 0x0507010B, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070B01, 0x05070D0D, 0x05090103, 0x0509010F, 0x05090501,
    0x05090507, 0x05090705, 0x0509070B, 0x05090903, 0x05090F05, 0x05090F0B, 0x050B0109, 0x050B0303,
    0x050B0505, 0x050B070F, 0x050B0901, 0x050B0B07, 0x050B0F01, 0x050D0101, 0x050D0105, 0x050D010F,
    0x050D0503, 0x050D0B0B, 0x050D0D03, 0x050F010B, 0x050F0303, 0x050F050D, 0x050F0701, 0x050F0907,
    0x050F0B01, 0x07010105, 0x07010303, 0x07010307, 0x0701030B, 0x0701030F, 0x07010505, 0x07010703,
    0x07010707, 0x0701070B, 0x07010905, 0x07010909, 0x0701090F, 0x07010B03, 0x07010D07, 0x07010F03,
    0x07030103, 0x07030107, 0x0703010B, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030D01,
    0x07030F05, 0x07030F0D, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050B01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050F, 0x07070701, 0x07070903,
    0x07070907, 0x0707090F, 0x07070B0B, 0x07070F07, 0x07090107, 0x07090303, 0x0709030D, 0x07090505,
    0x07090703, 0x07090B05, 0x07090D01, 0x07090D09, 0x070B0103, 0x070B0301, 0x070B0305, 0x070B050B,
    0x070B0705, 0x070B0909, 0x070B0B0D, 0x070B0F07, 0x070D030D, 0x070D0903, 0x070F0103, 0x070F0107,
    0x070F0501, 0x070F0505, 0x070F070B, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050F, 0x09010705, 0x09010903, 0x09010B01, 0x09010F01, 0x09030105, 0x0903010F, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070B, 0x09030907, 0x09030B03, 0x09030B0B, 0x09050103,
    0x09050107, 0x09050301, 0x0905030B, 0x09050503, 0x09050707, 0x09050901, 0x09050B0F, 0x09050D05,
    0x09050F01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070B,
    0x09090101, 0x09090105, 0x09090509, 0x0909070F, 0x09090901, 0x09090F03, 0x090B010B, 0x090B010F,
    0x090B0503, 0x090B0D05, 0x090D0307, 0x090D0709, 0x090D0D01, 0x090F0301, 0x090F030B, 0x090F0701,
    0x090F0907, 0x090F0B03, 0x0B010105, 0x0B010301, 0x0B010309, 0x0B010505, 0x0B010901, 0x0B010909,
    0x0B01090F, 0x0B010B05, 0x0B010D0D, 0x0B010F09, 0x0B030103, 0x0B030107, 0x0B03010B, 0x0B030305,
    0x0B030503, 0x0B030705, 0x0B030F05, 0x0B050101, 0x0B050303, 0x0B050507, 0x0B050701, 0x0B05070D,
    0x0B050B07, 0x0B070105, 0x0B07010F, 0x0B070301, 0x0B07050F, 0x0B070909, 0x0B070B03, 0x0B070D0B,
    0x0B070F07, 0x0B090103, 0x0B090109, 0x0B090501, 0x0B090705, 0x0B09090D, 0x0B0B0305, 0x0B0B050D,
    0x0B0B0B03, 0x0B0B0B07, 0x0B0D0905, 0x0B0F0105, 0x0B0F0109, 0x0B0F0505, 0x0D010303, 0x0D010307,
    0x0D01030B, 0x0D010703, 0x0D010707, 0x0D010D01, 0x0D030101, 0x0D030501, 0x0D03050F, 0x0D030D09,
    0x0D050305, 0x0D050709, 0x0D050905, 0x0D050B0B, 0x0D050D05, 0x0D050F01, 0x0D070101, 0x0D070309,
    0x0D070503, 0x0D070901, 0x0D09050B, 0x0D090907, 0x0D090D05, 0x0D0B0101, 0x0D0B0107, 0x0D0B0709,
    0x0D0B0D01, 0x0D0D010B, 0x0D0D0901, 0x0D0F0303, 0x0D0F0307, 0x0F010101, 0x0F010109, 0x0F01010F,
    0x0F010501, 0x0F010505, 0x0F01070D, 0x0F010901, 0x0F010B09, 0x0F010D05, 0x0F030105, 0x0F030303,
    0x0F030509, 0x0F030907, 0x0F03090B, 0x0F050103, 0x0F050109, 0x0F050301, 0x0F05030D, 0x0F050503,
    0x0F050701, 0x0F050B03, 0x0F070105, 0x0F070705, 0x0F07070B, 0x0F070B07, 0x0F090103, 0x0F09010B,
    0x0F090307, 0x0F090501, 0x0F090B01, 0x0F0B0505, 0x0F0B0905, 0x0F0D0105, 0x0F0D0703, 0x0F0F0101,
];

/// Convert an IEEE-754 binary16 bit pattern to f32, exactly.
///
/// Kept here rather than pulled from a crate: it is nine lines of bit work and
/// the project has no dependencies.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal half: value = mant * 2^-24. Normalise into f32 so the
            // implicit leading one is explicit again.
            let mut k = 0u32;
            while (mant >> (k + 1)) != 0 {
                k += 1;
            }
            let e = (k as i32 - 24 + 127) as u32;
            let frac = (mant - (1 << k)) << (23 - k);
            sign | (e << 23) | frac
        }
    } else if exp == 31 {
        // Infinity or NaN; carry the mantissa so NaN stays NaN.
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// BF16 bit pattern to f32.
///
/// BF16 is the top 16 bits of an f32, so widening is a shift and the low 16
/// mantissa bits are zero — that is the only rounding the format has.
/// `ggml_bf16_to_fp32` in upstream `ggml-impl.h` is the same shift.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Dequantise one BF16 value (2 bytes -> 1 value).
///
/// BF16 has no block structure, so this is the degenerate case of the
/// `dequant_*` family: one value per call, which is what `dequant_row` and
/// `dot_row` use for a whole row.
pub fn dequant_bf16(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), BF16_BYTES, "BF16 value must be 2 bytes");
    assert_eq!(out.len(), 1, "BF16 value must produce 1 output");
    out[0] = bf16_to_f32(u16::from_le_bytes([block[0], block[1]]));
}

/// A tensor type as it appears in a GGUF tensor-info record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    BF16,
    Q8_0,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4NL,
    IQ4XS,
    IQ3S,
    /// Anything else: the reader can list it but not size it.
    Unsupported(u32),
}

impl TensorType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => TensorType::F32,
            8 => TensorType::Q8_0,
            11 => TensorType::Q3K,
            12 => TensorType::Q4K,
            13 => TensorType::Q5K,
            14 => TensorType::Q6K,
            20 => TensorType::IQ4NL,
            21 => TensorType::IQ3S,
            23 => TensorType::IQ4XS,
            30 => TensorType::BF16,
            other => TensorType::Unsupported(other),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            TensorType::F32 => 0,
            TensorType::BF16 => 30,
            TensorType::Q8_0 => 8,
            TensorType::Q3K => 11,
            TensorType::Q4K => 12,
            TensorType::Q5K => 13,
            TensorType::Q6K => 14,
            TensorType::IQ4NL => 20,
            TensorType::IQ3S => 21,
            TensorType::IQ4XS => 23,
            TensorType::Unsupported(id) => id,
        }
    }

    pub fn name(self) -> String {
        match self {
            TensorType::F32 => "F32".to_string(),
            TensorType::BF16 => "BF16".to_string(),
            TensorType::Q8_0 => "Q8_0".to_string(),
            TensorType::Q3K => "Q3_K".to_string(),
            TensorType::Q4K => "Q4_K".to_string(),
            TensorType::Q5K => "Q5_K".to_string(),
            TensorType::Q6K => "Q6_K".to_string(),
            TensorType::IQ4NL => "IQ4_NL".to_string(),
            TensorType::IQ3S => "IQ3_S".to_string(),
            TensorType::IQ4XS => "IQ4_XS".to_string(),
            TensorType::Unsupported(id) => format!("type_{id}"),
        }
    }

    /// Values per block (1 for F32), or 0 when unsupported.
    pub fn block_values(self) -> usize {
        match self {
            TensorType::F32 => 1,
            TensorType::BF16 => 1,
            TensorType::Q8_0 => Q8_0_VALUES,
            TensorType::Q3K
            | TensorType::Q4K
            | TensorType::Q5K
            | TensorType::Q6K
            | TensorType::IQ4XS
            | TensorType::IQ3S => QK_K,
            TensorType::IQ4NL => IQ4_NL_VALUES,
            TensorType::Unsupported(id) => size_only(id).0,
        }
    }

    /// Bytes per block, or 0 when unsupported.
    pub fn block_bytes(self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::BF16 => BF16_BYTES,
            TensorType::Q8_0 => Q8_0_BYTES,
            TensorType::Q3K => Q3_K_BYTES,
            TensorType::Q4K => Q4_K_BYTES,
            TensorType::Q5K => Q5_K_BYTES,
            TensorType::Q6K => Q6_K_BYTES,
            TensorType::IQ4NL => IQ4_NL_BYTES,
            TensorType::IQ3S => IQ3_S_BYTES,
            TensorType::IQ4XS => IQ4_XS_BYTES,
            TensorType::Unsupported(id) => size_only(id).1,
        }
    }

    /// Size of `n_elements` values in bytes, or `None` when the type is unknown
    /// or the element count is not a whole number of blocks.
    pub fn nbytes(self, n_elements: u64) -> Option<u64> {
        let bv = self.block_values() as u64;
        let bb = self.block_bytes() as u64;
        if bv == 0 || n_elements % bv != 0 {
            return None;
        }
        Some(n_elements / bv * bb)
    }
}

/// `(values, bytes)` per block for GGML types that are sized but never
/// dequantised here, so a file that carries one still loads and the load
/// report can name it: F16, Q4_0, Q2_K.
///
/// The remaining IQ entries matter because Unsloth's "UD" dynamic quants mix
/// tensor types per layer; sizing them is metadata only, not a dequant path.
/// `(0, 0)` for anything else.
fn size_only(id: u32) -> (usize, usize) {
    match id {
        1 => (1, 2),
        2 => (32, 18),
        10 => (QK_K, 84),
        _ => (0, 0),
    }
}

/// Dequantise one Q4_K super-block (144 bytes -> 256 values).
///
/// `y = d * scale[j] * q - dmin * min[j]`, where each 6-bit `scale[j]`/`min[j]`
/// is packed into the 12-byte header. Low nibbles of a 32-byte group are the
/// even sub-block, high nibbles the odd one.
pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q4_K_BYTES, "Q4_K block must be 144 bytes");
    assert_eq!(out.len(), QK_K, "Q4_K block must produce 256 values");

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    for j in 0..8 {
        let (sc, mn) = q4_k_scale_min(j, scales);
        let scale = d * sc as f32;
        let min = dmin * mn as f32;
        let group = &qs[(j / 2) * 32..(j / 2) * 32 + 32];
        let base = j * 32;
        if j % 2 == 0 {
            for (l, &byte) in group.iter().enumerate() {
                out[base + l] = scale * (byte & 0xF) as f32 - min;
            }
        } else {
            for (l, &byte) in group.iter().enumerate() {
                out[base + l] = scale * (byte >> 4) as f32 - min;
            }
        }
    }
}

/// Unpack the 6-bit scale and min for sub-block `j` from the 12 header bytes.
///
/// Bytes 0..4 hold the low 6 bits of scales 0..4; bytes 4..8 the low 6 bits of
/// mins 0..4. Bytes 8..12 hold the low nibble of scales 4..8 and the high
/// nibble of mins 4..8; the top two bits of bytes 0..4 (scales) and 4..8 (mins)
/// are the missing high bits.
fn q4_k_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, mn)
    }
}

/// Dequantise one Q5_K super-block (176 bytes -> 256 values).
///
/// Port of `dequantize_row_q5_K` (ggml-quants.c:1741). Not in the original
/// scope — Qwen3.8-27B Q4_K_M stores exactly two Q5_K tensors
/// (`blk.51.ffn_down.weight`, `blk.55.ffn_down.weight`) and the forward pass
/// cannot run without them.
///
/// `y = d*scale[j]*(q + 16*qh_bit) - dmin*min[j]` where `q` is the 4-bit
/// `qs` nibble and `qh` supplies the fifth bit, two bits per 64-value group.
pub fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q5_K_BYTES, "Q5_K block must be 176 bytes");
    assert_eq!(out.len(), QK_K, "Q5_K block must produce 256 values");

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for j in (0..QK_K).step_by(64) {
        let (sc, mn) = q4_k_scale_min(is, scales);
        let (d1, m1) = (d * sc as f32, dmin * mn as f32);
        let (sc, mn) = q4_k_scale_min(is + 1, scales);
        let (d2, m2) = (d * sc as f32, dmin * mn as f32);
        let ql = &qs[(j / 64) * 32..(j / 64) * 32 + 32];
        for l in 0..32 {
            let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            out[j + l] = d1 * ((ql[l] & 0xF) as f32 + hi) - m1;
        }
        for l in 0..32 {
            let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            out[j + 32 + l] = d2 * ((ql[l] >> 4) as f32 + hi) - m2;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantise one Q6_K super-block (210 bytes -> 256 values).
///
/// `q = (ql nibble) | ((qh 2-bit plane) << 4)`, then `y = d * scale[j] * (q-32)`.
/// The 16 sub-blocks are 16 values each; sub-block `j` starts at `j*16`.
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q6_K_BYTES, "Q6_K block must be 210 bytes");
    assert_eq!(out.len(), QK_K, "Q6_K block must produce 256 values");

    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    for j in 0..16 {
        let scale = d * (scales[j] as i8) as f32;
        let half = j / 8; // first or second 128-value half
        let s = j % 8; // sub-block within that half
        let group = s / 2; // 0..3, selects q1..q4
        let is = s % 2; // which 16-value slice of the 32-value group
        let ql_off = half * 64 + (group & 1) * 32;
        let qh_off = half * 32;
        let l0 = is * 16;
        let high_nibble = group >= 2;
        let qh_shift = 2 * group;
        let base = j * 16;

        for l in 0..16 {
            let qb = ql[ql_off + l0 + l];
            let nib = if high_nibble { qb >> 4 } else { qb & 0xF };
            let hi = (qh[qh_off + l0 + l] >> qh_shift) & 3;
            let q = (nib | (hi << 4)) as i32 - 32;
            out[base + l] = scale * q as f32;
        }
    }
}

/// Dequantise one Q8_0 block (34 bytes -> 32 values). Used by the reader test
/// and by the Q8_0 GEMV reference.
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q8_0_BYTES, "Q8_0 block must be 34 bytes");
    assert_eq!(out.len(), Q8_0_VALUES, "Q8_0 block must produce 32 values");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..Q8_0_VALUES {
        out[i] = d * (block[2 + i] as i8) as f32;
    }
}

/// Dequantise one IQ4_NL block (18 bytes -> 32 values).
///
/// Port of `dequantize_row_iq4_nl` (ggml-quants.c:2741). Both halves index the
/// same 16-entry signed codebook: the low nibble of `qs[j]` is value `j`, the
/// high nibble is value `j + 16`.
pub fn dequant_iq4_nl(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), IQ4_NL_BYTES, "IQ4_NL block must be 18 bytes");
    assert_eq!(
        out.len(),
        IQ4_NL_VALUES,
        "IQ4_NL block must produce 32 values"
    );
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    for j in 0..16 {
        out[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
        out[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

/// Dequantise one IQ4_XS super-block (136 bytes -> 256 values).
///
/// Port of `dequantize_row_iq4_xs` (ggml-quants.c:2759). Eight 32-value
/// sub-blocks; each has a 6-bit signed scale assembled from the low nibble in
/// `scales_l` and two bits in `scales_h`, and `dl = d * (ls - 32)` is negative
/// for `ls < 32`. The 16 `qs` bytes of a sub-block produce values 0..15 from
/// their low nibbles and 16..31 from their high nibbles.
pub fn dequant_iq4_xs(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), IQ4_XS_BYTES, "IQ4_XS block must be 136 bytes");
    assert_eq!(out.len(), QK_K, "IQ4_XS block must produce 256 values");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..136];

    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..ib * 16 + 16];
        for j in 0..16 {
            out[ib * 32 + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            out[ib * 32 + 16 + j] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

/// Dequantise one Q3_K super-block (110 bytes -> 256 values).
///
/// Port of `dequantize_row_q3_K` (ggml-quants.c:1315). Two 128-value halves;
/// each half is four 32-value sub-blocks. A value is a 2-bit field of `qs`
/// shifted by `2*j`, minus 4 unless the matching `hmask` bit is set — so the
/// range is -4..3 and the low bit of the range is what `hmask` supplies. The
/// 16 six-bit scales are packed into 12 bytes and unpacked here exactly as the
/// C does it (four `u32` words, then a nibble/2-bit shuffle).
pub fn dequant_q3_k(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), Q3_K_BYTES, "Q3_K block must be 110 bytes");
    assert_eq!(out.len(), QK_K, "Q3_K block must produce 256 values");

    let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let hm = &block[0..32];
    let q = &block[32..96];
    let sc = &block[96..108];

    // memcpy(aux, scales, 12), then the shuffle from the reference. aux[3] is
    // written before it is read, so its initial value does not matter.
    let mut aux = [0u32; 4];
    for (i, a) in aux.iter_mut().take(3).enumerate() {
        *a = u32::from_le_bytes([sc[i * 4], sc[i * 4 + 1], sc[i * 4 + 2], sc[i * 4 + 3]]);
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for i in 0..4 {
        for (b, byte) in aux[i].to_le_bytes().iter().enumerate() {
            scales[i * 4 + b] = *byte as i8;
        }
    }

    let mut is = 0usize;
    for n in 0..2 {
        let mut shift = 0u32;
        for j in 0..4 {
            let m = 1u8 << (n * 4 + j);
            let base = n * 128 + j * 32;
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                let qo = n * 32 + half * 16;
                let ho = half * 16;
                for l in 0..16 {
                    let qv = ((q[qo + l] >> shift) & 3) as i32;
                    let sub = if hm[ho + l] & m != 0 { 0 } else { 4 };
                    out[base + half * 16 + l] = dl * (qv - sub) as f32;
                }
            }
            shift += 2;
        }
    }
}

/// Dequantise one IQ3_S super-block (110 bytes -> 256 values).
///
/// Port of `dequantize_row_iq3_s` (ggml-quants.c:2623). Eight 32-value
/// sub-blocks in four `(low nibble, high nibble)` scale pairs; each 32 values
/// are four grid lookups x 8 values. The grid index is 9 bits: `qs` supplies
/// the low 8 and one `qh` bit the ninth. Every value takes its sign from one
/// bit of `signs`, so a sign-transcription error changes fluent text without
/// crashing — the hand vector in `tests/quant_reference.rs` pins all 256.
pub fn dequant_iq3_s(block: &[u8], out: &mut [f32]) {
    assert_eq!(block.len(), IQ3_S_BYTES, "IQ3_S block must be 110 bytes");
    assert_eq!(out.len(), QK_K, "IQ3_S block must produce 256 values");

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];

    let mut qo = 0usize;
    let mut so = 0usize;
    for pair in 0..4 {
        let db1 = d * (1 + 2 * (scales[pair] & 0xF) as i32) as f32;
        let db2 = d * (1 + 2 * (scales[pair] >> 4) as i32) as f32;
        for half in 0..2 {
            let db = if half == 0 { db1 } else { db2 };
            let base = (pair * 2 + half) * 32;
            for l in 0..4 {
                let hi = qh[pair * 2 + half] as usize;
                let i1 = qs[qo + 2 * l] as usize | ((hi << (8 - 2 * l)) & 256);
                let i2 = qs[qo + 2 * l + 1] as usize | ((hi << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[i1].to_le_bytes();
                let g2 = IQ3S_GRID[i2].to_le_bytes();
                for j in 0..4 {
                    let s1 = if signs[so + l] & (1 << j) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s2 = if signs[so + l] & (1 << (j + 4)) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out[base + l * 8 + j] = db * g1[j] as f32 * s1;
                    out[base + l * 8 + 4 + j] = db * g2[j] as f32 * s2;
                }
            }
            qo += 8;
            so += 4;
        }
    }
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as Q4_K blocks, row-major.
pub fn gemv_q4_k(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0, "k must be a multiple of 256");
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * Q4_K_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/256 * 144 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; QK_K];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_q4_k(&row_data[b * Q4_K_BYTES..(b + 1) * Q4_K_BYTES], &mut tmp);
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            for l in 0..QK_K {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as Q6_K blocks, row-major.
pub fn gemv_q6_k(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0, "k must be a multiple of 256");
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * Q6_K_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/256 * 210 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; QK_K];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_q6_k(&row_data[b * Q6_K_BYTES..(b + 1) * Q6_K_BYTES], &mut tmp);
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            for l in 0..QK_K {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as Q8_0 blocks, row-major.
pub fn gemv_q8_0(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % Q8_0_VALUES, 0, "k must be a multiple of 32");
    let blocks_per_row = k / Q8_0_VALUES;
    let row_bytes = blocks_per_row * Q8_0_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/32 * 34 bytes");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; Q8_0_VALUES];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_q8_0(&row_data[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES], &mut tmp);
            let xb = &x[b * Q8_0_VALUES..(b + 1) * Q8_0_VALUES];
            for l in 0..Q8_0_VALUES {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as IQ4_XS blocks, row-major.
pub fn gemv_iq4_xs(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0, "k must be a multiple of 256");
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * IQ4_XS_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/256 * 136 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; QK_K];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_iq4_xs(
                &row_data[b * IQ4_XS_BYTES..(b + 1) * IQ4_XS_BYTES],
                &mut tmp,
            );
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            for l in 0..QK_K {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as IQ4_NL blocks, row-major.
pub fn gemv_iq4_nl(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % IQ4_NL_VALUES, 0, "k must be a multiple of 32");
    let blocks_per_row = k / IQ4_NL_VALUES;
    let row_bytes = blocks_per_row * IQ4_NL_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/32 * 18 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; IQ4_NL_VALUES];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_iq4_nl(
                &row_data[b * IQ4_NL_BYTES..(b + 1) * IQ4_NL_BYTES],
                &mut tmp,
            );
            let xb = &x[b * IQ4_NL_VALUES..(b + 1) * IQ4_NL_VALUES];
            for l in 0..IQ4_NL_VALUES {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as Q3_K blocks, row-major. The host
/// reference for the `gemv_q3_k` kernel and the CPU-oracle path.
pub fn gemv_q3_k(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0, "k must be a multiple of 256");
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * Q3_K_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/256 * 110 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; QK_K];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_q3_k(&row_data[b * Q3_K_BYTES..(b + 1) * Q3_K_BYTES], &mut tmp);
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            for l in 0..QK_K {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// `y[m] = W[m,k] · x[k]` with `W` stored as IQ3_S blocks, row-major.
pub fn gemv_iq3_s(w: &[u8], x: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % QK_K, 0, "k must be a multiple of 256");
    let blocks_per_row = k / QK_K;
    let row_bytes = blocks_per_row * IQ3_S_BYTES;
    assert_eq!(w.len(), m * row_bytes, "W must be m * k/256 * 110 bytes");
    assert_eq!(x.len(), k, "x must be k");

    let mut y = vec![0.0f32; m];
    let mut tmp = [0.0f32; QK_K];
    for (row, out) in y.iter_mut().enumerate() {
        let row_data = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            dequant_iq3_s(&row_data[b * IQ3_S_BYTES..(b + 1) * IQ3_S_BYTES], &mut tmp);
            let xb = &x[b * QK_K..(b + 1) * QK_K];
            for l in 0..QK_K {
                acc += tmp[l] * xb[l];
            }
        }
        *out = acc;
    }
    y
}

/// Fill `out` with deterministic pseudo-random bytes.
fn fill_random(rng: &mut Rng, out: &mut [u8]) {
    let mut i = 0;
    while i + 8 <= out.len() {
        out[i..i + 8].copy_from_slice(&rng.next_u64().to_le_bytes());
        i += 8;
    }
    if i < out.len() {
        let n = out.len() - i;
        let tail = rng.next_u64().to_le_bytes();
        out[i..].copy_from_slice(&tail[..n]);
    }
}

// d = 2^-10 and dmin = 2^-11 are exact binary16 values. With the random 6-bit
// scales this keeps dequantised weights in roughly [-0.1, 1.0], which is the
// range a real Q4_K_M tensor has.
const D_Q4K: [u8; 2] = 0x1400u16.to_le_bytes();
const DMIN_Q4K: [u8; 2] = 0x1000u16.to_le_bytes();
const D_Q5K: [u8; 2] = 0x1400u16.to_le_bytes();
const DMIN_Q5K: [u8; 2] = 0x1000u16.to_le_bytes();
const D_Q6K: [u8; 2] = 0x1400u16.to_le_bytes();
const D_Q8_0: [u8; 2] = 0x1400u16.to_le_bytes();
const D_Q3K: [u8; 2] = 0x1400u16.to_le_bytes();
const D_IQ3S: [u8; 2] = 0x1400u16.to_le_bytes();
const D_IQ4XS: [u8; 2] = 0x1400u16.to_le_bytes();
const D_IQ4NL: [u8; 2] = 0x1400u16.to_le_bytes();

/// Deterministic Q4_K block bytes: valid fp16 scales, random 6-bit headers and
/// random quants. Used by the parity and bench cases.
pub fn random_q4_k_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * Q4_K_BYTES];
    for b in out.chunks_mut(Q4_K_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_Q4K);
        b[2..4].copy_from_slice(&DMIN_Q4K);
    }
    out
}

/// Deterministic Q5_K block bytes: same scale range as Q4_K, random 6-bit
/// headers, `qh` bits and quants.
pub fn random_q5_k_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * Q5_K_BYTES];
    for b in out.chunks_mut(Q5_K_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_Q5K);
        b[2..4].copy_from_slice(&DMIN_Q5K);
    }
    out
}

/// Deterministic Q6_K block bytes.
pub fn random_q6_k_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * Q6_K_BYTES];
    for b in out.chunks_mut(Q6_K_BYTES) {
        fill_random(rng, b);
        b[208..210].copy_from_slice(&D_Q6K);
    }
    out
}

/// Deterministic Q8_0 block bytes.
pub fn random_q8_0_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * Q8_0_BYTES];
    for b in out.chunks_mut(Q8_0_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_Q8_0);
    }
    out
}

/// Deterministic Q3_K block bytes: random 2-bit quants, hmask and scales with a
/// valid fp16 super-scale at the end. The random 6-bit scales keep the
/// dequantised range close to a real Q3_K tensor.
pub fn random_q3_k_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * Q3_K_BYTES];
    for b in out.chunks_mut(Q3_K_BYTES) {
        fill_random(rng, b);
        b[108..110].copy_from_slice(&D_Q3K);
    }
    out
}

/// Deterministic IQ3_S block bytes: random grid indices, `qh`, signs and scale
/// nibbles, with a valid fp16 super-scale in front.
pub fn random_iq3_s_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * IQ3_S_BYTES];
    for b in out.chunks_mut(IQ3_S_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_IQ3S);
    }
    out
}

/// Deterministic IQ4_XS block bytes: random 6-bit sub-block scales and quants,
/// with a valid fp16 super-scale in front.
pub fn random_iq4_xs_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * IQ4_XS_BYTES];
    for b in out.chunks_mut(IQ4_XS_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_IQ4XS);
    }
    out
}

/// Deterministic IQ4_NL block bytes: random 4-bit codebook indices, with a
/// valid fp16 block scale in front.
pub fn random_iq4_nl_bytes(rng: &mut Rng, nblocks: usize) -> Vec<u8> {
    let mut out = vec![0u8; nblocks * IQ4_NL_BYTES];
    for b in out.chunks_mut(IQ4_NL_BYTES) {
        fill_random(rng, b);
        b[0..2].copy_from_slice(&D_IQ4NL);
    }
    out
}
