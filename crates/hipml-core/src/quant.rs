// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! Quantized block formats from GGML, and their reference dequantisation.
//!
//! Only the formats Qwen3.8-27B Q4_K_M actually stores are here: Q4_K, Q6_K,
//! Q8_0 (and Q5_K's size, so the reader can walk its bytes). The layout below is
//! the on-disk ggml layout, byte for byte — no padding, no reordering. Getting
//! the 6-bit unpacking in Q4_K and the qh bit planes in Q6_K right is the whole
//! point of this module, so the hand-computed vectors live in
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

/// A tensor type as it appears in a GGUF tensor-info record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    /// Anything else: the reader can list it but not size it.
    Unsupported(u32),
}

impl TensorType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => TensorType::F32,
            8 => TensorType::Q8_0,
            12 => TensorType::Q4K,
            13 => TensorType::Q5K,
            14 => TensorType::Q6K,
            other => TensorType::Unsupported(other),
        }
    }

    pub fn id(self) -> u32 {
        match self {
            TensorType::F32 => 0,
            TensorType::Q8_0 => 8,
            TensorType::Q4K => 12,
            TensorType::Q5K => 13,
            TensorType::Q6K => 14,
            TensorType::Unsupported(id) => id,
        }
    }

    pub fn name(self) -> String {
        match self {
            TensorType::F32 => "F32".to_string(),
            TensorType::Q8_0 => "Q8_0".to_string(),
            TensorType::Q4K => "Q4_K".to_string(),
            TensorType::Q5K => "Q5_K".to_string(),
            TensorType::Q6K => "Q6_K".to_string(),
            TensorType::Unsupported(id) => format!("type_{id}"),
        }
    }

    /// Values per block (1 for F32), or 0 when unsupported.
    pub fn block_values(self) -> usize {
        match self {
            TensorType::F32 => 1,
            TensorType::Q8_0 => Q8_0_VALUES,
            TensorType::Q4K | TensorType::Q5K | TensorType::Q6K => QK_K,
            TensorType::Unsupported(id) => size_only(id).0,
        }
    }

    /// Bytes per block, or 0 when unsupported.
    pub fn block_bytes(self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::Q8_0 => Q8_0_BYTES,
            TensorType::Q4K => Q4_K_BYTES,
            TensorType::Q5K => Q5_K_BYTES,
            TensorType::Q6K => Q6_K_BYTES,
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
/// report can name it: F16, Q4_0, Q2_K, Q3_K, IQ4_NL, IQ3_S, IQ4_XS.
///
/// The IQ entries matter because Unsloth's "UD" dynamic quants mix tensor
/// types per layer; sizing them is metadata only, not a dequant path.
/// `(0, 0)` for anything else.
fn size_only(id: u32) -> (usize, usize) {
    match id {
        1 => (1, 2),
        2 => (32, 18),
        10 => (QK_K, 84),
        11 => (QK_K, 110),
        20 => (32, 18),    // IQ4_NL
        21 => (QK_K, 110), // IQ3_S
        23 => (QK_K, 136), // IQ4_XS
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
