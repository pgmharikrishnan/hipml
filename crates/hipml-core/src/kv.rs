// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! Quantized KV-cache block formats: q8_0 and q4_0, encode and decode.
//!
//! Both formats copy the llama.cpp/ggml block layout exactly, so a cache packed
//! here is the same bytes `llama.cpp -ctk q8_0 -ctv q8_0` would hold:
//!
//! ```text
//! q8_0 block (34 bytes):  f16 d | 32 x i8   value = d * q
//! q4_0 block (18 bytes):  f16 d | 16 bytes  value = d * (q - 8)
//! ```
//!
//! q4_0 packs element `j` in the low nibble of byte `j` and element `j + 16` in
//! the high nibble, and `d` carries the sign of the largest-magnitude element
//! (`d = -vmax / 8`), so that extreme maps exactly onto the end of the `[-8, 7]`
//! range. That is the ggml convention, not a symmetric `max/7`; it is what makes
//! the bytes interchangeable with llama.cpp's q4_0 KV cache.
//!
//! ## Cache layout
//!
//! One layer of cache is two independent byte arrays, `K` and `V`, each
//!
//! ```text
//! [n_kv][seq][head_dim / 32] blocks
//! ```
//!
//! and the block for `(kvh, pos, chunk)` starts at byte
//!
//! ```text
//! ((kvh * seq + pos) * (head_dim / 32) + chunk) * BLOCK_BYTES
//! ```
//!
//! The head axis is outer, position next, the 32-value blocks contiguous along
//! `head_dim`. K and V are *not* interleaved: they are separate allocations with
//! separate pointers because a decode step streams each of them exactly once and
//! interleaving would only add a stride to both. `head_dim` must be a multiple
//! of 32 (256 and 64 are the shapes this engine runs).
//!
//! Per-token traffic for the Qwen3.8-27B hybrid (24 q heads, 4 kv heads, width 256,
//! 16 full-attention layers) is 65,536 B at f16, 34,816 B at q8 and 18,432 B at
//! q4: `2 * 4 * 16 * 256 * bytes_per_value`.

/// Values per quantization block.
pub const BLOCK: usize = 32;

/// The two cache encodings. `F16` is not stored here; the f32 reference and
/// `attn_decode_f32` cover the unquantized path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    Q8,
    Q4,
}

impl KvDtype {
    /// Bytes of one block of 32 values.
    pub const fn block_bytes(self) -> usize {
        match self {
            KvDtype::Q8 => 34,
            KvDtype::Q4 => 18,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            KvDtype::Q8 => "q8",
            KvDtype::Q4 => "q4",
        }
    }

    /// The integer the kernel ABI and the parity scalar table use: 0 q8, 1 q4.
    pub const fn id(self) -> usize {
        match self {
            KvDtype::Q8 => 0,
            KvDtype::Q4 => 1,
        }
    }

    /// Bytes of one `head_dim`-wide row (K or V) at one position.
    pub fn row_bytes(self, head_dim: usize) -> usize {
        assert_eq!(head_dim % BLOCK, 0, "head_dim must be a multiple of 32");
        head_dim / BLOCK * self.block_bytes()
    }

    /// Bytes of K plus V for one token across `n_kv * n_layers` cache rows.
    pub fn token_bytes(self, n_kv: usize, head_dim: usize, n_layers: usize) -> usize {
        2 * n_kv * n_layers * self.row_bytes(head_dim)
    }
}

/// f16 -> f32. Exact by construction; used on block scales only.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: mant * 2^-24, both exactly representable.
            let two_m24 = f32::from_bits((127 - 24) << 23);
            return if sign != 0 {
                -((mant as f32) * two_m24)
            } else {
                (mant as f32) * two_m24
            };
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> f16, round to nearest even. Overflow saturates to infinity.
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut exp = ((x >> 23) & 0xff) as i32;
    let mut mant = x & 0x007f_ffff;

    if exp == 0xff {
        let m = if mant == 0 {
            0
        } else {
            ((mant >> 13) as u16) | 1
        };
        return sign | 0x7c00 | m;
    }
    exp -= 127 - 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - exp;
        let half = 1u32 << (shift - 1);
        let mut m = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let half = 1u32 << 12;
    let mut m = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > half || (rem == half && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m as u16
}

/// The f16 scale of block `i`.
pub fn block_scale(dtype: KvDtype, bytes: &[u8], i: usize) -> f32 {
    let at = i * dtype.block_bytes();
    let raw = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    f16_to_f32(raw)
}

/// Encode `values` (a multiple of 32) into blocks. This is the definition of the
/// format; the kernel dequantizes with the matching arithmetic.
pub fn quantize(dtype: KvDtype, values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % BLOCK, 0, "values must be a multiple of 32");
    let bb = dtype.block_bytes();
    let mut out = vec![0u8; values.len() / BLOCK * bb];
    for (b, chunk) in values.chunks_exact(BLOCK).enumerate() {
        quantize_block(dtype, chunk, &mut out[b * bb..(b + 1) * bb]);
    }
    out
}

/// Encode one block of exactly 32 values.
pub fn quantize_block(dtype: KvDtype, xs: &[f32], out: &mut [u8]) {
    assert_eq!(xs.len(), BLOCK, "a block is 32 values");
    assert_eq!(out.len(), dtype.block_bytes(), "block byte count");
    match dtype {
        KvDtype::Q8 => {
            let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            for (j, &x) in xs.iter().enumerate() {
                let q = (x * id).round().clamp(-127.0, 127.0) as i8;
                out[2 + j] = q as u8;
            }
        }
        KvDtype::Q4 => {
            // ggml q4_0: the scale takes the sign of the largest-magnitude
            // element so that element lands exactly on q = 0 or q = 15.
            let vmax = xs
                .iter()
                .copied()
                .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
            let d = -vmax / 8.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            let mut q = [0u8; BLOCK];
            for (j, &x) in xs.iter().enumerate() {
                q[j] = ((x * id).round() + 8.0).clamp(0.0, 15.0) as u8;
            }
            for j in 0..BLOCK / 2 {
                out[2 + j] = q[j] | (q[j + BLOCK / 2] << 4);
            }
        }
    }
}

/// Value `j` of one encoded block, dequantized.
#[inline]
pub fn block_value(dtype: KvDtype, bytes: &[u8], j: usize) -> f32 {
    let d = block_scale(dtype, bytes, 0);
    match dtype {
        KvDtype::Q8 => d * (bytes[2 + j] as i8) as f32,
        KvDtype::Q4 => {
            let byte = bytes[2 + (j & 15)];
            let q = if j < 16 { byte & 0x0f } else { byte >> 4 };
            d * (q as f32 - 8.0)
        }
    }
}

/// One block decoded to 32 f32 values, scalar form of what the kernel does.
#[inline]
pub fn block_f32(dtype: KvDtype, src: &[u8]) -> [f32; BLOCK] {
    let mut out = [0.0f32; BLOCK];
    let d = block_scale(dtype, src, 0);
    match dtype {
        KvDtype::Q8 => {
            for (j, o) in out.iter_mut().enumerate() {
                *o = d * (src[2 + j] as i8) as f32;
            }
        }
        KvDtype::Q4 => {
            for j in 0..BLOCK / 2 {
                let byte = src[2 + j];
                out[j] = d * ((byte & 0x0f) as f32 - 8.0);
                out[j + BLOCK / 2] = d * ((byte >> 4) as f32 - 8.0);
            }
        }
    }
    out
}

/// Decode `bytes` back to f32 values.
pub fn dequantize(dtype: KvDtype, bytes: &[u8]) -> Vec<f32> {
    let bb = dtype.block_bytes();
    assert_eq!(bytes.len() % bb, 0, "truncated block");
    let mut out = vec![0.0f32; bytes.len() / bb * BLOCK];
    dequantize_into(dtype, bytes, &mut out);
    out
}

/// Decode `bytes` into `out` (already sized to the value count).
pub fn dequantize_into(dtype: KvDtype, bytes: &[u8], out: &mut [f32]) {
    let bb = dtype.block_bytes();
    assert_eq!(bytes.len() % bb, 0, "truncated block");
    assert_eq!(out.len(), bytes.len() / bb * BLOCK, "output length");
    for (b, dst) in out.chunks_exact_mut(BLOCK).enumerate() {
        let src = &bytes[b * bb..(b + 1) * bb];
        let d = block_scale(dtype, src, 0);
        match dtype {
            KvDtype::Q8 => {
                for j in 0..BLOCK {
                    dst[j] = d * (src[2 + j] as i8) as f32;
                }
            }
            KvDtype::Q4 => {
                for j in 0..BLOCK / 2 {
                    let byte = src[2 + j];
                    dst[j] = d * ((byte & 0x0f) as f32 - 8.0);
                    dst[j + BLOCK / 2] = d * ((byte >> 4) as f32 - 8.0);
                }
            }
        }
    }
}

/// Worst-case absolute error of one encoded block, in the same units as the
/// input. q8 is `amax/254`, q4 is `|d|/2 = amax/16` for values whose sign is
/// representable, and up to `amax/8` for the clipped opposite extreme.
pub fn max_abs_error(dtype: KvDtype, xs: &[f32]) -> f32 {
    let bytes = quantize(dtype, xs);
    let back = dequantize(dtype, &bytes);
    xs.iter()
        .zip(&back)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// Deterministic, format-valid filler for bandwidth benches: no host encode of
/// 3 GB, but every block has a real scale and real quantized payload, so the
/// kernel cannot take a denormal/NaN shortcut.
pub fn synthetic_blocks(dtype: KvDtype, n_blocks: usize, seed: u64) -> Vec<u8> {
    let bb = dtype.block_bytes();
    let mut out = vec![0u8; n_blocks * bb];
    let mut s = seed | 1;
    for b in 0..n_blocks {
        let base = b * bb;
        // Scale in [0.01, 0.05): normal f16, no denormals in the hot loop.
        let d = 0.01 + (next(&mut s) % 1000) as f32 * 0.00004;
        out[base..base + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..bb - 2 {
            out[base + 2 + j] = (next(&mut s) >> 24) as u8;
        }
    }
    out
}

fn next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}
