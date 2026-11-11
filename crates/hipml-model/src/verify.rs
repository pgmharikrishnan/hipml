// SPDX-License-Identifier: Apache-2.0
//! Load-time verification: device bytes against the file, and the two accessors
//! against host references.
//!
//! Everything here is deterministic. The round-trip sample is drawn from a fixed
//! seed, the lm_head vector is fixed, and the embedding token ids are fixed, so
//! two runs on the same file report the same numbers.

use std::path::Path;

use hipml_core::gguf::Gguf;
use hipml_core::quant::{
    dequant_iq3_s, dequant_iq4_nl, dequant_iq4_xs, dequant_q3_k, dequant_q4_k, dequant_q5_k,
    dequant_q6_k, dequant_q8_0, TensorType,
};
use hipml_core::Rng;

use crate::{Model, TOKEN_EMBD};

/// Host dequant of a whole block tensor, `blocks * block_values` f32.
fn dequant_all(ty: TensorType, raw: &[u8]) -> Result<Vec<f32>, String> {
    if ty == TensorType::F32 {
        return Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }
    let dequant: fn(&[u8], &mut [f32]) = match ty {
        TensorType::Q4K => dequant_q4_k,
        TensorType::Q5K => dequant_q5_k,
        TensorType::Q6K => dequant_q6_k,
        TensorType::Q8_0 => dequant_q8_0,
        TensorType::Q3K => dequant_q3_k,
        TensorType::IQ3S => dequant_iq3_s,
        TensorType::IQ4NL => dequant_iq4_nl,
        TensorType::IQ4XS => dequant_iq4_xs,
        other => return Err(format!("{}: no host dequant", other.name())),
    };
    let (values, bytes) = (ty.block_values(), ty.block_bytes());
    if bytes == 0 || raw.len() % bytes != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of {bytes}-byte blocks",
            ty.name(),
            raw.len()
        ));
    }
    let blocks = raw.len() / bytes;
    let mut out = vec![0.0f32; blocks * values];
    for b in 0..blocks {
        dequant(
            &raw[b * bytes..(b + 1) * bytes],
            &mut out[b * values..(b + 1) * values],
        );
    }
    Ok(out)
}

/// Host dequant of one row of a block format, `cols` values wide.
fn dequant_row(ty: TensorType, raw: &[u8], cols: usize) -> Result<Vec<f32>, String> {
    let out = dequant_all(ty, raw)?;
    if out.len() != cols {
        return Err(format!(
            "{}: {} bytes dequantises to {} values, row is {cols}",
            ty.name(),
            raw.len(),
            out.len()
        ));
    }
    Ok(out)
}

/// One tensor compared after upload.
pub struct RoundTrip {
    pub name: String,
    pub bytes: u64,
    pub file_hash: u64,
    pub device_hash: u64,
    pub matched: bool,
}

/// FNV-1a over raw bytes; cheap enough for a gigabyte and stable across runs.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Upload round-trip: draw `count` tensors with a fixed seed, read each one back
/// with `hipMemcpy`, and compare every byte against a fresh file read.
pub fn round_trip(
    model: &Model,
    path: &Path,
    count: usize,
    seed: u64,
) -> Result<Vec<RoundTrip>, String> {
    let n = model.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let take = count.min(n);
    let mut rng = Rng::new(seed);
    let mut picked: Vec<usize> = Vec::with_capacity(take);
    while picked.len() < take {
        let idx = (rng.next_u64() % n as u64) as usize;
        if !picked.contains(&idx) {
            picked.push(idx);
        }
    }
    picked.sort_unstable();

    let mut g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::with_capacity(take);
    for idx in picked {
        let t = &model.tensors()[idx];
        let file = g
            .read_tensor(t.name())
            .map_err(|e| format!("{}: {e}", t.name()))?;
        let device = t.read_back()?;
        // An expanded tensor holds f32 values the loader computed on the host,
        // so the comparison is against the same host dequant of the file row
        // set, not against the block bytes.
        let expected = if t.expanded() {
            let values = dequant_all(t.disk_ty(), &file)?;
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            file
        };
        let matched = expected.len() == device.len() && expected == device;
        out.push(RoundTrip {
            name: t.name().to_string(),
            bytes: t.nbytes(),
            file_hash: fnv1a(&expected),
            device_hash: fnv1a(&device),
            matched,
        });
    }
    Ok(out)
}

/// Result of comparing two f32 rows.
pub struct RowCompare {
    pub values: usize,
    pub exact: usize,
    pub max_abs: f32,
    /// Largest `|cpu|` in the row, so the relative figure is not divided by a
    /// near-zero weight.
    pub max_ref: f32,
}

impl RowCompare {
    pub fn max_rel(&self) -> f32 {
        if self.max_ref > 0.0 {
            self.max_abs / self.max_ref
        } else {
            self.max_abs
        }
    }
}

/// Compare a device row against a CPU reference of the same length.
pub fn compare_rows(device: &[f32], cpu: &[f32]) -> Option<RowCompare> {
    if device.len() != cpu.len() {
        return None;
    }
    let mut exact = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    for (&d, &c) in device.iter().zip(cpu) {
        if d == c {
            exact += 1;
        }
        let diff = (d - c).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let r = c.abs();
        if r > max_ref {
            max_ref = r;
        }
    }
    Some(RowCompare {
        values: device.len(),
        exact,
        max_abs,
        max_ref,
    })
}

/// Dequantise one `token_embd.weight` row straight from the file with the CPU
/// reference (`dequant_row`, i.e. the same kernels as `hipml-core`), the number
/// the device row is compared against. Handles every block type the loader can
/// place in the embedding, including the UD Q5_K / Q6_K pair and the IQ4 kinds.
pub fn cpu_embedding_row(path: &Path, token: u32) -> Result<Vec<f32>, String> {
    let mut g = Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let info = g
        .tensor(TOKEN_EMBD)
        .cloned()
        .ok_or_else(|| format!("no tensor {TOKEN_EMBD}"))?;
    let k = info.ne0() as usize;
    let rows = info.rows();
    if k % info.ty.block_values() != 0 {
        return Err(format!(
            "{TOKEN_EMBD}: ne0 {k} is not a multiple of {}",
            info.ty.block_values()
        ));
    }
    if token as u64 >= rows {
        return Err(format!(
            "{TOKEN_EMBD}: token {token} out of range (rows {rows})"
        ));
    }
    let row_bytes = info.nbytes / rows;
    let mut raw = vec![0u8; row_bytes as usize];
    g.read_tensor_range(TOKEN_EMBD, token as u64 * row_bytes, raw.len(), &mut raw)
        .map_err(|e| format!("{TOKEN_EMBD}: {e}"))?;
    dequant_row(info.ty, &raw, k)
}

/// Fixed lm_head probe vector: small zero-mean values, seeded so the top-5 is
/// reproducible run to run.
pub fn fixed_vector(k: usize) -> Vec<f32> {
    let mut rng = Rng::new(0x4C4F_4144_0000_0001);
    (0..k).map(|_| rng.normal_scaled(0.02)).collect()
}

/// Indices of the `n` largest logits, descending.
pub fn top_k(logits: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n);
    for (i, &v) in logits.iter().enumerate() {
        if top.len() == n && v <= top[n - 1].1 {
            continue;
        }
        let at = top.iter().position(|&(_, t)| v > t).unwrap_or(top.len());
        top.insert(at, (i, v));
        if top.len() > n {
            top.pop();
        }
    }
    top
}

/// Read a raw little-endian f32 vector from disk.
pub fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{}: {} bytes is not a whole number of f32 values",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_stable_and_input_sensitive() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"hipml"), fnv1a(b"hipml"));
        assert_ne!(fnv1a(b"hipml"), fnv1a(b"hipm1"));
    }

    #[test]
    fn top_k_is_descending_and_complete() {
        let v = [0.5, -1.0, 3.0, 2.0, 3.5];
        assert_eq!(top_k(&v, 3), vec![(4, 3.5), (2, 3.0), (3, 2.0)]);
        assert_eq!(top_k(&v, 9).len(), 5);
    }

    #[test]
    fn compare_counts_exact_values() {
        let c = compare_rows(&[1.0, 2.0, 3.5], &[1.0, 2.0, 3.0]).expect("same len");
        assert_eq!(c.values, 3);
        assert_eq!(c.exact, 2);
        assert!((c.max_abs - 0.5).abs() < 1e-6);
    }
}
