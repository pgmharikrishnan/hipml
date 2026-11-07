//! Parity and bench cases for the quantized GEMV kernels.
//!
//! Kept out of `parity.rs` only to keep both files readable; `parity.rs`
//! re-exports these so `use hipml_core::parity::quant_cases` is the one path
//! callers need.
//!
//! The inputs are deterministic raw block bytes, not quantize-then-check: the
//! hand-computed vectors in `tests/quant_reference.rs` pin the block layout, and
//! these cases pin the kernel against `gemv_q4_k`/`gemv_q6_k`.

use crate::parity::Tol;
use crate::quant::{
    gemv_q4_k, gemv_q6_k, gemv_q8_0, random_q4_k_bytes, random_q6_k_bytes, random_q8_0_bytes,
    Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES, QK_K,
};
use crate::rng::Rng;

/// Error budget for the 17408x5120 cases. Both sides are f32 and sum 5120
/// terms; a row whose true value is near zero can legitimately differ by ~1e-3
/// from the sequential reference because of cancellation, while a row of
/// magnitude ~100 is within ~1e-5 relative. The absolute term is the meaningful
/// one; a layout bug moves values by O(1), not O(1e-3).
const LARGE_TOL: Tol = Tol::new(2e-3, 3e-2);

/// A quantized GEMV case: raw weight bytes, one input vector, reference output.
pub struct QuantCase {
    /// Op name, matching the manifest and the reference function.
    pub name: &'static str,
    /// Human label for the bench table.
    pub label: &'static str,
    pub tol: Tol,
    pub seed: u64,
    pub m: usize,
    pub k: usize,
    /// `(seed, m, k) -> (weight bytes, x, reference y)`.
    pub build: fn(u64, usize, usize) -> (Vec<u8>, Vec<f32>, Vec<f32>),
}

impl QuantCase {
    pub fn reference(&self) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        (self.build)(self.seed, self.m, self.k)
    }

    /// Quantized weight bytes per row.
    pub fn row_bytes(&self) -> usize {
        match self.name {
            "gemv_q4_k" => self.k / QK_K * Q4_K_BYTES,
            "gemv_q6_k" => self.k / QK_K * Q6_K_BYTES,
            "gemv_q8_0" => self.k / 32 * Q8_0_BYTES,
            _ => 0,
        }
    }

    /// Total bytes the kernel streams.
    pub fn weight_bytes(&self) -> usize {
        self.m * self.row_bytes()
    }
}

fn build_q4_k(seed: u64, m: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q4_k_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..k).map(|_| rng.normal()).collect();
    let y = gemv_q4_k(&w, &x, m, k);
    (w, x, y)
}

fn build_q6_k(seed: u64, m: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q6_k_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..k).map(|_| rng.normal()).collect();
    let y = gemv_q6_k(&w, &x, m, k);
    (w, x, y)
}

fn build_q8_0(seed: u64, m: usize, k: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q8_0_bytes(&mut rng, m * k / 32);
    let x: Vec<f32> = (0..k).map(|_| rng.normal()).collect();
    let y = gemv_q8_0(&w, &x, m, k);
    (w, x, y)
}

/// Parity: the small shapes that run fast, plus the real ffn shape.
pub fn quant_cases() -> Vec<QuantCase> {
    vec![
        QuantCase {
            name: "gemv_q4_k",
            label: "gemv_q4_k 64x512",
            tol: Tol::DEFAULT,
            seed: 0x4B51_0001,
            m: 64,
            k: 512,
            build: build_q4_k,
        },
        QuantCase {
            name: "gemv_q6_k",
            label: "gemv_q6_k 64x512",
            tol: Tol::DEFAULT,
            seed: 0x4B51_0002,
            m: 64,
            k: 512,
            build: build_q6_k,
        },
        QuantCase {
            name: "gemv_q8_0",
            label: "gemv_q8_0 64x512",
            tol: Tol::DEFAULT,
            seed: 0x4B51_0005,
            m: 64,
            k: 512,
            build: build_q8_0,
        },
        QuantCase {
            name: "gemv_q4_k",
            label: "gemv_q4_k 17408x5120",
            tol: LARGE_TOL,
            seed: 0x4B51_0003,
            m: 17408,
            k: 5120,
            build: build_q4_k,
        },
        QuantCase {
            name: "gemv_q6_k",
            label: "gemv_q6_k 17408x5120",
            tol: LARGE_TOL,
            seed: 0x4B51_0004,
            m: 17408,
            k: 5120,
            build: build_q6_k,
        },
    ]
}

/// Production shapes for the bench table. These are never compared against the
/// CPU reference in `hipml bench`; parity at 17408x5120 already does that.
pub fn quant_bench_cases() -> Vec<QuantCase> {
    vec![
        QuantCase {
            name: "gemv_q6_k",
            label: "ffn_up Q6_K 17408x5120",
            tol: Tol::DEFAULT,
            seed: 0xB001_0001,
            m: 17408,
            k: 5120,
            build: build_q6_k,
        },
        QuantCase {
            name: "gemv_q4_k",
            label: "ffn_gate Q4_K 5120x17408",
            tol: Tol::DEFAULT,
            seed: 0xB001_0002,
            m: 5120,
            k: 17408,
            build: build_q4_k,
        },
        QuantCase {
            name: "gemv_q6_k",
            label: "ffn_down Q6_K 5120x17408",
            tol: Tol::DEFAULT,
            seed: 0xB001_0003,
            m: 5120,
            k: 17408,
            build: build_q6_k,
        },
        QuantCase {
            name: "gemv_q4_k",
            label: "ffn_down Q4_K 17408x5120",
            tol: Tol::DEFAULT,
            seed: 0xB001_0004,
            m: 17408,
            k: 5120,
            build: build_q4_k,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_quant_cases_build_and_are_deterministic() {
        for case in quant_cases().into_iter().filter(|c| c.m == 64) {
            let (w, x, y) = case.reference();
            assert_eq!(w.len(), case.weight_bytes());
            assert_eq!(x.len(), case.k);
            assert_eq!(y.len(), case.m);
            let (w2, x2, y2) = case.reference();
            assert_eq!(w, w2);
            assert_eq!(x, x2);
            assert_eq!(y, y2);
        }
    }
}
