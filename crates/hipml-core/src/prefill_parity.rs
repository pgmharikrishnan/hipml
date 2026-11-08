//! Parity and bench cases for the tiled prefill GEMM.
//!
//! Same contract as `quant_parity.rs`: the reference is `ops::gemm_q*_k`, the
//! device side is `prefill_q*_k`, and both consume the identical deterministic
//! bytes. Small shapes keep the reference cheap; the one real shape checks that
//! a 5120-term dot stays inside the stated tolerance at production size.

use crate::ops;
use crate::parity::Tol;
use crate::quant::{
    random_q4_k_bytes, random_q6_k_bytes, random_q8_0_bytes, Q4_K_BYTES, Q6_K_BYTES, Q8_0_BYTES,
    QK_K,
};
use crate::rng::Rng;

/// Error budget at production size. Both sides are f32 and sum 5120 products;
/// a near-zero output can move by ~1e-3 through cancellation, while a layout bug
/// moves values by O(1).
pub const LARGE_TOL: Tol = Tol::new(2e-3, 3e-2);

/// A prefill case: quantized weights `[m, k]`, `tokens` input rows, reference
/// output `[tokens, m]`.
pub struct PrefillCase {
    pub name: &'static str,
    pub label: &'static str,
    pub tol: Tol,
    pub seed: u64,
    pub m: usize,
    pub k: usize,
    pub tokens: usize,
    pub build: fn(u64, usize, usize, usize) -> (Vec<u8>, Vec<f32>, Vec<f32>),
    /// Weights and inputs only. Bench cases use this: the host reference at
    /// M=2048 would be minutes of work that says nothing about the GPU.
    pub inputs: fn(u64, usize, usize, usize) -> (Vec<u8>, Vec<f32>),
}

impl PrefillCase {
    pub fn reference(&self) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        (self.build)(self.seed, self.m, self.k, self.tokens)
    }

    pub fn weight_bytes(&self) -> usize {
        self.m * self.row_bytes()
    }

    pub fn row_bytes(&self) -> usize {
        match self.name {
            "prefill_q4_k" => self.k / QK_K * Q4_K_BYTES,
            "prefill_q6_k" => self.k / QK_K * Q6_K_BYTES,
            "prefill_q8_0" => self.k / 32 * Q8_0_BYTES,
            _ => 0,
        }
    }

    /// Multiply-accumulates: the prefill roofline is measured in these.
    pub fn flops(&self) -> u64 {
        2 * self.tokens as u64 * self.m as u64 * self.k as u64
    }
}

fn build_q4_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_q4_k(seed, m, k, tokens);
    let y = ops::gemm_q4_k(&w, &x, tokens, m, k);
    (w, x, y)
}

fn build_q6_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_q6_k(seed, m, k, tokens);
    let y = ops::gemm_q6_k(&w, &x, tokens, m, k);
    (w, x, y)
}

fn build_q8_0(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_q8_0(seed, m, k, tokens);
    let y = ops::gemm_q8_0(&w, &x, tokens, m, k);
    (w, x, y)
}

fn inputs_q4_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q4_k_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

fn inputs_q6_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q6_k_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

fn inputs_q8_0(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q8_0_bytes(&mut rng, m * k / 32);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

/// Parity: three formats at a small shape, plus one real Qwen3.8-27B shape.
pub fn prefill_cases() -> Vec<PrefillCase> {
    vec![
        PrefillCase {
            name: "prefill_q4_k",
            label: "prefill_q4_k 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0001,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "prefill_q6_k",
            label: "prefill_q6_k 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0002,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
        PrefillCase {
            name: "prefill_q8_0",
            label: "prefill_q8_0 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0003,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_q8_0,
            inputs: inputs_q8_0,
        },
        PrefillCase {
            name: "prefill_q4_k",
            label: "prefill_q4_k 64x512 t=40",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0005,
            m: 64,
            k: 512,
            tokens: 40,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "prefill_q6_k",
            label: "prefill_q6_k 64x512 t=40",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0006,
            m: 64,
            k: 512,
            tokens: 40,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
        PrefillCase {
            name: "prefill_q6_k",
            label: "prefill_q6_k 17408x5120 t=2",
            tol: LARGE_TOL,
            seed: 0x9F11_0004,
            m: 17408,
            k: 5120,
            tokens: 2,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
    ]
}

/// The three Qwen3.8-27B ffn shapes at the prompt length being measured. These are
/// timed only; parity at the same shapes would need minutes of host reference.
pub fn qwen38_prefill_cases(tokens: usize) -> Vec<PrefillCase> {
    vec![
        PrefillCase {
            name: "prefill_q6_k",
            label: "ffn_up Q6_K 17408x5120",
            tol: LARGE_TOL,
            seed: 0x9F12_0001,
            m: 17408,
            k: 5120,
            tokens,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
        PrefillCase {
            name: "prefill_q4_k",
            label: "ffn_gate Q4_K 5120x17408",
            tol: LARGE_TOL,
            seed: 0x9F12_0002,
            m: 5120,
            k: 17408,
            tokens,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "prefill_q6_k",
            label: "ffn_down Q6_K 17408x5120",
            tol: LARGE_TOL,
            seed: 0x9F12_0003,
            m: 17408,
            k: 5120,
            tokens,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parity::compare;

    #[test]
    fn prefill_reference_matches_gemv_at_one_token() {
        // With one token the GEMM must reduce to the existing GEMV exactly.
        let mut rng = Rng::new(7);
        let (m, k) = (64usize, 512usize);
        let w = random_q4_k_bytes(&mut rng, m * k / QK_K);
        let x: Vec<f32> = (0..k).map(|_| rng.normal()).collect();
        let gemv = crate::quant::gemv_q4_k(&w, &x, m, k);
        let gemm = ops::gemm_q4_k(&w, &x, 1, m, k);
        compare(&gemv, &gemm, crate::parity::Tol::TIGHT).expect("1-token GEMM == GEMV");
    }

    #[test]
    fn small_cases_are_deterministic_and_sized() {
        for case in prefill_cases().into_iter().filter(|c| c.m == 64) {
            let (w, x, y) = case.reference();
            assert_eq!(w.len(), case.weight_bytes());
            assert_eq!(x.len(), case.tokens * case.k);
            assert_eq!(y.len(), case.tokens * case.m);
            let (w2, x2, y2) = case.reference();
            assert_eq!((w, x, y), (w2, x2, y2));
        }
    }
}
