// SPDX-License-Identifier: Apache-2.0
//! Parity and bench cases for the tiled prefill GEMM.
//!
//! Same contract as `quant_parity.rs`: the reference is `ops::gemm_q*_k`, the
//! device side is `prefill_q*_k`, and both consume the identical deterministic
//! bytes. Small shapes keep the reference cheap; the one real shape checks that
//! a 5120-term dot stays inside the stated tolerance at production size.

use crate::ops;
use crate::parity::Tol;
use crate::quant::{
    random_iq3_s_bytes, random_iq4_nl_bytes, random_iq4_xs_bytes, random_q3_k_bytes,
    random_q4_k_bytes, random_q5_k_bytes, random_q6_k_bytes, random_q8_0_bytes, IQ3_S_BYTES,
    IQ4_NL_BYTES, IQ4_NL_VALUES, IQ4_XS_BYTES, Q3_K_BYTES, Q4_K_BYTES, Q5_K_BYTES, Q6_K_BYTES,
    Q8_0_BYTES, QK_K,
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
            "prefill_q4_k" | "gemm_skinny_q4_k" => self.k / QK_K * Q4_K_BYTES,
            "prefill_q5_k" | "gemm_skinny_q5_k" => self.k / QK_K * Q5_K_BYTES,
            "prefill_q6_k" | "gemm_skinny_q6_k" => self.k / QK_K * Q6_K_BYTES,
            "prefill_q8_0" | "gemm_skinny_q8_0" => self.k / 32 * Q8_0_BYTES,
            "prefill_q3_k" | "gemm_skinny_q3_k" => self.k / QK_K * Q3_K_BYTES,
            "prefill_iq3_s" | "gemm_skinny_iq3_s" => self.k / QK_K * IQ3_S_BYTES,
            "gemm_skinny_iq4_xs" => self.k / QK_K * IQ4_XS_BYTES,
            "gemm_skinny_iq4_nl" => self.k / IQ4_NL_VALUES * IQ4_NL_BYTES,
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

fn build_q5_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_q5_k(seed, m, k, tokens);
    let y = ops::gemm_q5_k(&w, &x, tokens, m, k);
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

fn inputs_q5_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q5_k_bytes(&mut rng, m * k / QK_K);
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

fn build_q3_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_q3_k(seed, m, k, tokens);
    let y = ops::gemm_q3_k(&w, &x, tokens, m, k);
    (w, x, y)
}

fn build_iq3_s(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_iq3_s(seed, m, k, tokens);
    let y = ops::gemm_iq3_s(&w, &x, tokens, m, k);
    (w, x, y)
}

fn build_iq4_xs(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_iq4_xs(seed, m, k, tokens);
    let y = ops::gemm_iq4_xs(&w, &x, tokens, m, k);
    (w, x, y)
}

fn build_iq4_nl(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (w, x) = inputs_iq4_nl(seed, m, k, tokens);
    let y = ops::gemm_iq4_nl(&w, &x, tokens, m, k);
    (w, x, y)
}

fn inputs_q3_k(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_q3_k_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

fn inputs_iq3_s(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_iq3_s_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

fn inputs_iq4_xs(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_iq4_xs_bytes(&mut rng, m * k / QK_K);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

fn inputs_iq4_nl(seed: u64, m: usize, k: usize, tokens: usize) -> (Vec<u8>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let w = random_iq4_nl_bytes(&mut rng, m * k / IQ4_NL_VALUES);
    let x: Vec<f32> = (0..tokens * k).map(|_| rng.normal()).collect();
    (w, x)
}

/// Parity: each format at a small shape, plus one real Qwen3.8-27B shape.
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
            name: "prefill_q5_k",
            label: "prefill_q5_k 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0007,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_q5_k,
            inputs: inputs_q5_k,
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
            name: "prefill_q5_k",
            label: "prefill_q5_k 64x512 t=40",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0008,
            m: 64,
            k: 512,
            tokens: 40,
            build: build_q5_k,
            inputs: inputs_q5_k,
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
        PrefillCase {
            name: "prefill_q3_k",
            label: "prefill_q3_k 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_0009,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_q3_k,
            inputs: inputs_q3_k,
        },
        PrefillCase {
            name: "prefill_iq3_s",
            label: "prefill_iq3_s 64x512 t=4",
            tol: Tol::DEFAULT,
            seed: 0x9F11_000A,
            m: 64,
            k: 512,
            tokens: 4,
            build: build_iq3_s,
            inputs: inputs_iq3_s,
        },
        // Two k-tiles at a production-ish k: also exercises the split-K path,
        // which `prefill_should_split` enables from `blocks >= 32` up.
        PrefillCase {
            name: "prefill_q3_k",
            label: "prefill_q3_k 64x4096 t=4",
            tol: LARGE_TOL,
            seed: 0x9F11_000B,
            m: 64,
            k: 4096,
            tokens: 4,
            build: build_q3_k,
            inputs: inputs_q3_k,
        },
        PrefillCase {
            name: "prefill_iq3_s",
            label: "prefill_iq3_s 64x4096 t=4",
            tol: LARGE_TOL,
            seed: 0x9F11_000C,
            m: 64,
            k: 4096,
            tokens: 4,
            build: build_iq3_s,
            inputs: inputs_iq3_s,
        },
    ]
}

/// Skinny quantized GEMM parity: `R <= 16` activation rows, the same reference
/// (`ops::gemm_q*_k`) and the same block layouts as prefill, but a 16-row
/// WMMA token tile with an fp16 operand path. `m` is deliberately not a
/// multiple of any sweep geometry so the row tail is covered.
///
/// The tolerance is the fp16-operand budget, not `Tol::DEFAULT`: the fragment
/// conversion loses ~1e-3 relative before the accumulation order changes.
pub fn skinny_cases() -> Vec<PrefillCase> {
    // fp16 operands: ~1e-2 absolute on a 512-term dot, and a near-zero output
    // moves the ratio by an order of magnitude. O(1) below a layout bug, which
    // swaps or zeroes whole token rows.
    let small = Tol::new(5e-2, 5e-1);
    // Production k (5120 terms): the same few 1e-2 relative to the output scale.
    let large = Tol::new(2.5e-1, 5e-2);
    vec![
        PrefillCase {
            name: "gemm_skinny_q4_k",
            label: "skinny_q4_k 200x512 r=1",
            tol: small,
            seed: 0x5C11_0001,
            m: 200,
            k: 512,
            tokens: 1,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "gemm_skinny_q4_k",
            label: "skinny_q4_k 200x512 r=3",
            tol: small,
            seed: 0x5C11_0002,
            m: 200,
            k: 512,
            tokens: 3,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "gemm_skinny_q4_k",
            label: "skinny_q4_k 200x512 r=16",
            tol: small,
            seed: 0x5C11_0003,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
        PrefillCase {
            name: "gemm_skinny_q6_k",
            label: "skinny_q6_k 200x512 r=2",
            tol: small,
            seed: 0x5C11_0004,
            m: 200,
            k: 512,
            tokens: 2,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
        PrefillCase {
            name: "gemm_skinny_q6_k",
            label: "skinny_q6_k 200x512 r=16",
            tol: small,
            seed: 0x5C11_0005,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_q6_k,
            inputs: inputs_q6_k,
        },
        PrefillCase {
            name: "gemm_skinny_q8_0",
            label: "skinny_q8_0 200x512 r=4",
            tol: small,
            seed: 0x5C11_0006,
            m: 200,
            k: 512,
            tokens: 4,
            build: build_q8_0,
            inputs: inputs_q8_0,
        },
        PrefillCase {
            name: "gemm_skinny_q8_0",
            label: "skinny_q8_0 200x512 r=16",
            tol: small,
            seed: 0x5C11_0007,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_q8_0,
            inputs: inputs_q8_0,
        },
        // The dynamic-quant formats. Their weight staging goes through the
        // prefill reader, so one case each at the full 16-row tile covers the
        // shape the batched server actually launches.
        PrefillCase {
            name: "gemm_skinny_q5_k",
            label: "skinny_q5_k 200x512 r=16",
            tol: small,
            seed: 0x5C11_0009,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_q5_k,
            inputs: inputs_q5_k,
        },
        PrefillCase {
            name: "gemm_skinny_iq4_xs",
            label: "skinny_iq4_xs 200x512 r=16",
            tol: small,
            seed: 0x5C11_000A,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_iq4_xs,
            inputs: inputs_iq4_xs,
        },
        PrefillCase {
            name: "gemm_skinny_iq4_nl",
            label: "skinny_iq4_nl 200x512 r=16",
            tol: small,
            seed: 0x5C11_000B,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_iq4_nl,
            inputs: inputs_iq4_nl,
        },
        PrefillCase {
            name: "gemm_skinny_q3_k",
            label: "skinny_q3_k 200x512 r=16",
            tol: small,
            seed: 0x5C11_000C,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_q3_k,
            inputs: inputs_q3_k,
        },
        PrefillCase {
            name: "gemm_skinny_iq3_s",
            label: "skinny_iq3_s 200x512 r=16",
            tol: small,
            seed: 0x5C11_000D,
            m: 200,
            k: 512,
            tokens: 16,
            build: build_iq3_s,
            inputs: inputs_iq3_s,
        },
        // One production-size dot (5120 terms) so a k-tile or fragment-layout
        // bug that only shows at full k cannot pass on the small cases.
        PrefillCase {
            name: "gemm_skinny_q4_k",
            label: "skinny_q4_k 5120x5120 r=4",
            tol: large,
            seed: 0x5C11_0008,
            m: 5120,
            k: 5120,
            tokens: 4,
            build: build_q4_k,
            inputs: inputs_q4_k,
        },
    ]
}

/// The three Qwen3.8-27B ffn shapes at the prompt length being measured.
/// These are timed only; parity at the same shapes would need minutes of host
/// reference.
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

/// Weight builder and host GEMM for a format, by case name.
fn gu_format(
    name: &str,
) -> Option<(
    fn(u64, usize, usize, usize) -> (Vec<u8>, Vec<f32>),
    fn(&[u8], &[f32], usize, usize, usize) -> Vec<f32>,
)> {
    Some(match name {
        "prefill_q4_k" => (inputs_q4_k, ops::gemm_q4_k),
        "prefill_q5_k" => (inputs_q5_k, ops::gemm_q5_k),
        "prefill_q6_k" => (inputs_q6_k, ops::gemm_q6_k),
        "prefill_q8_0" => (inputs_q8_0, ops::gemm_q8_0),
        "prefill_q3_k" => (inputs_q3_k, ops::gemm_q3_k),
        "prefill_iq3_s" => (inputs_iq3_s, ops::gemm_iq3_s),
        "prefill_iq4_xs" => (inputs_iq4_xs, ops::gemm_iq4_xs),
        "prefill_iq4_nl" => (inputs_iq4_nl, ops::gemm_iq4_nl),
        _ => return None,
    })
}

/// A fused gate+up case: two weight matrices over one activation tile, and the
/// swiGLU the fused epilogue has to reproduce.
pub struct GuCase {
    pub label: &'static str,
    pub gate: &'static str,
    pub up: &'static str,
    pub seed: u64,
    pub m: usize,
    pub k: usize,
    pub tokens: usize,
}

impl GuCase {
    /// The converted-activation entry for one side of the pair.
    pub fn entry(name: &str) -> &'static str {
        match name {
            "prefill_q4_k" => "prefill_q4_k_w2h",
            "prefill_q5_k" => "prefill_q5_k_w2h",
            "prefill_q6_k" => "prefill_q6_k_w2h",
            "prefill_q8_0" => "prefill_q8_0_w2h",
            "prefill_q3_k" => "prefill_q3_k_w2h",
            "prefill_iq3_s" => "prefill_iq3_s_w2h",
            "prefill_iq4_xs" => "prefill_iq4_xs_w2h",
            "prefill_iq4_nl" => "prefill_iq4_nl_w2h",
            other => panic!("{other}: no converted-activation entry"),
        }
    }

    /// Weights for both matrices, the shared activation rows, and
    /// `act = silu(gate) * up` on the host.
    pub fn reference(&self) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
        let (gi, gg) = gu_format(self.gate).expect("gate format");
        let (ui, ug) = gu_format(self.up).expect("up format");
        let (wg, x) = gi(self.seed, self.m, self.k, self.tokens);
        let (wu, _) = ui(self.seed + 1, self.m, self.k, self.tokens);
        let yg = gg(&wg, &x, self.tokens, self.m, self.k);
        let yu = ug(&wu, &x, self.tokens, self.m, self.k);
        let act = yg
            .iter()
            .zip(&yu)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        (wg, wu, x, act)
    }
}

/// Fused gate+up parity: the two formats the UD mix pairs most often, plus the
/// same-format pair the stock file uses, and a token count well inside one
/// block so the epilogue's row mask is exercised.
pub fn gu_cases() -> Vec<GuCase> {
    vec![
        GuCase {
            label: "prefill_gu q4_k/q4_k 256x512 t=8",
            gate: "prefill_q4_k",
            up: "prefill_q4_k",
            seed: 0x9F13_0001,
            m: 256,
            k: 512,
            tokens: 8,
        },
        GuCase {
            label: "prefill_gu q4_k/q5_k 256x512 t=8",
            gate: "prefill_q4_k",
            up: "prefill_q5_k",
            seed: 0x9F13_0002,
            m: 256,
            k: 512,
            tokens: 8,
        },
        GuCase {
            label: "prefill_gu iq4_xs/iq4_xs 256x512 t=8",
            gate: "prefill_iq4_xs",
            up: "prefill_iq4_xs",
            seed: 0x9F13_0003,
            m: 256,
            k: 512,
            tokens: 8,
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
    fn skinny_cases_are_deterministic_and_sized() {
        for case in skinny_cases() {
            assert!(case.tokens >= 1 && case.tokens <= 16, "{}: r", case.label);
            let (w, x, y) = case.reference();
            assert_eq!(w.len(), case.weight_bytes(), "{}: w", case.label);
            assert_eq!(x.len(), case.tokens * case.k, "{}: x", case.label);
            assert_eq!(y.len(), case.tokens * case.m, "{}: y", case.label);
            let (w2, x2, y2) = case.reference();
            assert_eq!((w, x, y), (w2, x2, y2));
        }
    }

    #[test]
    fn gu_cases_are_deterministic_and_sized() {
        for case in gu_cases() {
            let (wg, wu, x, act) = case.reference();
            assert_eq!(x.len(), case.tokens * case.k, "{}: x", case.label);
            assert_eq!(act.len(), case.tokens * case.m, "{}: act", case.label);
            assert!(!wg.is_empty() && !wu.is_empty(), "{}: w", case.label);
            assert_eq!((wg, wu, x, act), case.reference());
        }
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
