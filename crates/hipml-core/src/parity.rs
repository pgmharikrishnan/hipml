//! The parity harness: one table of cases that both the reference and the GPU
//! run, so "does the kernel match" is a single command and a single verdict.
//!
//! M0 rule: no kernel ships without a case here, and a case that only ever runs
//! on the CPU is reported as skipped, never as passing.

use crate::config::Config;
use crate::delta::{gdn_ar_f32, l2norm_f32, ssm_conv1d_f32, ssm_layer_f32, SsmState, SsmW};
use crate::mat::{Bytes, Kind, Mat};
use crate::ops;
use crate::rng::Rng;
use crate::tensor::Tensor;

pub use crate::prefill_parity::{qwen38_prefill_cases, prefill_cases, PrefillCase};
pub use crate::quant_parity::{quant_bench_cases, quant_cases, QuantCase};

/// Absolute and relative tolerance for a case.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub abs: f32,
    pub rel: f32,
}

impl Tol {
    pub const fn new(abs: f32, rel: f32) -> Self {
        Self { abs, rel }
    }

    /// Default for reductions where the GPU changes the summation order.
    pub const DEFAULT: Tol = Tol::new(1e-4, 1e-3);

    /// For elementwise work that should be bit-comparable modulo fma.
    pub const TIGHT: Tol = Tol::new(1e-6, 1e-5);

    /// For exact outcomes (indices).
    pub const EXACT: Tol = Tol::new(0.0, 0.0);
}

/// How far apart two results are.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub max_abs: f32,
    pub max_rel: f32,
}

impl Stats {
    pub fn within(&self, tol: Tol) -> bool {
        self.max_abs <= tol.abs || self.max_rel <= tol.rel
    }
}

/// Compare two equal-length results. Returns the stats either way so a failure
/// can print how far off it was.
pub fn compare(expected: &[f32], actual: &[f32], tol: Tol) -> Result<Stats, Stats> {
    assert_eq!(
        expected.len(),
        actual.len(),
        "length mismatch: reference {} vs device {}",
        expected.len(),
        actual.len()
    );
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (e, a) in expected.iter().zip(actual) {
        let abs = (e - a).abs();
        let denom = e.abs().max(1e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / denom);
    }
    let stats = Stats { max_abs, max_rel };
    if stats.within(tol) {
        Ok(stats)
    } else {
        Err(stats)
    }
}

/// One parity case: deterministic inputs, the reference result, and the
/// tolerances a device result must meet.
pub struct OpCase {
    pub name: &'static str,
    pub tol: Tol,
    pub seed: u64,
    /// Scalar arguments a kernel needs beyond the tensors (eps, pos, theta…).
    pub scalars: &'static [(&'static str, f32)],
    /// `(named inputs, expected flat output)`. Input order and names are part of
    /// the contract the device launcher reads.
    pub build: fn(&mut Rng) -> (Vec<Tensor>, Vec<f32>),
}

impl OpCase {
    /// Run the reference side of this case.
    pub fn reference(&self) -> (Vec<Tensor>, Vec<f32>) {
        let mut rng = Rng::new(self.seed);
        (self.build)(&mut rng)
    }

    /// Scalar lookup with a clear panic when the case forgot to declare it.
    pub fn scalar(&self, key: &str) -> f32 {
        self.scalars
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| panic!("case {} has no scalar {key}", self.name))
    }
}

/// Every op the engine can run, with shapes chosen to expose tail handling:
/// sizes are deliberately not all powers of two.
pub fn cases() -> Vec<OpCase> {
    let mut out = vec![
        OpCase {
            name: "gemv_f32",
            tol: Tol::DEFAULT,
            seed: 0x9E37_79B9_7F4A_7C15,
            scalars: &[("m", 64.0), ("k", 512.0)],
            build: |rng| {
                let (m, k) = (64usize, 512usize);
                let w = Tensor::from_fn(vec![m, k], |_| rng.normal_scaled(0.02));
                let x = Tensor::from_fn(vec![k], |_| rng.normal());
                let expected = ops::gemv_f32(w.data(), x.data(), m, k);
                (vec![w, x], expected)
            },
        },
        OpCase {
            name: "gemm_f32",
            tol: Tol::DEFAULT,
            seed: 0x9E37_79B9_7F4A_7C15,
            scalars: &[("r", 4.0), ("m", 64.0), ("k", 512.0)],
            build: |rng| {
                let (r, m, k) = (4usize, 64usize, 512usize);
                let w = Tensor::from_fn(vec![m, k], |_| rng.normal_scaled(0.02));
                let x = Tensor::from_fn(vec![r, k], |_| rng.normal());
                let expected = ops::gemm_f32(w.data(), x.data(), r, m, k);
                (vec![w, x], expected)
            },
        },
        OpCase {
            name: "rmsnorm_f32",
            tol: Tol::DEFAULT,
            seed: 0x0123_4567_89AB_CDEF,
            scalars: &[("n", 2048.0), ("eps", 1e-6)],
            build: |rng| {
                let n = 2048usize;
                let x = Tensor::from_fn(vec![n], |_| rng.normal());
                let w = Tensor::from_fn(vec![n], |_| 1.0 + rng.normal_scaled(0.1));
                let expected = ops::rmsnorm_f32(x.data(), w.data(), 1e-6);
                (vec![x, w], expected)
            },
        },
        OpCase {
            name: "rope_neox_f32",
            tol: Tol::TIGHT,
            seed: 0xDEAD_BEEF_CAFE_0001,
            scalars: &[
                ("n_heads", 8.0),
                ("head_dim", 128.0),
                ("pos", 37.0),
                ("theta_base", 10000.0),
            ],
            build: |rng| {
                let (n_heads, head_dim, pos, theta_base) = (8usize, 128usize, 37usize, 10000.0f32);
                let x = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let expected = ops::rope_neox_f32(x.data(), n_heads, head_dim, pos, theta_base);
                (vec![x], expected)
            },
        },
        OpCase {
            name: "silu_mul_f32",
            tol: Tol::TIGHT,
            seed: 0x1111_2222_3333_4444,
            scalars: &[("n", 4096.0)],
            build: |rng| {
                let n = 4096usize;
                let gate = Tensor::from_fn(vec![n], |_| rng.normal_scaled(1.5));
                let up = Tensor::from_fn(vec![n], |_| rng.normal_scaled(0.5));
                let expected = ops::silu_mul_f32(gate.data(), up.data());
                (vec![gate, up], expected)
            },
        },
        OpCase {
            name: "softmax_f32",
            tol: Tol::DEFAULT,
            seed: 0x5555_6666_7777_8888,
            scalars: &[("n", 1024.0)],
            build: |rng| {
                let n = 1024usize;
                let x = Tensor::from_fn(vec![n], |_| rng.normal_scaled(3.0));
                let expected = ops::softmax_f32(x.data());
                (vec![x], expected)
            },
        },
        OpCase {
            name: "residual_add_f32",
            tol: Tol::TIGHT,
            seed: 0x9999_AAAA_BBBB_CCCC,
            scalars: &[("n", 2048.0)],
            build: |rng| {
                let n = 2048usize;
                let a = Tensor::from_fn(vec![n], |_| rng.normal());
                let b = Tensor::from_fn(vec![n], |_| rng.normal());
                let expected = ops::residual_add_f32(a.data(), b.data());
                (vec![a, b], expected)
            },
        },
        OpCase {
            name: "attn_decode_f32",
            tol: Tol::DEFAULT,
            seed: 0x0F1E_2D3C_4B5A_6978,
            scalars: &[
                ("n_heads", 8.0),
                ("n_kv", 2.0),
                ("head_dim", 64.0),
                ("seq", 64.0),
                ("scale", 0.125),
            ],
            build: |rng| {
                let (n_heads, n_kv, head_dim, seq, scale) =
                    (8usize, 2usize, 64usize, 64usize, 0.125f32);
                let q = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let k = Tensor::from_fn(vec![n_kv, seq, head_dim], |_| rng.normal_scaled(0.3));
                let v = Tensor::from_fn(vec![n_kv, seq, head_dim], |_| rng.normal());
                let expected = ops::attn_decode_f32(
                    q.data(),
                    k.data(),
                    v.data(),
                    n_heads,
                    n_kv,
                    head_dim,
                    seq,
                    scale,
                );
                (vec![q, k, v], expected)
            },
        },
        OpCase {
            name: "ssm_conv1d_f32",
            tol: Tol::TIGHT,
            seed: 0x5C0F_F1D0_0000_0001,
            scalars: &[("channels", 56.0), ("d_conv", 4.0)],
            build: |rng| {
                let (channels, d_conv) = (56usize, 4usize);
                let hist = d_conv - 1;
                let x = Tensor::from_fn(vec![channels], |_| rng.normal());
                let w = Tensor::from_fn(vec![d_conv * channels], |_| rng.normal_scaled(0.5));
                let state = Tensor::from_fn(vec![hist * channels], |_| rng.normal_scaled(0.3));
                let mut st = state.data().to_vec();
                let y = ssm_conv1d_f32(&mut st, x.data(), w.data(), d_conv);
                // The reference returns both the output and the carried window.
                let mut expected = y;
                expected.extend_from_slice(&st);
                (vec![x, w, state], expected)
            },
        },
        OpCase {
            name: "gdn_ar_f32",
            tol: Tol::DEFAULT,
            seed: 0x5C0F_F1D0_0000_0002,
            scalars: &[("n_v", 3.0), ("n_k", 2.0), ("sd", 8.0)],
            build: |rng| {
                let (n_v, n_k, sd) = (3usize, 2usize, 8usize);
                let q = Tensor::from_fn(vec![n_k, sd], |_| rng.normal());
                let k = Tensor::from_fn(vec![n_k, sd], |_| rng.normal());
                let v = Tensor::from_fn(vec![n_v, sd], |_| rng.normal());
                // `g` is a log decay, so keep it negative.
                let g = Tensor::from_fn(vec![n_v], |_| rng.range(-2.5, -0.1));
                let beta = Tensor::from_fn(vec![n_v], |_| rng.uniform());
                let s_in = Tensor::from_fn(vec![n_v * sd * sd], |_| rng.normal_scaled(0.3));
                let mut s = s_in.data().to_vec();
                let out = gdn_ar_f32(
                    &mut s,
                    q.data(),
                    k.data(),
                    v.data(),
                    g.data(),
                    beta.data(),
                    n_v,
                    n_k,
                    sd,
                );
                let mut expected = out;
                expected.extend_from_slice(&s);
                (vec![q, k, v, g, beta, s_in], expected)
            },
        },
        OpCase {
            name: "ssm_l2norm_f32",
            tol: Tol::TIGHT,
            seed: 0x5C0F_F1D0_0000_0003,
            scalars: &[("heads", 5.0), ("dim", 16.0), ("eps", 1e-6)],
            build: |rng| {
                let (heads, dim) = (5usize, 16usize);
                let x = Tensor::from_fn(vec![heads, dim], |_| rng.normal());
                let expected = l2norm_f32(x.data(), heads, dim, 1e-6);
                (vec![x], expected)
            },
        },
        OpCase {
            name: "ssm_gate_norm_f32",
            tol: Tol::DEFAULT,
            seed: 0x5C0F_F1D0_0000_0004,
            scalars: &[("n_v", 3.0), ("head_v", 8.0), ("eps", 1e-6)],
            build: |rng| {
                let (n_v, head_v) = (3usize, 8usize);
                let o = Tensor::from_fn(vec![n_v, head_v], |_| rng.normal());
                let norm = Tensor::from_fn(vec![head_v], |_| 1.0 + rng.normal_scaled(0.1));
                let z = Tensor::from_fn(vec![n_v, head_v], |_| rng.normal_scaled(1.5));
                let mut expected = Vec::with_capacity(n_v * head_v);
                for h in 0..n_v {
                    let r = ops::rmsnorm_f32(
                        &o.data()[h * head_v..(h + 1) * head_v],
                        norm.data(),
                        1e-6,
                    );
                    expected.extend_from_slice(&ops::silu_mul_f32(
                        &z.data()[h * head_v..(h + 1) * head_v],
                        &r,
                    ));
                }
                (vec![o, norm, z], expected)
            },
        },
        OpCase {
            name: "ssm_layer_f32",
            tol: Tol::DEFAULT,
            seed: 0x5C0F_F1D0_0000_0005,
            scalars: SSM_LAYER_SCALARS,
            build: build_small_ssm_layer,
        },
        OpCase {
            name: "sample_top_p_f32",
            tol: Tol::EXACT,
            seed: 0xDDDD_EEEE_FFFF_0001,
            scalars: &[("n", 256.0), ("top_p", 0.9), ("u", 0.42)],
            build: |rng| {
                let n = 256usize;
                let logits = Tensor::from_fn(vec![n], |_| rng.normal_scaled(2.0));
                let probs = ops::softmax_f32(logits.data());
                let idx = ops::sample_top_p_f32(&probs, 0.9, 0.42);
                (vec![Tensor::new(vec![n], probs)], vec![idx as f32])
            },
        },
    ];
    out.extend(crate::attn_parity::cases());
    out
}

/// The GDN shapes used by the small parity case and by the real-layer case.
#[derive(Clone, Copy, Debug)]
pub struct SsmParityDims {
    pub n_embd: usize,
    pub conv_ch: usize,
    pub d_conv: usize,
    pub n_v: usize,
    pub n_k: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub eps: f32,
}

impl SsmParityDims {
    pub fn key_dim(&self) -> usize {
        self.n_k * self.head_k
    }
}

/// Small enough that the reference and the upload stay cheap.
pub const SMALL_SSM: SsmParityDims = SsmParityDims {
    n_embd: 64,
    conv_ch: 56,
    d_conv: 4,
    n_v: 3,
    n_k: 2,
    head_k: 8,
    head_v: 8,
    eps: 1e-6,
};

/// Qwen3.8-27B: 5120 wide, 10240 conv channels, 48 value heads of 128.
pub const REAL_SSM: SsmParityDims = SsmParityDims {
    n_embd: 5120,
    conv_ch: 10240,
    d_conv: 4,
    n_v: 48,
    n_k: 16,
    head_k: 128,
    head_v: 128,
    eps: 1e-6,
};

const SSM_LAYER_SCALARS: &[(&str, f32)] = &[
    ("n_embd", SMALL_SSM.n_embd as f32),
    ("conv_ch", SMALL_SSM.conv_ch as f32),
    ("d_conv", SMALL_SSM.d_conv as f32),
    ("n_v", SMALL_SSM.n_v as f32),
    ("n_k", SMALL_SSM.n_k as f32),
    ("head_k", SMALL_SSM.head_k as f32),
    ("head_v", SMALL_SSM.head_v as f32),
    ("eps", SMALL_SSM.eps),
];

const REAL_LAYER_SCALARS: &[(&str, f32)] = &[
    ("n_embd", REAL_SSM.n_embd as f32),
    ("conv_ch", REAL_SSM.conv_ch as f32),
    ("d_conv", REAL_SSM.d_conv as f32),
    ("n_v", REAL_SSM.n_v as f32),
    ("n_k", REAL_SSM.n_k as f32),
    ("head_k", REAL_SSM.head_k as f32),
    ("head_v", REAL_SSM.head_v as f32),
    ("eps", REAL_SSM.eps),
];

/// One full GDN layer at the production shape. 464 MB of f32 weights, so this
/// is built only by the explicit `hipml ssm` command, never by `cases()`.
pub fn ssm_real_layer_case() -> OpCase {
    OpCase {
        name: "ssm_layer_f32",
        tol: Tol::new(2e-3, 1e-2),
        seed: 0x5C0F_F1D0_0000_1005,
        scalars: REAL_LAYER_SCALARS,
        build: build_real_ssm_layer,
    }
}

fn build_small_ssm_layer(rng: &mut Rng) -> (Vec<Tensor>, Vec<f32>) {
    build_ssm_layer(rng, &SMALL_SSM)
}

fn build_real_ssm_layer(rng: &mut Rng) -> (Vec<Tensor>, Vec<f32>) {
    build_ssm_layer(rng, &REAL_SSM)
}

/// Inputs in the order the device launcher reads them:
/// `x, wqkv, wgate, conv_w, alpha, beta, dt_bias, a, norm, wout, conv_s, s`.
fn build_ssm_layer(rng: &mut Rng, d: &SsmParityDims) -> (Vec<Tensor>, Vec<f32>) {
    let key_dim = d.key_dim();
    let inner = d.n_v * d.head_v;
    assert_eq!(d.conv_ch, 2 * key_dim + inner, "conv channel count");

    let x = Tensor::from_fn(vec![d.n_embd], |_| rng.normal_scaled(0.5));
    let wqkv = Tensor::from_fn(vec![d.conv_ch, d.n_embd], |_| rng.normal_scaled(0.02));
    let wgate = Tensor::from_fn(vec![inner, d.n_embd], |_| rng.normal_scaled(0.02));
    let conv_w = Tensor::from_fn(vec![d.d_conv * d.conv_ch], |_| rng.normal_scaled(0.2));
    let alpha = Tensor::from_fn(vec![d.n_v, d.n_embd], |_| rng.normal_scaled(0.02));
    let beta = Tensor::from_fn(vec![d.n_v, d.n_embd], |_| rng.normal_scaled(0.02));
    let dt_bias = Tensor::from_fn(vec![d.n_v], |_| rng.normal_scaled(0.1));
    let a = Tensor::from_fn(vec![d.n_v], |_| -rng.range(0.5, 1.5));
    let norm = Tensor::from_fn(vec![d.head_v], |_| 1.0 + rng.normal_scaled(0.1));
    let wout = Tensor::from_fn(vec![d.n_embd, inner], |_| rng.normal_scaled(0.02));
    let conv_s = Tensor::from_fn(vec![(d.d_conv - 1) * d.conv_ch], |_| rng.normal_scaled(0.3));
    let s = Tensor::from_fn(vec![d.n_v * d.head_v * d.head_v], |_| rng.normal_scaled(0.1));

    let w = SsmW {
        wqkv: f32_mat(d.conv_ch, d.n_embd, wqkv.data()),
        wgate: f32_mat(inner, d.n_embd, wgate.data()),
        conv1d: conv_w.data().to_vec(),
        alpha: f32_mat(d.n_v, d.n_embd, alpha.data()),
        beta: f32_mat(d.n_v, d.n_embd, beta.data()),
        dt_bias: dt_bias.data().to_vec(),
        a: a.data().to_vec(),
        norm: norm.data().to_vec(),
        wout: f32_mat(d.n_embd, inner, wout.data()),
    };
    let mut st = SsmState {
        conv: conv_s.data().to_vec(),
        s: s.data().to_vec(),
    };
    let out = ssm_layer_f32(&w, &mut st, x.data(), &parity_config(d));

    let mut expected = out;
    expected.extend_from_slice(&st.conv);
    expected.extend_from_slice(&st.s);
    (
        vec![
            x, wqkv, wgate, conv_w, alpha, beta, dt_bias, a, norm, wout, conv_s, s,
        ],
        expected,
    )
}

/// An f32 `Mat` so the reference layer can be called without a GGUF file.
fn f32_mat(rows: usize, cols: usize, data: &[f32]) -> Mat {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Mat {
        rows,
        cols,
        kind: Kind::F32,
        data: Bytes::Owned(bytes),
    }
}

/// Only the SSM fields of `Config` are read by `ssm_layer_f32`.
fn parity_config(d: &SsmParityDims) -> Config {
    Config {
        arch: "parity".to_string(),
        n_layer: 0,
        n_embd: d.n_embd,
        n_ff: 0,
        n_vocab: 0,
        n_head: 0,
        n_head_kv: 0,
        head_dim: 0,
        eps: d.eps,
        rope_base: 0.0,
        n_rot: 0,
        rope_sections: [0; 4],
        is_recr: Vec::new(),
        ssm_d_conv: d.d_conv,
        ssm_d_state: d.head_k,
        ssm_n_group: d.n_k,
        ssm_dt_rank: d.n_v,
        ssm_d_inner: d.n_v * d.head_v,
        context_length: 0,
    }
}

/// Production-shape cases, for `hipml bench` only.
///
/// The parity shapes are deliberately small so the reference stays cheap; at
/// those sizes a launch is ~2 µs and the bandwidth figure is launch overhead.
/// These are the shapes that decide whether a kernel approaches the card's
/// roofline, so they are measured and never compared against a CPU reference.
pub fn bench_cases() -> Vec<OpCase> {
    let attention = OpCase {
        name: "attn_decode_f32",
        tol: Tol::DEFAULT,
        seed: 0x1234_5678_9ABC_DEF0,
        // Ripple-65B at a 12K context: 80 query heads over 8 KV heads, head_dim
        // 128. K and V are 100 MB of fp32, which is the cache a token streams.
        scalars: &[
            ("n_heads", 80.0),
            ("n_kv", 8.0),
            ("head_dim", 128.0),
            ("seq", 12288.0),
            ("scale", 0.08838834764831845),
        ],
        build: |rng| {
            let (n_heads, n_kv, head_dim, seq) = (80usize, 8usize, 128usize, 12288usize);
            let q = Tensor::from_fn(vec![n_heads, head_dim], |_| (rng.uniform() - 0.5) * 0.1);
            let k = Tensor::from_fn(vec![n_kv, seq, head_dim], |_| (rng.uniform() - 0.5) * 0.1);
            let v = Tensor::from_fn(vec![n_kv, seq, head_dim], |_| rng.uniform() - 0.5);
            (vec![q, k, v], Vec::new())
        },
    };

    vec![
        OpCase {
            name: "gemv_f32",
            tol: Tol::DEFAULT,
            seed: 0xBEEF_1234_5678_9ABC,
            // Ripple-65B's ffn_up: [40960, 5120] — 838 MB of fp32 weights, which is
            // the bandwidth-bound case a dense model actually runs.
            scalars: &[("m", 40960.0), ("k", 5120.0)],
            build: |rng| {
                let (m, k) = (40960usize, 5120usize);
                // Uniform, not normal: 210M Box-Muller draws would take seconds and
                // the values are irrelevant to a bandwidth measurement.
                let w = Tensor::from_fn(vec![m, k], |_| (rng.uniform() - 0.5) * 0.04);
                let x = Tensor::from_fn(vec![k], |_| rng.uniform() - 0.5);
                let expected = Vec::new();
                (vec![w, x], expected)
            },
        },
        gemm_bench_case(4),
        gemm_bench_case(1),
        attention,
    ]
}

fn gemm_bench_case(r: usize) -> OpCase {
    let (scalars, build): (&'static [(&'static str, f32)], _) = match r {
        4 => (&[("r", 4.0), ("m", 40960.0), ("k", 5120.0)], 4),
        1 => (&[("r", 1.0), ("m", 40960.0), ("k", 5120.0)], 1),
        _ => unreachable!(),
    };
    OpCase {
        name: "gemm_f32",
        tol: Tol::DEFAULT,
        seed: 0xBEEF_1234_5678_9ABC,
        scalars,
        build: match build {
            4 => |rng| gemm_bench_inputs(rng, 4),
            _ => |rng| gemm_bench_inputs(rng, 1),
        },
    }
}

fn gemm_bench_inputs(rng: &mut Rng, r: usize) -> (Vec<Tensor>, Vec<f32>) {
    let (m, k) = (40960usize, 5120usize);
    let w = Tensor::from_fn(vec![m, k], |_| (rng.uniform() - 0.5) * 0.04);
    let x = Tensor::from_fn(vec![r, k], |_| rng.uniform() - 0.5);
    (vec![w, x], Vec::new())
}
