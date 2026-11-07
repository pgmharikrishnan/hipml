//! Quantized decode-attention parity cases.
//!
//! Separated from `parity.rs` only by size: the cache-shape matrix is twelve
//! cases of data, and the table stays readable without it.
//!
//! The cache is encoded on the host and both the reference and the kernel read
//! those same bytes, so the comparison measures accumulation order only; the
//! quantization error is common to both sides and cancels. Tolerance is
//! [`Tol::DEFAULT`] (abs 1e-4, rel 1e-3).

use crate::kv::{self, KvDtype};
use crate::ops;
use crate::parity::{OpCase, Tol};
use crate::tensor::Tensor;

/// One quantized decode-attention case.
///
/// The cache is encoded on the host and both the reference and the kernel read
/// those same bytes, so the comparison measures accumulation order only; the
/// quantization error is common to both sides and cancels. Tolerance is
/// [`Tol::DEFAULT`] (abs 1e-4, rel 1e-3).
macro_rules! kv_case {
    ($name:literal, $dtype:expr, $dtypeid:literal, $heads:literal, $kv:literal,
     $dim:literal, $seq:literal, $splits:literal, $scale:literal, $seed:literal) => {
        OpCase {
            name: $name,
            tol: Tol::DEFAULT,
            seed: $seed,
            scalars: &[
                ("n_heads", $heads as f32),
                ("n_kv", $kv as f32),
                ("head_dim", $dim as f32),
                ("seq", $seq as f32),
                ("scale", $scale),
                ("splits", $splits as f32),
                ("dtype", $dtypeid),
            ],
            build: |rng| {
                let q = Tensor::from_fn(vec![$heads, $dim], |_| rng.normal());
                let kf = Tensor::from_fn(vec![$kv, $seq, $dim], |_| rng.normal_scaled(0.3));
                let vf = Tensor::from_fn(vec![$kv, $seq, $dim], |_| rng.normal());
                let kb = kv::quantize($dtype, kf.data());
                let vb = kv::quantize($dtype, vf.data());
                let expected = ops::attn_decode_kv(
                    q.data(),
                    &kb,
                    &vb,
                    $heads,
                    $kv,
                    $dim,
                    $seq,
                    $scale,
                    $dtype,
                );
                (
                    vec![q, Tensor::from_bytes_le(&kb), Tensor::from_bytes_le(&vb)],
                    expected,
                )
            },
        }
    };
}

/// The cache-shape matrix: length {1, 37, 1000, 4096} x {q8, q4} at head_dim
/// 256, plus head_dim 64. The 1000/4096 rows also exercise the split reduction
/// (7 and 16 splits). GQA is 6:1 in the 256 rows and 4:1 in the 64 rows.
pub fn kv_cases() -> Vec<OpCase> {
    vec![
        kv_case!(
            "attn_decode_kv_q8_hd256_s1",
            KvDtype::Q8,
            0.0,
            6,
            1,
            256,
            1,
            1,
            0.0625,
            0x9E00_0001
        ),
        kv_case!(
            "attn_decode_kv_q8_hd256_s37",
            KvDtype::Q8,
            0.0,
            6,
            1,
            256,
            37,
            1,
            0.0625,
            0x9E00_0002
        ),
        kv_case!(
            "attn_decode_kv_q8_hd256_s1000",
            KvDtype::Q8,
            0.0,
            6,
            1,
            256,
            1000,
            7,
            0.0625,
            0x9E00_0003
        ),
        kv_case!(
            "attn_decode_kv_q8_hd256_s4096",
            KvDtype::Q8,
            0.0,
            6,
            1,
            256,
            4096,
            16,
            0.0625,
            0x9E00_0004
        ),
        kv_case!(
            "attn_decode_kv_q4_hd256_s1",
            KvDtype::Q4,
            1.0,
            6,
            1,
            256,
            1,
            1,
            0.0625,
            0x9E00_0005
        ),
        kv_case!(
            "attn_decode_kv_q4_hd256_s37",
            KvDtype::Q4,
            1.0,
            4,
            2,
            256,
            37,
            1,
            0.0625,
            0x9E00_0006
        ),
        kv_case!(
            "attn_decode_kv_q4_hd256_s1000",
            KvDtype::Q4,
            1.0,
            6,
            1,
            256,
            1000,
            7,
            0.0625,
            0x9E00_0007
        ),
        kv_case!(
            "attn_decode_kv_q4_hd256_s4096",
            KvDtype::Q4,
            1.0,
            6,
            1,
            256,
            4096,
            16,
            0.0625,
            0x9E00_0008
        ),
        kv_case!(
            "attn_decode_kv_q8_hd64_s37",
            KvDtype::Q8,
            0.0,
            8,
            2,
            64,
            37,
            1,
            0.125,
            0x9E00_0009
        ),
        kv_case!(
            "attn_decode_kv_q8_hd64_s1000",
            KvDtype::Q8,
            0.0,
            8,
            2,
            64,
            1000,
            7,
            0.125,
            0x9E00_000A
        ),
        kv_case!(
            "attn_decode_kv_q4_hd64_s37",
            KvDtype::Q4,
            1.0,
            8,
            2,
            64,
            37,
            1,
            0.125,
            0x9E00_000B
        ),
        kv_case!(
            "attn_decode_kv_q4_hd64_s1000",
            KvDtype::Q4,
            1.0,
            8,
            2,
            64,
            1000,
            7,
            0.125,
            0x9E00_000C
        ),
    ]
}

/// 100K-depth cases for `hipml parity --deep`. One head keeps the host reference
/// under a second; the kernel path is the same one the bench runs.
pub fn deep_cases() -> Vec<OpCase> {
    vec![
        kv_case!(
            "attn_decode_kv_q8_hd256_s100000",
            KvDtype::Q8,
            0.0,
            1,
            1,
            256,
            100000,
            64,
            0.0625,
            0x9E00_1001
        ),
        kv_case!(
            "attn_decode_kv_q4_hd256_s100000",
            KvDtype::Q4,
            1.0,
            1,
            1,
            256,
            100000,
            64,
            0.0625,
            0x9E00_1002
        ),
    ]
}
