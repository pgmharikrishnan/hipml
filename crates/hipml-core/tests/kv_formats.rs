// SPDX-License-Identifier: Apache-2.0
//! The quantized KV formats are a contract with the kernels, so they get exact
//! tests: known vectors, bit layout, and the error bound each encode promises.
//!
//! Nothing here needs a GPU. The device kernel is compared against
//! `ops::attn_decode_kv`, which reads these same bytes.

use hipml_core::kv::{self, KvDtype, BLOCK};
use hipml_core::ops;
use hipml_core::rng::Rng;

const Q8: KvDtype = KvDtype::Q8;
const Q4: KvDtype = KvDtype::Q4;

#[test]
fn block_byte_counts_are_the_documented_layout() {
    assert_eq!(Q8.block_bytes(), 34); // f16 d + 32 i8
    assert_eq!(Q4.block_bytes(), 18); // f16 d + 16 packed nibbles
    // Qwen3.8-27B: 2 (K,V) * 4 kv heads * 16 layers * 256 values/token.
    assert_eq!(Q8.token_bytes(4, 256, 16), 34_816);
    assert_eq!(Q4.token_bytes(4, 256, 16), 18_432);
}

#[test]
fn f16_round_trips_exactly_and_rounds_to_nearest() {
    // Every f16-representable value must come back bit-exact.
    for h in [0x0000u16, 0x3800, 0x3c00, 0xbc00, 0x7bff, 0x0001, 0x03ff, 0x7c00] {
        let f = kv::f16_to_f32(h);
        if h & 0x7fff == 0x7c00 {
            continue; // inf has no finite round trip worth asserting
        }
        assert_eq!(kv::f32_to_f16(f), h, "f16 {h:#06x} did not round trip");
    }
    // 0.1 is not representable; the nearest f16 is 0.099975586.
    assert_eq!(kv::f16_to_f32(kv::f32_to_f16(0.1)), 0.099_975_586);
    assert_eq!(kv::f32_to_f16(f32::INFINITY), 0x7c00);
    assert_eq!(kv::f32_to_f16(-0.0), 0x8000);
}

#[test]
fn q8_known_vector_is_exact_and_packs_as_documented() {
    // amax 63.5 -> d = 0.5, which f16 holds exactly, so this block must
    // round-trip bit-exact.
    let mut xs = [0.0f32; BLOCK];
    xs[0] = 63.5;
    xs[1] = -63.5;
    for (j, x) in xs.iter_mut().enumerate().skip(2) {
        *x = 0.5 * (j as f32 - 8.0);
    }
    let bytes = kv::quantize(Q8, &xs);
    assert_eq!(bytes.len(), 34);
    assert_eq!(&bytes[0..2], &[0x00, 0x38], "scale must be f16 0.5");
    assert_eq!(bytes[2], 127u8 as i8 as u8);
    assert_eq!(bytes[3], (-127i8) as u8);
    let back = kv::dequantize(Q8, &bytes);
    for j in 0..BLOCK {
        assert_eq!(back[j], xs[j], "value {j}");
    }
}

#[test]
fn q8_round_trip_error_is_half_a_step() {
    let mut rng = Rng::new(0x08);
    for _ in 0..64 {
        let xs: Vec<f32> = (0..BLOCK).map(|_| rng.normal_scaled(3.0)).collect();
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let step = amax / 127.0;
        let err = kv::max_abs_error(Q8, &xs);
        // Two terms: half a quantization step, plus the f16 scale's own
        // relative rounding (2^-11) applied to a value of up to 127 steps,
        // i.e. 0.0625 step. 0.5625 is the guaranteed bound.
        assert!(
            err <= step * 0.5625 + 1e-6,
            "err {err} > 0.5625*step {}",
            step * 0.5625
        );
        // And the sign and rough magnitude survive.
        let back = kv::dequantize(Q8, &kv::quantize(Q8, &xs));
        for (a, b) in xs.iter().zip(&back) {
            assert!(*b == 0.0 || a.signum() == b.signum());
        }
    }
    // All zeros: d is 0, payload is 0, decode is 0.
    let zeros = kv::quantize(Q8, &[0.0; BLOCK]);
    assert_eq!(&zeros[0..2], &[0x00, 0x00]);
    assert!(kv::dequantize(Q8, &zeros).iter().all(|v| *v == 0.0));
}

#[test]
fn q4_known_vector_packs_low_nibble_j_high_nibble_j16() {
    // d = 0.25 exactly, values 0.25 * (j%16 - 8) are exact in q4.
    let mut xs = [0.0f32; BLOCK];
    for (j, x) in xs.iter_mut().enumerate() {
        *x = 0.25 * ((j % 16) as f32 - 8.0);
    }
    let bytes = kv::quantize(Q4, &xs);
    assert_eq!(bytes.len(), 18);
    assert_eq!(&bytes[0..2], &[0x00, 0x34], "scale must be f16 0.25");
    for j in 0..16 {
        assert_eq!(bytes[2 + j], ((j as u8) << 4) | j as u8, "packed byte {j}");
    }
    let back = kv::dequantize(Q4, &bytes);
    for j in 0..BLOCK {
        assert_eq!(back[j], xs[j], "value {j}");
    }
}

#[test]
fn q4_extreme_maps_exactly_and_error_is_bounded() {
    // The ggml q4_0 scale takes the sign of the largest element, so that
    // element lands exactly on q = 0 or q = 15.
    for vmax in [2.0f32, -2.0, 0.125, -0.125] {
        let mut xs = [0.1f32; BLOCK];
        xs[5] = vmax;
        let bytes = kv::quantize(Q4, &xs);
        let back = kv::dequantize(Q4, &bytes);
        assert_eq!(back[5], vmax, "extreme {vmax} must be exact");
    }

    let mut rng = Rng::new(0x4_0);
    for _ in 0..64 {
        let xs: Vec<f32> = (0..BLOCK).map(|_| rng.normal_scaled(2.0)).collect();
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let back = kv::dequantize(Q4, &kv::quantize(Q4, &xs));
        let err = xs
            .iter()
            .zip(&back)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= amax / 8.0 + 1e-6, "err {err} > amax/8 {}", amax / 8.0);
    }
}

#[test]
fn row_layout_matches_block_order() {
    // The cache is [head][position][chunk]; value (h, pos, d) must come back
    // from block ((h*seq + pos)*nchunk + d/32), element d%32.
    let dtype = Q4;
    let (n_kv, seq, dim) = (2usize, 7usize, 64usize);
    let mut rng = Rng::new(77);
    let values: Vec<f32> = (0..n_kv * seq * dim).map(|_| rng.normal_scaled(0.7)).collect();
    let bytes = kv::quantize(dtype, &values);
    let nb = dim / BLOCK;
    let bb = dtype.block_bytes();
    let back = kv::dequantize(dtype, &bytes);
    for h in 0..n_kv {
        for pos in 0..seq {
            for d in 0..dim {
                let at = ((h * seq + pos) * nb + d / BLOCK) * bb;
                let want = kv::block_f32(dtype, &bytes[at..at + bb])[d % BLOCK];
                assert_eq!(back[(h * seq + pos) * dim + d], want);
            }
        }
    }
}

#[test]
fn quantized_attention_equals_attention_over_decoded_values() {
    // Same bytes on both sides, different summation order: this is the check
    // that the reference's online softmax and the dense two-pass agree.
    for dtype in [Q8, Q4] {
        let (heads, n_kv, dim, seq) = (6usize, 1usize, 64usize, 257usize);
        let mut rng = Rng::new(0xABCD + dtype as u64);
        let q: Vec<f32> = (0..heads * dim).map(|_| rng.normal()).collect();
        let k: Vec<f32> = (0..n_kv * seq * dim).map(|_| rng.normal_scaled(0.3)).collect();
        let v: Vec<f32> = (0..n_kv * seq * dim).map(|_| rng.normal()).collect();
        let kb = kv::quantize(dtype, &k);
        let vb = kv::quantize(dtype, &v);
        let quantized = ops::attn_decode_kv(&q, &kb, &vb, heads, n_kv, dim, seq, 0.125, dtype);
        let dense = ops::attn_decode_f32(
            &q,
            &kv::dequantize(dtype, &kb),
            &kv::dequantize(dtype, &vb),
            heads,
            n_kv,
            dim,
            seq,
            0.125,
        );
        let err = quantized
            .iter()
            .zip(&dense)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-5, "{dtype:?}: online vs two-pass max_abs {err}");
    }
}

#[test]
fn quantized_attention_error_against_the_f32_cache_is_small() {
    // Accuracy of the format, not of the kernel: attention over quantized K/V
    // versus attention over the original f32 K/V.
    let (heads, n_kv, dim, seq) = (6usize, 1usize, 64usize, 512usize);
    let mut rng = Rng::new(0xFEED);
    let q: Vec<f32> = (0..heads * dim).map(|_| rng.normal()).collect();
    let k: Vec<f32> = (0..n_kv * seq * dim).map(|_| rng.normal_scaled(0.3)).collect();
    let v: Vec<f32> = (0..n_kv * seq * dim).map(|_| rng.normal()).collect();
    let exact = ops::attn_decode_f32(&q, &k, &v, heads, n_kv, dim, seq, 0.125);
    for (dtype, bound) in [(Q8, 2e-3f32), (Q4, 3e-2)] {
        let got = ops::attn_decode_kv(
            &q,
            &kv::quantize(dtype, &k),
            &kv::quantize(dtype, &v),
            heads,
            n_kv,
            dim,
            seq,
            0.125,
            dtype,
        );
        let err = exact
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= bound, "{dtype:?}: max_abs {err} > {bound}");
    }
}

#[test]
fn synthetic_bench_cache_is_well_formed() {
    let bytes = kv::synthetic_blocks(Q8, 1000, 7);
    assert_eq!(bytes.len(), 1000 * 34);
    let back = kv::dequantize(Q8, &bytes);
    assert!(back.iter().all(|v| v.is_finite()), "bench bytes must be finite");
    assert!(back.iter().any(|v| v.abs() > 0.0), "bench bytes must be non-zero");
    for b in 0..1000 {
        let d = kv::block_scale(Q8, &bytes, b);
        assert!((0.01..0.05).contains(&d), "scale {d} out of range");
    }
}
