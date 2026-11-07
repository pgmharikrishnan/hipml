//! The reference kernels have to be right before anything is compared to them.
//!
//! These tests run anywhere (no GPU): they check the maths by identity rather
//! than by golden numbers, so they survive refactoring.

use hipml_core::ops;
use hipml_core::parity::{cases, compare, Tol};
use hipml_core::rng::Rng;
use hipml_core::tensor::Tensor;

#[test]
fn gemv_matches_hand_computed_values() {
    // 2x3 row-major: [[1,2,3],[4,5,6]] times [1,1,1] is [6,15].
    let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    assert_eq!(ops::gemv_f32(&w, &x, 2, 3), vec![6.0, 15.0]);

    // And a one-hot picks out a column.
    let y = ops::gemv_f32(&w, &[0.0, 1.0, 0.0], 2, 3);
    assert_eq!(y, vec![2.0, 5.0]);
}

#[test]
fn rmsnorm_output_has_unit_rms_before_the_gain() {
    let mut rng = Rng::new(7);
    let n = 513; // not a power of two on purpose
    let x: Vec<f32> = (0..n).map(|_| rng.normal_scaled(4.0)).collect();
    let ones = vec![1.0f32; n];
    let y = ops::rmsnorm_f32(&x, &ones, 1e-6);
    let mean_sq: f32 = y.iter().map(|v| v * v).sum::<f32>() / n as f32;
    assert!(
        (mean_sq.sqrt() - 1.0).abs() < 1e-3,
        "rms was {}",
        mean_sq.sqrt()
    );

    // The gain scales elementwise.
    let gain: Vec<f32> = (0..n).map(|i| (i % 5) as f32 + 0.5).collect();
    let scaled = ops::rmsnorm_f32(&x, &gain, 1e-6);
    for i in 0..n {
        assert!((scaled[i] - y[i] * gain[i]).abs() < 1e-5);
    }
}

#[test]
fn rope_preserves_norm_and_is_identity_at_position_zero() {
    let mut rng = Rng::new(11);
    let (heads, dim) = (4usize, 64usize);
    let x: Vec<f32> = (0..heads * dim).map(|_| rng.normal()).collect();

    let at_zero = ops::rope_neox_f32(&x, heads, dim, 0, 10000.0);
    assert_eq!(at_zero, x, "position 0 must not rotate");

    for pos in [1usize, 37, 4096] {
        let y = ops::rope_neox_f32(&x, heads, dim, pos, 10000.0);
        for h in 0..heads {
            let base = h * dim;
            let n0: f32 = x[base..base + dim].iter().map(|v| v * v).sum();
            let n1: f32 = y[base..base + dim].iter().map(|v| v * v).sum();
            assert!(
                (n0 - n1).abs() < 1e-3,
                "head {h} at pos {pos}: {n0} vs {n1}"
            );
        }
    }
}

#[test]
fn silu_mul_matches_the_formula() {
    let gate = vec![0.0, 1.0, -1.0, 8.0, -8.0];
    let up = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let out = ops::silu_mul_f32(&gate, &up);
    for i in 0..gate.len() {
        let want = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
        assert!((out[i] - want).abs() < 1e-6);
    }
    assert!((out[0] - 0.0).abs() < 1e-7, "silu(0) is 0");
}

#[test]
fn softmax_is_normalised_monotone_and_stable() {
    let mut rng = Rng::new(13);
    let x: Vec<f32> = (0..777).map(|_| rng.normal_scaled(2.0)).collect();
    let p = ops::softmax_f32(&x);
    let sum: f32 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum was {sum}");
    assert!(p.iter().all(|v| *v >= 0.0 && v.is_finite()));

    // Order preserved.
    for i in 0..x.len() {
        for j in 0..x.len() {
            if x[i] > x[j] {
                assert!(p[i] >= p[j]);
            }
        }
    }

    // Large logits must not overflow.
    let big = vec![10_000.0, 9_999.0, -10_000.0];
    let pb = ops::softmax_f32(&big);
    assert!(pb.iter().all(|v| v.is_finite()), "softmax overflowed");
    assert!((pb.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn residual_add_is_elementwise() {
    let a = vec![1.0, -2.0, 3.5];
    let b = vec![0.5, 2.0, -3.5];
    assert_eq!(ops::residual_add_f32(&a, &b), vec![1.5, 0.0, 0.0]);
}

#[test]
fn top_p_sampling_respects_the_nucleus_and_is_deterministic() {
    let probs = vec![0.5, 0.25, 0.15, 0.07, 0.03];
    // u=0 always lands on the highest-probability candidate.
    assert_eq!(ops::sample_top_p_f32(&probs, 0.9, 0.0), 0);
    // Determinism: same inputs, same answer.
    for u in [0.0, 0.25, 0.5, 0.75, 0.999] {
        assert_eq!(
            ops::sample_top_p_f32(&probs, 0.9, u),
            ops::sample_top_p_f32(&probs, 0.9, u)
        );
    }
    // A tiny nucleus can only return the argmax.
    for u in [0.0, 0.5, 0.99] {
        assert_eq!(ops::sample_top_p_f32(&probs, 0.1, u), 0);
    }
    // u just under 1 stays inside the nucleus, never off the end.
    let idx = ops::sample_top_p_f32(&probs, 0.6, 0.999_999);
    assert!(idx < probs.len());
}

#[test]
fn every_parity_case_builds_and_is_internally_consistent() {
    for case in cases() {
        let (inputs, expected) = case.reference();
        assert!(!inputs.is_empty(), "{} has no inputs", case.name);
        assert!(
            !expected.is_empty(),
            "{} produced no reference output",
            case.name
        );
        // Re-running with the same seed must reproduce exactly, or a failure on
        // the GPU host cannot be replayed.
        let (inputs2, expected2) = case.reference();
        assert_eq!(expected, expected2, "{} is not deterministic", case.name);
        for (a, b) in inputs.iter().zip(&inputs2) {
            assert_eq!(a, b, "{} inputs are not deterministic", case.name);
        }
        // And the reference must compare clean against itself.
        compare(&expected, &expected, case.tol).expect("self-comparison failed");
    }
}

#[test]
fn tensor_rejects_mismatched_shapes() {
    let t = Tensor::new(vec![2, 3], vec![0.0; 6]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.row(1).len(), 3);
    let caught = std::panic::catch_unwind(|| Tensor::new(vec![2, 3], vec![0.0; 5]));
    assert!(caught.is_err(), "shape/size mismatch must panic");
}

#[test]
fn compare_reports_distance_beyond_tolerance() {
    let tol = Tol::new(1e-3, 1e-3);
    compare(&[1.0, 2.0], &[1.0, 2.0001], tol).expect("inside tolerance");
    let err = compare(&[1.0, 2.0], &[1.0, 3.0], tol).expect_err("outside tolerance");
    assert!(err.max_abs > 0.9);
}

#[test]
fn attention_matches_a_hand_computed_case() {
    // One head, one KV head, head_dim 2, seq 2, no scale.
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![10.0, 0.0, 0.0, 100.0];
    let out = ops::attn_decode_f32(&q, &k, &v, 1, 1, 2, 2, 1.0);
    // scores 1.0 and 0.0 -> softmax [0.731, 0.269] -> weighted V.
    let e = std::f32::consts::E;
    let (p0, p1) = (e / (e + 1.0), 1.0 / (e + 1.0));
    assert!((out[0] - p0 * 10.0).abs() < 1e-5, "got {}", out[0]);
    assert!((out[1] - p1 * 100.0).abs() < 1e-5, "got {}", out[1]);
}

#[test]
fn attention_of_uniform_scores_is_the_mean_of_values() {
    let (heads, kv, dim, seq) = (4usize, 1usize, 8usize, 16usize);
    let q = vec![0.0f32; heads * dim];
    let k = vec![1.0f32; kv * seq * dim];
    let mut rng = Rng::new(29);
    let v: Vec<f32> = (0..kv * seq * dim).map(|_| rng.normal()).collect();
    let out = ops::attn_decode_f32(&q, &k, &v, heads, kv, dim, seq, 0.1);

    for h in 0..heads {
        for d in 0..dim {
            let mean: f32 = (0..seq).map(|s| v[s * dim + d]).sum::<f32>() / seq as f32;
            assert!(
                (out[h * dim + d] - mean).abs() < 1e-5,
                "head {h} dim {d}: {} vs {mean}",
                out[h * dim + d]
            );
        }
    }
}

#[test]
fn attention_routes_each_query_head_to_its_kv_head() {
    // Two KV heads with disjoint value patterns; GQA 2:1 must not mix them.
    let (heads, kv, dim, seq) = (4usize, 2usize, 4usize, 3usize);
    let q = vec![1.0f32; heads * dim];
    let k = vec![1.0f32; kv * seq * dim];
    let mut v = vec![0.0f32; kv * seq * dim];
    for s in 0..seq {
        for d in 0..dim {
            v[s * dim + d] = 1.0; // kv head 0 -> all ones
            v[seq * dim + s * dim + d] = 7.0; // kv head 1 -> all sevens
        }
    }
    let out = ops::attn_decode_f32(&q, &k, &v, heads, kv, dim, seq, 0.1);
    for h in 0..heads {
        let want = if h < 2 { 1.0 } else { 7.0 };
        for d in 0..dim {
            assert!((out[h * dim + d] - want).abs() < 1e-5, "head {h} dim {d}");
        }
    }
}

#[test]
fn attention_output_is_a_convex_combination() {
    // Every output element must sit inside the range of the values it attended.
    let (heads, kv, dim, seq) = (2usize, 1usize, 6usize, 32usize);
    let mut rng = Rng::new(31);
    let q: Vec<f32> = (0..heads * dim).map(|_| rng.normal()).collect();
    let k: Vec<f32> = (0..kv * seq * dim).map(|_| rng.normal()).collect();
    let v: Vec<f32> = (0..kv * seq * dim).map(|_| rng.normal()).collect();
    let out = ops::attn_decode_f32(&q, &k, &v, heads, kv, dim, seq, 0.125);
    for d in 0..dim {
        let lo = (0..seq)
            .map(|s| v[s * dim + d])
            .fold(f32::INFINITY, f32::min);
        let hi = (0..seq)
            .map(|s| v[s * dim + d])
            .fold(f32::NEG_INFINITY, f32::max);
        for h in 0..heads {
            let x = out[h * dim + d];
            assert!(
                x >= lo - 1e-5 && x <= hi + 1e-5,
                "head {h} dim {d}: {x} outside [{lo},{hi}]"
            );
        }
    }
}
