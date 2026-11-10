// SPDX-License-Identifier: Apache-2.0
//! Parity cases for the GPU full-attention block, kept out of the main table so
//! the two files stay readable.

use crate::ops;
use crate::parity::{OpCase, Tol};
use crate::tensor::Tensor;

pub fn cases() -> Vec<OpCase> {
    vec![
        OpCase {
            name: "qg_split_f32",
            tol: Tol::TIGHT,
            seed: 0x2222_3333_4444_5555,
            scalars: &[("n_heads", 6.0), ("head_dim", 64.0)],
            build: |rng| {
                let (n_heads, head_dim) = (6usize, 64usize);
                let qg = Tensor::from_fn(vec![n_heads, 2 * head_dim], |_| rng.normal());
                let expected = ops::qg_split_f32(qg.data(), n_heads, head_dim);
                (vec![qg], expected)
            },
        },
        OpCase {
            name: "head_rmsnorm_f32",
            tol: Tol::DEFAULT,
            seed: 0x3333_4444_5555_6666,
            scalars: &[("n_heads", 6.0), ("head_dim", 64.0), ("eps", 1e-6)],
            build: |rng| {
                let (n_heads, head_dim) = (6usize, 64usize);
                let x = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let w = Tensor::from_fn(vec![head_dim], |_| 1.0 + rng.normal_scaled(0.1));
                let expected = ops::head_rmsnorm_f32(x.data(), w.data(), n_heads, head_dim, 1e-6);
                (vec![x, w], expected)
            },
        },
        OpCase {
            name: "mrope_f32",
            tol: Tol::TIGHT,
            seed: 0x4444_5555_6666_7777,
            scalars: &[
                ("n_heads", 4.0),
                ("head_dim", 64.0),
                ("n_rot", 64.0),
                ("s0", 11.0),
                ("s1", 11.0),
                ("s2", 10.0),
                ("s3", 0.0),
                ("pos", 37.0),
                ("theta_base", 1e7),
            ],
            build: |rng| {
                let (n_heads, head_dim, n_rot) = (4usize, 64usize, 64usize);
                let x = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let expected = ops::rope_multi_f32(
                    x.data(),
                    n_heads,
                    head_dim,
                    n_rot,
                    [11, 11, 10, 0],
                    [37; 4],
                    1e7,
                );
                (vec![x], expected)
            },
        },
        OpCase {
            name: "q_gate_f32",
            tol: Tol::TIGHT,
            seed: 0x5555_6666_7777_8888,
            scalars: &[("n", 384.0)],
            build: |rng| {
                let n = 384usize;
                let q = Tensor::from_fn(vec![n], |_| rng.normal_scaled(0.5));
                let gate = Tensor::from_fn(vec![n], |_| rng.normal_scaled(2.0));
                let expected = ops::q_gate_f32(q.data(), gate.data());
                (vec![q, gate], expected)
            },
        },
        OpCase {
            name: "kv_append_f32",
            tol: Tol::EXACT,
            seed: 0x6666_7777_8888_9999,
            scalars: &[("n_ctx", 8.0), ("width", 128.0), ("off", 3.0)],
            build: |rng| {
                let (n_ctx, width, off) = (8usize, 128usize, 3usize);
                let k_cache = Tensor::from_fn(vec![n_ctx, width], |_| rng.uniform());
                let v_cache = Tensor::from_fn(vec![n_ctx, width], |_| rng.uniform());
                let k = Tensor::from_fn(vec![width], |_| rng.normal());
                let v = Tensor::from_fn(vec![width], |_| rng.normal());
                let expected = ops::kv_append_f32(
                    k_cache.data(),
                    v_cache.data(),
                    k.data(),
                    v.data(),
                    off,
                    width,
                );
                (vec![k_cache, v_cache, k, v], expected)
            },
        },
        OpCase {
            name: "attn_decode_pm_f32",
            tol: Tol::DEFAULT,
            seed: 0x7777_8888_9999_AAAA,
            scalars: &[
                ("n_heads", 8.0),
                ("n_kv", 2.0),
                ("head_dim", 64.0),
                ("seq", 37.0),
                ("scale", 0.125),
            ],
            build: |rng| {
                let (n_heads, n_kv, head_dim, seq, scale) =
                    (8usize, 2usize, 64usize, 37usize, 0.125f32);
                let q = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let k = Tensor::from_fn(vec![seq, n_kv * head_dim], |_| rng.normal_scaled(0.3));
                let v = Tensor::from_fn(vec![seq, n_kv * head_dim], |_| rng.normal());
                let expected = crate::attention::attn_causal_f32(
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
            name: "attn_decode_split_f32",
            tol: Tol::DEFAULT,
            seed: 0x8888_9999_AAAA_BBBB,
            scalars: &[
                ("n_heads", 8.0),
                ("n_kv", 2.0),
                ("head_dim", 64.0),
                ("seq", 37.0),
                ("scale", 0.125),
                ("splits", 5.0),
            ],
            build: |rng| {
                let (n_heads, n_kv, head_dim, seq, scale) =
                    (8usize, 2usize, 64usize, 37usize, 0.125f32);
                let q = Tensor::from_fn(vec![n_heads, head_dim], |_| rng.normal());
                let k = Tensor::from_fn(vec![seq, n_kv * head_dim], |_| rng.normal_scaled(0.3));
                let v = Tensor::from_fn(vec![seq, n_kv * head_dim], |_| rng.normal());
                let expected = crate::attention::attn_causal_f32(
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
    ]
}
