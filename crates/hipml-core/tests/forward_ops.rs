//! Reference checks for the forward-pass ops that are not covered by the
//! kernel parity suite (which compares against the device, not against
//! llama.cpp's algebra).
//!
//! The numbers here are hand-computed from the formulas ported in
//! `delta.rs` / `ops.rs`, so they catch a transposition or an inverted gate
//! without needing a GPU or the 17 GB model.

use hipml_core::delta::{gdn_ar_f32, ssm_conv1d_f32};
use hipml_core::gen::argmax_f32;
use hipml_core::ops::{rope_multi_f32, rope_neox_f32};
use hipml_core::quant::{dequant_q5_k, Q5_K_BYTES, QK_K};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn rope_multi_with_full_rotation_is_plain_neox() {
    let head_dim = 8;
    let x: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.5 - 1.0).collect();
    let multi = rope_multi_f32(&x, 1, head_dim, head_dim, [0; 4], [3; 4], 10_000.0);
    let neox = rope_neox_f32(&x, 1, head_dim, 3, 10_000.0);
    for (a, b) in multi.iter().zip(&neox) {
        assert!(close(*a, *b, 1e-5), "{a} vs {b}");
    }
}

#[test]
fn rope_multi_rotates_only_the_first_n_rot_channels() {
    let head_dim = 8;
    let x: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
    let y = rope_multi_f32(&x, 1, head_dim, 4, [0; 4], [5; 4], 10_000.0);
    // Channels 4..8 are passthrough.
    assert_eq!(&y[4..], &x[4..]);
    // Pairs 0 and 1 rotate; the pair at i+2 must not equal the input.
    assert!((y[0] - x[0]).abs() > 1e-3);
    assert!((y[2] - x[2]).abs() > 1e-3);
    // Pair k has frequency base^(-2k/n_rot): k = 0 is 1, k = 1 is base^-0.5.
    let expect0 = x[0] * 5.0f32.cos() - x[2] * 5.0f32.sin();
    assert!(close(y[0], expect0, 1e-5), "{} vs {expect0}", y[0]);
    let ang = 5.0 * 10_000f64.powf(-2.0 / 4.0);
    let expect1 = x[1] * ang.cos() as f32 - x[3] * ang.sin() as f32;
    assert!(close(y[1], expect1, 1e-5), "{} vs {expect1}", y[1]);
}

#[test]
fn gdn_single_head_hand_computed() {
    // state rows: S[:,0] = [1,2], S[:,1] = [3,4]; q=[1,0], k=[0,1], v=[5,6].
    let mut s = vec![1.0, 2.0, 3.0, 4.0];
    let q = [1.0, 0.0];
    let k = [0.0, 1.0];
    let v = [5.0, 6.0];
    let g = [(0.5f32).ln()]; // decay = 0.5
    let beta = [1.0];
    let out = gdn_ar_f32(&mut s, &q, &k, &v, &g, &beta, 1, 1, 2);

    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    // j=0: dot = 2*0.5 = 1, delta = 4, row = [0.5, 5], attn = 0.5
    // j=1: dot = 4*0.5 = 2, delta = 4, row = [1.5, 6], attn = 1.5
    assert!(close(out[0], 0.5 * inv_sqrt2, 1e-6), "{}", out[0]);
    assert!(close(out[1], 1.5 * inv_sqrt2, 1e-6), "{}", out[1]);
    assert!(close(s[0], 0.5, 1e-6));
    assert!(close(s[1], 5.0, 1e-6));
    assert!(close(s[2], 1.5, 1e-6));
    assert!(close(s[3], 6.0, 1e-6));
}

#[test]
fn gdn_broadcasts_key_heads_by_modulo() {
    // Two value heads, one key head: both read k head 0.
    let mut s = vec![0.0; 2 * 2 * 2];
    let q = [1.0, 0.0];
    let k = [1.0, 0.0];
    let v = [1.0, 2.0, 3.0, 4.0];
    let g = [0.0, 0.0];
    let beta = [1.0, 1.0];
    let out = gdn_ar_f32(&mut s, &q, &k, &v, &g, &beta, 2, 1, 2);
    // Zero state, decay 1: S[:,j] = k * v[h][j], and q = [1,0] selects k[0]=1,
    // so o[h][j] = v[h][j] / sqrt(2) for both value heads.
    let inv = 1.0 / 2.0f32.sqrt();
    for (got, want) in out.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!(close(*got, want * inv, 1e-6), "{got} vs {}", want * inv);
    }
}

#[test]
fn conv1d_matches_a_hand_computed_window() {
    let mut state = vec![10.0, 20.0, 30.0, 40.0]; // 2 history rows, 2 channels
    let x = [50.0, 60.0];
    let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // per channel, k fastest
    let y = ssm_conv1d_f32(&mut state, &x, &w, 3);
    assert_eq!(y, vec![220.0, 640.0]);
    assert_eq!(state, vec![30.0, 40.0, 50.0, 60.0]);
}

#[test]
fn q5_k_dequant_follows_the_scale_and_high_bit_layout() {
    let mut b = vec![0u8; Q5_K_BYTES];
    b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    b[2..4].copy_from_slice(&0u16.to_le_bytes()); // dmin = 0
    // scales: sc = 1 for all eight sub-blocks, min = 0.
    b[4..16].copy_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
    // qh bit 2 -> fifth bit for the first 32-value group of sub-block 2.
    b[16] = 0x04;
    // qs: first group low nibble 2 / high nibble 1.
    b[48] = 0x12;
    let mut out = vec![0.0f32; QK_K];
    dequant_q5_k(&b, &mut out);
    assert_eq!(out[0], 2.0);
    assert_eq!(out[32], 1.0);
    // 64..96 is sub-block 2, whose high bit is u1 = 4 for l = 0.
    assert_eq!(out[64], 16.0);
    assert_eq!(out[96], 0.0);
}

#[test]
fn argmax_keeps_the_first_maximum() {
    assert_eq!(argmax_f32(&[0.1, 2.0, 2.0, -1.0]), 1);
    assert_eq!(argmax_f32(&[-5.0]), 0);
}
