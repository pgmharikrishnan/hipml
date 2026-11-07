//! Greedy generation: prompt tokens in, argmax tokens out.

use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub struct Generated {
    pub prompt_ids: Vec<u32>,
    pub ids: Vec<u32>,
    pub text: String,
}

/// Feed the prompt one token at a time, then emit `n` greedy tokens.
///
/// `n_ctx` sizes the KV cache; pass prompt length + `n` + slack.
pub fn greedy(model: &Model, tok: &Tokenizer, prompt_ids: &[u32], n: usize, n_ctx: usize) -> Generated {
    greedy_with(model, tok, prompt_ids, n, n_ctx, |_, _| {})
}

/// As [`greedy`], calling `on_token(index, id)` as each token is produced so a
/// caller can stream progress.
pub fn greedy_with<F: FnMut(usize, u32)>(
    model: &Model,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    n: usize,
    n_ctx: usize,
    mut on_token: F,
) -> Generated {
    assert!(!prompt_ids.is_empty(), "need at least one prompt token");
    assert!(
        prompt_ids.len() + n <= n_ctx,
        "prompt {} + {n} tokens exceeds the {n_ctx}-token cache",
        prompt_ids.len()
    );

    let mut st = model.new_state(n_ctx);
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id, &mut st);
    }

    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let next = argmax_f32(&logits) as u32;
        on_token(i, next);
        ids.push(next);
        if i + 1 < n {
            logits = model.forward_token(next, &mut st);
        }
    }

    Generated {
        prompt_ids: prompt_ids.to_vec(),
        text: tok.decode(&ids),
        ids,
    }
}

/// Index of the largest value; ties go to the lowest index, as llama.cpp's
/// greedy sampler keeps the first maximum.
///
/// Kernel: `argmax_f32`.
pub fn argmax_f32(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
