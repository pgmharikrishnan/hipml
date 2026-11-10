// SPDX-License-Identifier: Apache-2.0
//! Greedy generation: prompt tokens in, argmax tokens out.

use crate::model::Model;
use crate::tokenizer::Tokenizer;

pub struct Generated {
    pub prompt_ids: Vec<u32>,
    pub ids: Vec<u32>,
    pub text: String,
    /// True when decoding stopped because a stop id was sampled.
    pub stopped: bool,
}

/// Feed the prompt one token at a time, then emit `n` greedy tokens.
///
/// `n_ctx` sizes the KV cache; pass prompt length + `n` + slack.
pub fn greedy(model: &Model, tok: &Tokenizer, prompt_ids: &[u32], n: usize, n_ctx: usize) -> Generated {
    greedy_with(model, tok, prompt_ids, n, n_ctx, |_, _, _| {})
}

/// As [`greedy`], calling `on_token(index, id)` as each token is produced so a
/// caller can stream progress.
pub fn greedy_with<F: FnMut(usize, u32, &[f32])>(
    model: &Model,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    n: usize,
    n_ctx: usize,
    on_token: F,
) -> Generated {
    greedy_with_stop(model, tok, prompt_ids, n, n_ctx, &[], on_token)
}

/// Greedy decode that stops as soon as a sampled id is in `stop_ids`
/// (the model's EOS, 248046, for a chat completion). The stop token is kept in
/// `ids` so the caller can report `stopped_eos`; it is not fed back through the
/// model. Without this a chat model closes its turn and then invents a user
/// turn, and every acceptance number is measured past the answer.
pub fn greedy_with_stop<F: FnMut(usize, u32, &[f32])>(
    model: &Model,
    tok: &Tokenizer,
    prompt_ids: &[u32],
    n: usize,
    n_ctx: usize,
    stop_ids: &[u32],
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
    let mut stopped = false;
    for i in 0..n {
        let next = argmax_f32(&logits) as u32;
        on_token(i, next, &logits);
        ids.push(next);
        if stop_ids.contains(&next) {
            stopped = true;
            break;
        }
        if i + 1 < n {
            logits = model.forward_token(next, &mut st);
        }
    }

    Generated {
        prompt_ids: prompt_ids.to_vec(),
        text: tok.decode_text(&ids),
        ids,
        stopped,
    }
}

/// `log softmax(logits)[id]`, i.e. the log-probability llama-server reports
/// in `completion_probabilities`.
///
/// Kernel: `logprob_f32`; max-shifted so the exponent cannot overflow.
pub fn logprob_of(logits: &[f32], id: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|v| ((v - max) as f64).exp()).sum();
    let log_z = max as f64 + sum.ln();
    (logits[id] as f64 - log_z) as f32
}

/// The `k` largest logits as `(index, log-probability)`, descending.
///
/// Kernel: `top_logprobs_f32`.
pub fn top_logprobs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|v| ((v - max) as f64).exp()).sum();
    let log_z = max as f64 + sum.ln();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
        .into_iter()
        .take(k)
        .map(|i| (i, (logits[i] as f64 - log_z) as f32))
        .collect()
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
