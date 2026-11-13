// SPDX-License-Identifier: Apache-2.0
//! `hipml ppl`: the perplexity of a token corpus, computed the way
//! `llama-perplexity` computes it.
//!
//! Until this command there was no quality number anywhere in hipml: `logits`
//! applies `lm_head` to a caller-supplied 5,120-float vector, so it has no text,
//! no ids and no tokenizer, and a benchmark that only measures tok/s cannot say
//! whether the tokens it produced are the ones the reference engine would have
//! produced. This command closes that: it prefills a window of ids through the
//! production chunked-prefill path, applies `lm_head` to each position, and
//! reports the mean negative log-likelihood of the true next token.
//!
//! The convention is deliberately `llama-perplexity`'s, because the point of
//! the number is to be comparable with the reference engine:
//!
//! * no BOS is prepended (Qwen3.8's vocab does not add one, and neither do we);
//! * `--ppl-stride 0`: windows are non-overlapping and `n_tokens / n_ctx` of
//!   them are scored, so the tail shorter than a window is not scored at all;
//! * only the second half of each window is scored (`first = n_ctx / 2`), which
//!   is the HuggingFace convention llama.cpp follows so that every scored
//!   position has real context behind it;
//! * the last position of a window has no next token inside the window, so a
//!   512-token window scores 255 positions, not 256;
//! * `PPL = exp(mean NLL)` with the NLL computed as
//!   `-(logit[t] - max - ln(sum(exp(logit - max))))`, f32 logits, f64 sum.
//!
//! It does not tokenize. `--ids-file` takes the ids the other commands read, so
//! the ids that reach the model are the ids the caller measured. That matters:
//! `llama-perplexity -f` pops the file's trailing newline before tokenizing
//! (`common/arg.cpp`, the `-f` handler), so its corpus is the file minus that
//! one token. We do not silently drop an id; the token count is printed so the
//! difference is visible instead of hidden.

use hipml_hip::Buffer;
use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::gpu_weights::gemv_entry;
use hipml_model::{Model, OUTPUT};

struct Args {
    gguf: String,
    ids_file: Option<String>,
    ctx: usize,
    chunk: usize,
    kv: KvMode,
    chunks: Option<usize>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        ids_file: None,
        ctx: 512,
        chunk: 0,
        kv: KvMode::F32,
        chunks: None,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--gguf" => {
                a.gguf = need(i)?.to_string();
                i += 2;
            }
            "--ids-file" => {
                a.ids_file = Some(need(i)?.to_string());
                i += 2;
            }
            "--ctx" => {
                a.ctx = need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?;
                i += 2;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--chunks" => {
                a.chunks = Some(need(i)?.parse().map_err(|e| format!("bad --chunks: {e}"))?);
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.ids_file.is_none() {
        return Err("--ids-file PATH is required".to_string());
    }
    if a.ctx < 4 {
        return Err("--ctx must be at least 4 (the second half is what gets scored)".to_string());
    }
    if a.chunk == 0 {
        a.chunk = a.ctx;
    }
    if a.chunk > a.ctx {
        return Err(format!(
            "--chunk {} exceeds --ctx {}; a window is one context",
            a.chunk, a.ctx
        ));
    }
    Ok(a)
}

/// `-(logit[target] - max - ln(sum(exp(logit - max))))`, the expression
/// `llama-perplexity`'s `log_softmax` evaluates: f32 logits, f32 exp, f64 sum.
fn nll_of(logits: &[f32], target: u32) -> f64 {
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f64;
    for &v in logits {
        sum += f32::exp(v - max) as f64;
    }
    let t = logits
        .get(target as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);
    (max as f64 - t as f64) + sum.ln()
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("ppl: {e}");
            return 2;
        }
    };
    let ids = match crate::device_cmd::read_ids_file(a.ids_file.as_deref().unwrap_or("")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ppl: {e}");
            return 2;
        }
    };

    // One full window is the minimum: a shorter corpus has no window to score.
    // `llama-perplexity` additionally refuses anything under two windows, but
    // that guard is about a text file that still has to be tokenized; here the
    // ids file *is* the corpus, and `--chunks 1` is a valid reference run we
    // must be able to compare against.
    if ids.len() < a.ctx {
        eprintln!(
            "ppl: need at least {} tokens for --ctx {}, the ids file has {}",
            a.ctx,
            a.ctx,
            ids.len()
        );
        return 2;
    }
    let n_chunk_max = ids.len() / a.ctx;
    let n_chunk = match a.chunks {
        Some(n) if n > 0 => n.min(n_chunk_max),
        _ => n_chunk_max,
    };
    let first = a.ctx / 2;

    let (model, _report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ppl: load: {e}");
            return 1;
        }
    };
    let mut engine = match GpuEngine::new(&model, a.ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ppl: graph: {e}");
            return 1;
        }
    };
    let n_embd = engine.n_embd();
    let n_vocab = engine.n_vocab();
    let out = match model.tensor(OUTPUT) {
        Some(t) => t,
        None => {
            eprintln!("ppl: model has no {OUTPUT}");
            return 1;
        }
    };
    let entry = match gemv_entry(out.ty()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ppl: {e}");
            return 1;
        }
    };
    let l = model.launcher();

    // One row of post-`output_norm` hidden state per position, and one logit
    // row: `lm_head` is applied a position at a time, exactly as the decode
    // path applies it, so the numbers come from the shipped GEMV.
    let capture = match Buffer::alloc(a.ctx * n_embd) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ppl: capture buffer: {e}");
            return 1;
        }
    };
    let logits = match Buffer::alloc(n_vocab) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("ppl: logits buffer: {e}");
            return 1;
        }
    };

    println!(
        "ppl: {} tokens, ctx {}, chunk {}, {} KV, windows {}, scoring {}..{} of each",
        ids.len(),
        a.ctx,
        a.chunk,
        a.kv.name(),
        n_chunk,
        first,
        a.ctx - 2
    );

    let (mut nll, mut nll2, mut count) = (0.0f64, 0.0f64, 0usize);
    for c in 0..n_chunk {
        let win = &ids[c * a.ctx..(c + 1) * a.ctx];
        // Every window starts cold, as `llama_memory_clear` makes llama's. The
        // KV rows do not need clearing (position t is written before it is
        // read) but the SSM layers' carried state does, and it lives in the
        // engine, not in `GpuState`.
        if let Err(e) = engine.reset_state() {
            eprintln!("ppl: chunk {c}: reset: {e}");
            return 1;
        }
        let mut st = engine.new_state();
        if let Err(e) = engine.prefill_chunked_capture(win, &mut st, a.chunk, &capture) {
            eprintln!("ppl: chunk {c}: prefill: {e}");
            return 1;
        }
        for t in first..a.ctx - 1 {
            let row = unsafe { (capture.as_ptr() as *const u8).add(t * n_embd * 4) };
            if let Err(e) = l.run_quant_resident_ptr(
                entry,
                out.buffer(),
                row as *mut std::ffi::c_void,
                logits.as_ptr(),
                n_vocab,
                n_embd,
            ) {
                eprintln!("ppl: chunk {c} position {t}: lm_head: {e}");
                return 1;
            }
            let host = match logits.to_host() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ppl: chunk {c} position {t}: logits: {e}");
                    return 1;
                }
            };
            let v = nll_of(&host, win[t + 1]);
            nll += v;
            nll2 += v * v;
            count += 1;
        }
        // llama prints a running estimate after each chunk; same shape, so the
        // two logs can be read side by side.
        println!("[{}]{:.4},", c + 1, (nll / count as f64).exp());
    }

    // Same standard error llama reports: the spread of the per-position NLL.
    let mean = nll / count as f64;
    let ppl = mean.exp();
    let var = nll2 / count as f64 - mean * mean;
    let se = if var > 0.0 && count > 1 {
        (var / (count - 1) as f64).sqrt()
    } else {
        0.0
    };
    println!("ppl: scored {count} positions over {n_chunk} windows");
    println!("Final estimate: PPL = {ppl:.4} +/- {:.5}", se * ppl);
    0
}
