// SPDX-License-Identifier: Apache-2.0
//! `hipml spec-bench`: MTP speculative decode on the resident GPU engine.
//!
//! Runs plain greedy decode (0 drafts) and speculative decode with 1..D drafts
//! over the same prompt, on fresh engines, and reports the token-exactness
//! check, acceptance per draft position, tokens per step, ms per step and
//! tok/s. The prompt is the `--ids-file` fixture the rest of the bench uses.
//!
//! ```text
//! hipml spec-bench --gguf Qwen3.8-27B.gguf --mtp mtp.gguf --ids-file prompt.ids \
//!     --tokens 128 --drafts 3 --kv f32
//! ```

use std::time::Instant;

use hipml_model::gpu::{GpuEngine, KvMode, MtpGpu, SpecRun};
use hipml_model::Model;

struct Args {
    gguf: String,
    mtp: String,
    ids_file: String,
    tokens: usize,
    drafts: usize,
    kv: KvMode,
    ctx: Option<usize>,
    chunk: usize,
    json: bool,
    stop_ids: Vec<u32>,
    stop_eos: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        mtp: String::new(),
        ids_file: String::new(),
        tokens: 128,
        drafts: 3,
        kv: KvMode::F32,
        ctx: None,
        chunk: 2048,
        json: false,
        stop_ids: Vec::new(),
        stop_eos: false,
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
            "--mtp" => {
                a.mtp = need(i)?.to_string();
                i += 2;
            }
            "--ids-file" => {
                a.ids_file = need(i)?.to_string();
                i += 2;
            }
            "--tokens" => {
                a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--drafts" => {
                a.drafts = need(i)?.parse().map_err(|e| format!("bad --drafts: {e}"))?;
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--ctx" => {
                a.ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                i += 2;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--json" => {
                a.json = true;
                i += 1;
            }
            "--stop-eos" => {
                a.stop_eos = true;
                i += 1;
            }
            "--stop-ids" => {
                let parsed: Result<Vec<u32>, _> = need(i)?
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(str::parse::<u32>)
                    .collect();
                a.stop_ids = parsed.map_err(|e| format!("bad --stop-ids: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() || a.mtp.is_empty() || a.ids_file.is_empty() {
        return Err("--gguf, --mtp and --ids-file are required".to_string());
    }
    Ok(a)
}

/// One run's numbers.
struct Row {
    drafts: usize,
    ids: Vec<u32>,
    steps: usize,
    tokens: usize,
    accept: Vec<f64>,
    tokens_per_step: f64,
    ms_per_step: f64,
    tok_s: f64,
    prefill_s: f64,
    draft_prefill_s: f64,
    /// Inside the decode loop: draft, verify, rollback (ms/step).
    draft_ms: f64,
    verify_ms: f64,
    rollback_ms: f64,
    stopped: bool,
}

pub fn bench(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("spec-bench: {e}");
            return 2;
        }
    };
    let prompt = match crate::device_cmd::read_ids_file(&a.ids_file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("spec-bench: {e}");
            return 1;
        }
    };
    let mut stop_ids = a.stop_ids.clone();
    if a.stop_eos {
        // The file is the authority; read its EOS rather than hard-coding.
        match hipml_core::gguf::Gguf::open(&a.gguf)
            .ok()
            .and_then(|g| g.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32()))
        {
            Some(eos) => stop_ids.push(eos),
            None => eprintln!("spec-bench: --stop-eos but the GGUF has no eos id"),
        }
    }
    let t0 = Instant::now();
    let (mut model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("spec-bench: resident load: {e}");
            return 1;
        }
    };
    let sidecar = match model.load_sidecar(&a.mtp) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("spec-bench: sidecar: {e}");
            return 1;
        }
    };
    let n_ctx = a.ctx.unwrap_or(prompt.len() + a.tokens + 16);
    let chunk = a.chunk.min(prompt.len()).max(1);

    // The drafter borrows the model, so build it before the engines.
    let mut mtp = match MtpGpu::new(&model, n_ctx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("spec-bench: drafter: {e}");
            return 1;
        }
    };
    let drafter_bytes = mtp.weight_bytes();
    let drafter_kv = mtp.kv_bytes();
    let mut rows: Vec<Row> = Vec::new();

    // 0 drafts: the plain decode reference.
    let base_ids;
    {
        let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("spec-bench: graph: {e}");
                return 1;
            }
        };
        let mut st = engine.new_state();
        let t = Instant::now();
        let first = match engine.prefill_chunked(&prompt, &mut st, chunk) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("spec-bench: prefill: {e}");
                return 1;
            }
        };
        let prefill_s = t.elapsed().as_secs_f64();
        let mut ids = Vec::with_capacity(a.tokens);
        ids.push(first);
        let mut next = first;
        let mut stopped = stop_ids.contains(&first);
        let t = Instant::now();
        while ids.len() < a.tokens && !stopped {
            next = match engine.forward_token_id(next, &mut st) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("spec-bench: decode: {e}");
                    return 1;
                }
            };
            ids.push(next);
            stopped = stop_ids.contains(&next);
        }
        let decode_s = t.elapsed().as_secs_f64();
        let steps = ids.len() - 1;
        base_ids = ids.clone();
        rows.push(Row {
            drafts: 0,
            ids,
            steps,
            tokens: steps,
            accept: Vec::new(),
            tokens_per_step: 1.0,
            ms_per_step: decode_s * 1e3 / steps.max(1) as f64,
            tok_s: steps as f64 / decode_s.max(1e-9),
            prefill_s,
            draft_prefill_s: 0.0,
            draft_ms: 0.0,
            verify_ms: decode_s * 1e3 / steps.max(1) as f64,
            rollback_ms: 0.0,
            stopped,
        });
        if !a.json {
            println!(
                "resident: {:.2} GiB in {} tensors + sidecar {:.2} GiB; \
                 drafter {:.2} MiB, drafter KV {:.1} MiB",
                report.tensor_bytes as f64 / (1u64 << 30) as f64,
                report.tensor_count,
                sidecar as f64 / (1u64 << 30) as f64,
                drafter_bytes as f64 / (1024.0 * 1024.0),
                drafter_kv as f64 / (1024.0 * 1024.0),
            );
        }
    }

    for d in 1..=a.drafts {
        let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("spec-bench: graph: {e}");
                return 1;
            }
        };
        let run: SpecRun = match engine.spec_generate(&mut mtp, &prompt, a.tokens, d, chunk, &stop_ids) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("spec-bench: spec {d}: {e}");
                return 1;
            }
        };
        let accept: Vec<f64> = (0..d).map(|k| run.stats.accept_rate(k)).collect();
        let steps = run.stats.steps;
        rows.push(Row {
            drafts: d,
            ids: run.ids,
            steps,
            tokens: run.stats.tokens,
            accept,
            tokens_per_step: run.stats.tokens as f64 / steps.max(1) as f64,
            ms_per_step: run.decode_s * 1e3 / steps.max(1) as f64,
            tok_s: run.stats.tokens as f64 / run.decode_s.max(1e-9),
            prefill_s: run.prefill_s,
            draft_prefill_s: run.draft_prefill_s,
            draft_ms: run.draft_s * 1e3 / steps.max(1) as f64,
            verify_ms: run.verify_s * 1e3 / steps.max(1) as f64,
            rollback_ms: run.rollback_s * 1e3 / steps.max(1) as f64,
            stopped: run.stopped,
        });
    }

    // Exactness against plain decode.
    let base = base_ids;
    let mut exact = Vec::new();
    for r in rows.iter().skip(1) {
        let n = r.ids.len().min(base.len());
        let same = r.ids.iter().zip(&base).take(n).filter(|(a, b)| a == b).count();
        let first_diff = r
            .ids
            .iter()
            .zip(&base)
            .position(|(a, b)| a != b);
        exact.push((r.drafts, same, n, first_diff));
    }

    if a.json {
        let rows_json: Vec<String> = rows
            .iter()
            .map(|r| {
                let acc: Vec<String> = r.accept.iter().map(|x| format!("{x:.4}")).collect();
                format!(
                    "{{\"drafts\":{},\"steps\":{},\"tokens\":{},\"ms_per_step\":{:.4},\
                     \"tokens_per_step\":{:.4},\"tok_s\":{:.4},\"prefill_s\":{:.4},\
                     \"draft_prefill_s\":{:.4},\"draft_ms\":{:.4},\"verify_ms\":{:.4},\
                     \"rollback_ms\":{:.4},\"stopped\":{},\"accept\":[{}],\"ids\":{:?}}}",
                    r.drafts,
                    r.steps,
                    r.tokens,
                    r.ms_per_step,
                    r.tokens_per_step,
                    r.tok_s,
                    r.prefill_s,
                    r.draft_prefill_s,
                    r.draft_ms,
                    r.verify_ms,
                    r.rollback_ms,
                    r.stopped,
                    acc.join(","),
                    r.ids
                )
            })
            .collect();
        println!(
            "{{\"prompt_tokens\":{},\"n_ctx\":{},\"kv\":\"{}\",\"drafter_bytes\":{},\
             \"drafter_kv_bytes\":{},\"rows\":[{}]}}",
            prompt.len(),
            n_ctx,
            a.kv.name(),
            drafter_bytes,
            drafter_kv,
            rows_json.join(",")
        );
        return 0;
    }

    let plain_ms = rows[0].ms_per_step;
    println!();
    println!(
        "{:>6}  {:>10}  {:>13}  {:>10}  {:>9}  {:>10}  {:>8}  {}",
        "drafts", "ms/step", "tokens/step", "tok/s", "vs plain", "accept", "exact", "prefill/draft"
    );
    for r in &rows {
        let acc: Vec<String> = r.accept.iter().map(|x| format!("{x:.3}")).collect();
        let ex = exact.iter().find(|(d, ..)| *d == r.drafts);
        let ex_s = match ex {
            None => "-".to_string(),
            Some((_, same, n, None)) => format!("{same}/{n}"),
            Some((_, same, n, Some(at))) => format!("{same}/{n} @{at}"),
        };
        println!(
            "{:>6}  {:>10.2}  {:>13.3}  {:>10.2}  {:>9.2}x  {:>10}  {:>8}  {:.1}/{:.1} ms",
            r.drafts,
            r.ms_per_step,
            r.tokens_per_step,
            r.tok_s,
            r.tok_s / rows[0].tok_s.max(1e-9),
            if acc.is_empty() { "-".to_string() } else { acc.join(",") },
            ex_s,
            r.prefill_s * 1e3,
            r.draft_prefill_s * 1e3,
        );
    }
    // No-accept downside: the step cost is fixed, so a step that accepts
    // nothing still costs ms_per_step and yields one token.
    for r in rows.iter().skip(1) {
        println!(
            "drafts {}: no-accept {:.2}x a plain step ({:.1} tok/s); \
             step {:.2} ms = draft {:.2} + verify {:.2} + rollback {:.2}",
            r.drafts,
            r.ms_per_step / plain_ms.max(1e-9),
            1e3 / r.ms_per_step.max(1e-9),
            r.ms_per_step,
            r.draft_ms,
            r.verify_ms,
            r.rollback_ms,
        );
    }
    println!(
        "drafter: {:.2} MiB weights + {:.1} MiB KV at ctx {n_ctx}; load+run wall {:.1} s",
        drafter_bytes as f64 / (1024.0 * 1024.0),
        drafter_kv as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64()
    );
    0
}
