//! `hipml generate --device`: the same greedy loop as the CPU oracle, run on
//! the resident-weight GPU graph.
//!
//! The token loop feeds the prompt one token at a time (the hybrid model's
//! recurrent state and KV cache are exactly the decode path) and then emits
//! `--tokens` greedy ids. Decode tok/s excludes the prompt, as the acceptance
//! number asks.
//!
//! `--reps N` keeps the weights resident and builds a fresh `GpuEngine` for
//! every repetition. That is a genuinely cold run (zeroed KV, zeroed GDN
//! state, `pos = 0`) without paying the 17 GB upload again, which is what the
//! bench needs to get five cold prefill numbers inside one GPU-lock window.
//! `--json` makes the result machine-readable for `bench/job.py`.
//!
//! `--stage-report` additionally runs the synchronised stage breakdown once,
//! on its own fresh engine. Those numbers include one stream sync per stage, so
//! they are a diagnostic and never the TTFT that goes in the table.

use std::time::Instant;

use hipml_core::gen::argmax_f32;
use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;
use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

struct Args {
    gguf: String,
    text: Option<String>,
    ids: Option<Vec<u32>>,
    ids_file: Option<String>,
    tokens: usize,
    ctx: Option<usize>,
    compare_cpu: bool,
    kv: KvMode,
    chunk: Option<usize>,
    reps: usize,
    json: bool,
    stage_report: bool,
    graph: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        text: None,
        ids: None,
        ids_file: None,
        tokens: 20,
        ctx: None,
        compare_cpu: false,
        kv: KvMode::F32,
        chunk: None,
        reps: 1,
        json: false,
        stage_report: false,
        graph: false,
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
            "--text" => {
                a.text = Some(need(i)?.to_string());
                i += 2;
            }
            "--ids" => {
                let parsed: Result<Vec<u32>, _> = need(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect();
                a.ids = Some(parsed.map_err(|e| format!("bad --ids: {e}"))?);
                i += 2;
            }
            "--ids-file" => {
                a.ids_file = Some(need(i)?.to_string());
                i += 2;
            }
            "--tokens" => {
                a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--ctx" => {
                a.ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                i += 2;
            }
            "--device" | "--cpu" => i += 1,
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--compare-cpu" => {
                a.compare_cpu = true;
                i += 1;
            }
            "--chunk" => {
                a.chunk = Some(need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?);
                i += 2;
            }
            "--reps" => {
                a.reps = need(i)?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --reps: {e}"))?
                    .max(1);
                i += 2;
            }
            "--json" => {
                a.json = true;
                i += 1;
            }
            "--stage-report" => {
                a.stage_report = true;
                i += 1;
            }
            "--graph" => {
                a.graph = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.compare_cpu && (a.reps > 1 || a.json) {
        return Err("--compare-cpu cannot be combined with --reps > 1 or --json".to_string());
    }
    Ok(a)
}

/// Read a token-id array out of a JSON object (`"ids":[...]`) or a bare array.
/// The fixture is written by `bench/prepare_fixture.py`, so a flat integer
/// array is the only shape that has to work; no serde in this crate.
fn read_ids_file(path: &str) -> Result<Vec<u32>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let start = s
        .find("\"ids\"")
        .and_then(|i| s[i..].find('[').map(|j| i + j))
        .or_else(|| s.find('['))
        .ok_or_else(|| format!("{path}: no token array"))?;
    let rest = &s[start + 1..];
    let end = rest
        .find(']')
        .ok_or_else(|| format!("{path}: unterminated array"))?;
    let mut ids = Vec::new();
    for tok in rest[..end]
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
    {
        ids.push(
            tok.parse::<u32>()
                .map_err(|e| format!("{path}: bad token {tok:?}: {e}"))?,
        );
    }
    if ids.is_empty() {
        return Err(format!("{path}: empty token array"));
    }
    Ok(ids)
}

struct Rep {
    index: usize,
    prefill_ms: f64,
    decode_ms: f64,
    decode_tokens: usize,
    first_id: u32,
    ids: Vec<u32>,
    engine_build_ms: f64,
}

pub fn generate(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("generate: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let g = match Gguf::open(&a.gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("generate: cannot open {}: {e}", a.gguf);
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("generate: tokenizer: {e}");
            return 1;
        }
    };
    let prompt_ids = match (&a.ids_file, &a.ids, &a.text) {
        (Some(path), _, _) => match read_ids_file(path) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("generate: {e}");
                return 1;
            }
        },
        (None, Some(ids), _) => ids.clone(),
        (None, None, Some(text)) => tok.encode(text),
        (None, None, None) => {
            eprintln!("generate: need --text, --ids or --ids-file");
            return 2;
        }
    };
    if prompt_ids.is_empty() {
        eprintln!("generate: empty prompt");
        return 2;
    }
    if !a.json && prompt_ids.len() <= 64 {
        println!("prompt ids: {prompt_ids:?}");
    }
    let n_ctx = a.ctx.unwrap_or(prompt_ids.len() + a.tokens + 8);

    let t_load = Instant::now();
    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("generate: resident load: {e}");
            return 1;
        }
    };
    let model_load_s = t_load.elapsed().as_secs_f64();
    if !a.json {
        println!(
            "resident: {:.2} GiB in {} tensors on {} (upload {:.1}s)",
            report.tensor_bytes as f64 / (1u64 << 30) as f64,
            report.tensor_count,
            model.device_name(),
            report.upload_secs
        );
    }

    // The chunked graph is the default: a prompt tile is batched GEMMs plus a
    // chunked recurrence, not N decode steps. `--chunk 0` (or --compare-cpu,
    // which needs the logit vector) keeps the token loop as the reference.
    let chunk = match a.chunk {
        Some(0) => None,
        Some(n) => Some(n),
        None if a.compare_cpu => None,
        None => Some(512),
    };
    if chunk.is_some() && a.compare_cpu {
        eprintln!(
            "generate: --chunk returns only the sampled id; --compare-cpu needs the token loop"
        );
        return 2;
    }

    let mut reps: Vec<Rep> = Vec::with_capacity(a.reps);
    let mut weight_bytes = 0u64;
    let mut kv_bytes = 0u64;
    for rep in 0..a.reps {
        let t_engine = Instant::now();
        let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("generate: graph: {e}");
                return 1;
            }
        };
        let engine_build_ms = t_engine.elapsed().as_secs_f64() * 1e3;
        if rep == 0 {
            weight_bytes = engine.weight_bytes() as u64;
            kv_bytes = engine.kv_bytes(n_ctx) as u64;
        }
        if rep == 0 && !a.json {
            println!(
                "graph: {} layers, {} streamed bytes/token ({:.2} GiB), {} KV {:.1} MiB at ctx {n_ctx}",
                engine.cfg.n_layer,
                engine.weight_bytes(),
                engine.weight_bytes() as f64 / (1u64 << 30) as f64,
                engine.kv_mode.name(),
                engine.kv_bytes(n_ctx) as f64 / (1024.0 * 1024.0),
            );
            println!(
                "load: {:.1}s total (resident upload + graph build)",
                t0.elapsed().as_secs_f64()
            );
        }

        let mut st = engine.new_state();
        let t_prefill = Instant::now();
        let mut first_id: Option<u32> = None;
        let mut logits = Vec::new();
        if let Some(chunk) = chunk {
            match engine.prefill_chunked(&prompt_ids, &mut st, chunk) {
                Ok(id) => first_id = Some(id),
                Err(e) => {
                    eprintln!("generate: chunked prefill: {e}");
                    return 1;
                }
            }
        } else {
            logits = match first_logits(&mut engine, &mut st, &prompt_ids) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("generate: prefill: {e}");
                    return 1;
                }
            };
        }
        let prefill_s = t_prefill.elapsed().as_secs_f64();

        if a.compare_cpu {
            match cpu_first_logits(&a.gguf, &prompt_ids) {
                Ok(cpu) => {
                    let (mut worst, mut at) = (0.0f32, 0usize);
                    for i in 0..logits.len().min(cpu.len()) {
                        let d = (logits[i] - cpu[i]).abs();
                        if d > worst {
                            worst = d;
                            at = i;
                        }
                    }
                    println!(
                        "first-step logits vs CPU: max |d| = {worst:.6e} at {at} \
                         (device argmax {} / cpu argmax {})",
                        argmax_f32(&logits),
                        argmax_f32(&cpu)
                    );
                }
                Err(e) => eprintln!("compare-cpu: {e}"),
            }
        }

        if a.graph {
            match engine.build_token_graph() {
                Ok(()) => {
                    if !a.json && rep == 0 {
                        println!("decode graph: captured (one hipGraphLaunch per token)");
                    }
                }
                Err(e) => {
                    eprintln!("generate: capture: {e}");
                    return 1;
                }
            }
        }
        let mut ids = Vec::with_capacity(a.tokens);
        let mut decode_s = 0.0f64;
        let mut next = match first_id {
            Some(id) => id,
            None => argmax_f32(&logits) as u32,
        };
        for i in 0..a.tokens {
            ids.push(next);
            if i + 1 < a.tokens {
                let t = Instant::now();
                let got = if a.graph {
                    engine.graph_token(next, &mut st)
                } else {
                    engine.forward_token_id(next, &mut st)
                };
                next = match got {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("generate: decode: {e}");
                        return 1;
                    }
                };
                decode_s += t.elapsed().as_secs_f64();
            }
        }
        let steps = a.tokens.saturating_sub(1);
        if !a.json {
            if a.reps == 1 {
                for (i, id) in ids.iter().enumerate() {
                    println!("  [{i}] {id}");
                }
            }
            println!(
                "rep {}: prefill {:.2} tok/s ({} tokens in {:.1} ms), decode {:.2} tok/s ({} steps in {:.1} ms)",
                rep,
                prompt_ids.len() as f64 / prefill_s.max(1e-9),
                prompt_ids.len(),
                prefill_s * 1e3,
                steps as f64 / decode_s.max(1e-9),
                steps,
                decode_s * 1e3,
            );
        }
        reps.push(Rep {
            index: rep,
            prefill_ms: prefill_s * 1e3,
            decode_ms: decode_s * 1e3,
            decode_tokens: steps,
            first_id: ids.first().copied().unwrap_or(0),
            ids,
            engine_build_ms,
        });
        // Drop the engine before the next rep so every run starts from zeroed
        // KV and zeroed GDN state.
    }

    // Optional synchronised stage breakdown, on its own fresh engine.
    let mut stage: Option<Vec<(String, f64)>> = None;
    if a.stage_report {
        if let Some(chunk) = chunk {
            let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("generate: stage graph: {e}");
                    return 1;
                }
            };
            let mut st = engine.new_state();
            let t = Instant::now();
            match engine.prefill_chunked_report(&prompt_ids, &mut st, chunk) {
                Ok((_id, r)) => {
                    let total = t.elapsed().as_secs_f64() * 1e3;
                    stage = Some(vec![
                        ("total_wall".to_string(), total),
                        ("embed".to_string(), r.embed_ms),
                        ("elementwise".to_string(), r.elementwise_ms),
                        ("gemm_attn".to_string(), r.attn_gemm_ms),
                        ("gemm_ssm".to_string(), r.ssm_gemm_ms),
                        ("gemm_ffn".to_string(), r.ffn_gemm_ms),
                        ("final".to_string(), r.final_ms),
                        ("seq_ssm".to_string(), r.ssm_seq_ms),
                        ("seq_attn".to_string(), r.attn_seq_ms),
                        ("stage_sum".to_string(), r.total_ms()),
                    ]);
                }
                Err(e) => {
                    eprintln!("generate: stage report: {e}");
                    return 1;
                }
            }
        } else {
            eprintln!("generate: --stage-report needs the chunked path");
        }
    }

    if a.json {
        let stage_json = match &stage {
            None => "null".to_string(),
            Some(v) => {
                let fields: Vec<String> = v
                    .iter()
                    .map(|(k, ms)| format!("\"{k}_ms\":{ms:.4}"))
                    .collect();
                format!("{{{}}}", fields.join(","))
            }
        };
        let rep_json: Vec<String> = reps
            .iter()
            .map(|r| {
                let prefill_tps = prompt_ids.len() as f64 / (r.prefill_ms / 1e3).max(1e-9);
                let decode_tps = r.decode_tokens as f64 / (r.decode_ms / 1e3).max(1e-9);
                format!(
                    "{{\"rep\":{},\"ttft_ms\":{:.4},\"prefill_ms\":{:.4},\"prefill_tok_s\":{:.4},\"decode_ms\":{:.4},\"decode_tok_s\":{:.4},\"decode_tokens\":{},\"first_id\":{},\"engine_build_ms\":{:.4},\"ids\":{:?}}}",
                    r.index, r.prefill_ms, r.prefill_ms, prefill_tps,
                    r.decode_ms, decode_tps, r.decode_tokens, r.first_id,
                    r.engine_build_ms, r.ids
                )
            })
            .collect();
        let out = format!(
            "{{\"engine\":\"hipml\",\"prompt_tokens\":{},\"n_ctx\":{},\"gen_tokens\":{},\"model_load_s\":{:.4},\"resident_upload_s\":{:.4},\"weights_bytes\":{},\"kv_bytes\":{},\"kv_mode\":\"{}\",\"reps\":[{}],\"stage\":{}}}",
            prompt_ids.len(),
            n_ctx,
            a.tokens,
            model_load_s,
            report.upload_secs,
            weight_bytes,
            kv_bytes,
            a.kv.name(),
            rep_json.join(","),
            stage_json
        );
        println!("{out}");
        return 0;
    }

    // Human summary uses the last rep, matching the old single-run output.
    let last = match reps.last() {
        Some(r) => r,
        None => return 1,
    };
    let steps = a.tokens.saturating_sub(1);
    println!("gen ids:    {:?}", last.ids);
    println!(
        "prefill:    {:.2} tok/s ({} prompt tokens in {:.1} ms)",
        prompt_ids.len() as f64 / (last.prefill_ms / 1e3).max(1e-9),
        prompt_ids.len(),
        last.prefill_ms
    );
    println!(
        "decode:     {:.2} tok/s ({} steps in {:.1} ms, {:.2} ms/token)",
        steps as f64 / (last.decode_ms / 1e3).max(1e-9),
        steps,
        last.decode_ms,
        last.decode_ms / steps.max(1) as f64
    );
    println!("text:       {}", tok.decode(&last.ids));
    0
}

fn first_logits(
    engine: &mut GpuEngine<'_>,
    st: &mut hipml_model::gpu::GpuState,
    prompt: &[u32],
) -> Result<Vec<f32>, String> {
    // Every prompt token but the last only has to advance the state; the last
    // one produces the logits the caller samples from (and, with
    // `--compare-cpu`, the vector compared against the oracle).
    let (last, rest) = prompt.split_last().ok_or("empty prompt")?;
    for &id in rest {
        engine.forward_token_id(id, st)?;
    }
    engine.forward_token(*last, st)
}

fn cpu_first_logits(path: &str, prompt: &[u32]) -> Result<Vec<f32>, String> {
    use hipml_core::model::Model as CpuModel;
    let m = CpuModel::load_file(path).map_err(|e| format!("cpu load: {e}"))?;
    let mut st = m.new_state(prompt.len() + 1);
    let mut logits = Vec::new();
    for &id in prompt {
        logits = m.forward_token(id, &mut st);
    }
    Ok(logits)
}
