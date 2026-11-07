//! `hipml generate --device`: the same greedy loop as the CPU oracle, run on
//! the resident-weight GPU graph.
//!
//! The token loop feeds the prompt one token at a time (the hybrid model's
//! recurrent state and KV cache are exactly the decode path) and then emits
//! `--tokens` greedy ids. Decode tok/s excludes the prompt, as the acceptance
//! number asks.

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
    tokens: usize,
    ctx: Option<usize>,
    compare_cpu: bool,
    kv: KvMode,
    chunk: Option<usize>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        text: None,
        ids: None,
        tokens: 20,
        ctx: None,
        compare_cpu: false,
        kv: KvMode::F32,
        chunk: None,
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
                let parsed: Result<Vec<u32>, _> =
                    need(i)?.split(',').map(|s| s.trim().parse::<u32>()).collect();
                a.ids = Some(parsed.map_err(|e| format!("bad --ids: {e}"))?);
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
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    Ok(a)
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
    let prompt_ids = match (&a.ids, &a.text) {
        (Some(ids), _) => ids.clone(),
        (None, Some(text)) => tok.encode(text),
        (None, None) => {
            eprintln!("generate: need --text or --ids");
            return 2;
        }
    };
    if prompt_ids.is_empty() {
        eprintln!("generate: empty prompt");
        return 2;
    }
    println!("prompt ids: {prompt_ids:?}");
    let n_ctx = a.ctx.unwrap_or(prompt_ids.len() + a.tokens + 8);

    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("generate: resident load: {e}");
            return 1;
        }
    };
    println!(
        "resident: {:.2} GiB in {} tensors on {} (upload {:.1}s)",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        report.upload_secs
    );

    let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("generate: graph: {e}");
            return 1;
        }
    };
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

    if a.chunk.is_some() && a.compare_cpu {
        eprintln!("generate: --chunk returns only the sampled id; --compare-cpu needs the token loop");
        return 2;
    }
    let mut st = engine.new_state();
    let t_prefill = Instant::now();
    let mut first_id: Option<u32> = None;
    let mut logits = Vec::new();
    if let Some(chunk) = a.chunk {
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

    let mut ids = Vec::with_capacity(a.tokens);
    let mut decode_s = 0.0f64;
    let mut next = match first_id {
        Some(id) => id,
        None => argmax_f32(&logits) as u32,
    };
    for i in 0..a.tokens {
        println!("  [{i}] {next}");
        ids.push(next);
        if i + 1 < a.tokens {
            let t = Instant::now();
            next = match engine.forward_token_id(next, &mut st) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("generate: decode: {e}");
                    return 1;
                }
            };
            decode_s += t.elapsed().as_secs_f64();
        }
    }
    let steps = a.tokens.saturating_sub(1).max(1) as f64;
    println!("gen ids:    {ids:?}");
    println!(
        "prefill:    {:.2} tok/s ({} prompt tokens in {:.1} ms)",
        prompt_ids.len() as f64 / prefill_s.max(1e-9),
        prompt_ids.len(),
        prefill_s * 1e3
    );
    println!(
        "decode:     {:.2} tok/s ({} steps in {:.1} ms, {:.2} ms/token)",
        steps / decode_s.max(1e-9),
        a.tokens.saturating_sub(1),
        decode_s * 1e3,
        decode_s * 1e3 / steps
    );
    println!("text:       {}", tok.decode(&ids));
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

struct BenchArgs {
    gguf: String,
    depth: usize,
    tokens: usize,
    kv: KvMode,
    kv_set: bool,
    prefill: Option<usize>,
    chunk: Option<usize>,
}

fn parse_bench(args: &[String]) -> Result<BenchArgs, String> {
    let mut a = BenchArgs {
        gguf: String::new(),
        depth: 0,
        tokens: 8,
        kv: KvMode::Q8,
        kv_set: false,
        prefill: None,
        chunk: None,
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
            "--depth" => {
                a.depth = need(i)?.parse().map_err(|e| format!("bad --depth: {e}"))?;
                i += 2;
            }
            "--tokens" => {
                a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                a.kv_set = true;
                i += 2;
            }
            "--prefill" => {
                a.prefill = Some(need(i)?.parse().map_err(|e| format!("bad --prefill: {e}"))?);
                i += 2;
            }
            "--chunk" => {
                a.chunk = Some(need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?);
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    Ok(a)
}

/// `hipml gpu-bench`: measured decode tok/s of the full graph.
///
/// `--prefill N` runs a real N-token prompt through the same token loop and
/// reports prefill tok/s. `--depth D` pre-fills the quantized KV caches with
/// valid blocks to D positions and times decode there; the SSM state is zero,
/// whose step cost is identical.
pub fn bench(args: &[String]) -> i32 {
    let a = match parse_bench(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gpu-bench: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gpu-bench: load: {e}");
            return 1;
        }
    };
    println!(
        "resident: {:.2} GiB in {} tensors on {} (upload {:.1}s)",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        report.upload_secs
    );

    // A prompt tile appends to the f32 cache; the quantized caches are the
    // decode-at-depth path.
    let kv = if a.prefill.is_some() && !a.kv_set {
        KvMode::F32
    } else {
        a.kv
    };
    let n_ctx = if let Some(n) = a.prefill { n } else { a.depth } + a.tokens + 8;
    let mut engine = match GpuEngine::new(&model, n_ctx, kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("gpu-bench: graph: {e}");
            return 1;
        }
    };
    let kv_bytes = engine.kv_bytes(n_ctx);
    println!(
        "graph: {} layers, {:.2} GiB streamed/token, {} KV {:.3} GiB at depth {}",
        engine.cfg.n_layer,
        engine.weight_bytes() as f64 / (1u64 << 30) as f64,
        engine.kv_mode.name(),
        kv_bytes as f64 / (1u64 << 30) as f64,
        if a.prefill.is_some() { a.prefill.unwrap() } else { a.depth },
    );
    println!("build: {:.1}s", t0.elapsed().as_secs_f64());

    let mut st = engine.new_state();
    if let Some(n) = a.prefill {
        let prompt: Vec<u32> = (0..n).map(|i| 100 + (i % 1000) as u32).collect();
        let chunk = a.chunk.unwrap_or(256);
        let t = Instant::now();
        let sink = match engine.prefill_chunked(&prompt, &mut st, chunk) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("gpu-bench: prefill: {e}");
                return 1;
            }
        };
        let s = t.elapsed().as_secs_f64();
        println!(
            "prefill: {n} tokens in {:.2}s ({:.2} tok/s, {:.2} ms/token) [chunk {chunk}, {:.1} MiB scratch, last id {sink}]",
            s,
            n as f64 / s,
            s * 1e3 / n as f64,
            engine.chunk_bytes(chunk) as f64 / (1024.0 * 1024.0)
        );
        return 0;
    }

    if kv == KvMode::F32 {
        eprintln!("gpu-bench: --depth needs a quantized cache (--kv q8 or q4)");
        return 2;
    }
    if let Err(e) = engine.fill_synthetic(a.depth) {
        eprintln!("gpu-bench: synthetic fill: {e}");
        return 1;
    }
    let mut st = engine.state_at(a.depth);
    // Warm: two tokens, so clocks and caches settle before the measured loop.
    let mut last = 0u32;
    for _ in 0..2 {
        match engine.forward_token_id(1, &mut st) {
            Ok(l) => last = l,
            Err(e) => {
                eprintln!("gpu-bench: warm: {e}");
                return 1;
            }
        }
    }
    let n = a.tokens.max(2);
    let t = Instant::now();
    for _ in 0..n {
        match engine.forward_token_id(1, &mut st) {
            Ok(l) => last = l,
            Err(e) => {
                eprintln!("gpu-bench: decode: {e}");
                return 1;
            }
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "decode at {}: {} tokens in {:.3}s ({:.3} ms/token, {:.2} tok/s) [argmax {}]",
        a.depth,
        n,
        s,
        s * 1e3 / n as f64,
        n as f64 / s,
        last
    );
    0
}
