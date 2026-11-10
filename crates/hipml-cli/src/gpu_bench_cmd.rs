// SPDX-License-Identifier: Apache-2.0
//! `hipml gpu-bench`: decode speed at depth and prefill throughput.
//!
//! Split from `device_cmd` because it drives the engine differently: no
//! tokenizer, no prompt text, and the depth numbers come from a cache filled
//! with format-valid blocks.

use std::time::Instant;

use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

struct BenchArgs {
    gguf: String,
    depth: usize,
    tokens: usize,
    kv: KvMode,
    kv_set: bool,
    prefill: Option<usize>,
    chunk: Option<usize>,
    /// Report the prefill stage breakdown (adds a sync per stage).
    split: bool,
    graph_probe: bool,
    graph: bool,
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
        split: false,
        graph_probe: false,
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
            "--prefill-split" => {
                a.split = true;
                i += 1;
            }
            "--prefill-f32" => {
                ::hipml_kernels::set_prefill_path(::hipml_kernels::PrefillPath::F32);
                i += 1;
            }
            "--prefill-dot2" => {
                ::hipml_kernels::set_prefill_path(::hipml_kernels::PrefillPath::Dot2);
                i += 1;
            }
            "--graph-probe" => {
                a.graph_probe = true;
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
        let chunk = a.chunk.unwrap_or(2048);
        let t = Instant::now();
        let sink = if a.split {
            match engine.prefill_chunked_report(&prompt, &mut st, chunk) {
                Ok((id, rep)) => {
                    let s = t.elapsed().as_secs_f64();
                    println!(
                        "prefill: {n} tokens in {:.2}s ({:.2} tok/s) [chunk {chunk}, last id {id}]",
                        s,
                        n as f64 / s
                    );
                    println!(
                        "  embed      {:8.1} ms\n  elementwise{:8.1} ms\n  gemm attn  {:8.1} ms\n  gemm ssm   {:8.1} ms\n  gemm ffn   {:8.1} ms\n  final      {:8.1} ms\n  seq ssm    {:8.1} ms\n  seq attn   {:8.1} ms",
                        rep.embed_ms,
                        rep.elementwise_ms,
                        rep.attn_gemm_ms,
                        rep.ssm_gemm_ms,
                        rep.ffn_gemm_ms,
                        rep.final_ms,
                        rep.ssm_seq_ms,
                        rep.attn_seq_ms
                    );
                    id
                }
                Err(e) => {
                    eprintln!("gpu-bench: prefill: {e}");
                    return 1;
                }
            }
        } else {
            match engine.prefill_chunked(&prompt, &mut st, chunk) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("gpu-bench: prefill: {e}");
                    return 1;
                }
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
    if a.graph_probe {
        match engine.graph_probe(1, &mut st, 16) {
            Ok(g) => {
                println!(
                    "graph probe ({} launches): eager {:7.3} ms/token (submit {:6.3}) | graph {:7.3} ms/token (submit {:6.3}) | graph speedup {:.2}x",
                    g.launches,
                    g.eager_ms,
                    g.eager_submit_ms,
                    g.graph_ms,
                    g.graph_submit_ms,
                    g.eager_ms / g.graph_ms
                );
            }
            Err(e) => {
                eprintln!("gpu-bench: graph probe: {e}");
                return 1;
            }
        }
        return 0;
    }

    let n = a.tokens.max(2);
    let mut path = "eager";
    if a.graph {
        if let Err(e) = engine.build_token_graph() {
            eprintln!("gpu-bench: capture: {e}");
            return 1;
        }
        path = "graph";
    }
    let t = Instant::now();
    for _ in 0..n {
        let got = if a.graph {
            engine.graph_token(1, &mut st)
        } else {
            engine.forward_token_id(1, &mut st)
        };
        match got {
            Ok(l) => last = l,
            Err(e) => {
                eprintln!("gpu-bench: decode: {e}");
                return 1;
            }
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "decode at {} [{path}]: {} tokens in {:.3}s ({:.3} ms/token, {:.2} tok/s) [argmax {}]",
        a.depth,
        n,
        s,
        s * 1e3 / n as f64,
        n as f64 / s,
        last
    );
    0
}
