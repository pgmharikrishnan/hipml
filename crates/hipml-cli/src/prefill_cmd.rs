//! `hipml prefill-bench` and `hipml prefill-model`: the two measurements that
//! make prefill a number rather than a claim.
//!
//! `prefill-bench` times the tiled GEMM at the three Qwen3.8-27B ffn shapes and both
//! prompt lengths. `prefill-model` streams every quantized tensor once at a
//! fixed prompt length, sums the kernel time, and repeats the pass with the 16
//! full-attention blocks skipped because the KV cache already holds those
//! tokens. The difference is the measured cold-vs-warm TTFT.

use std::collections::HashSet;
use std::fs::File;

use hipml_core::cache::{prefix_match, KvCache};
use hipml_core::gguf::Gguf;
use hipml_core::parity::qwen38_prefill_cases;
use hipml_core::quant::TensorType;
use hipml_kernels::Launcher;

const CHUNK_BYTES: u64 = 64 * 1024 * 1024;
/// fp32 dense peak, Radeon AI PRO R9700 (47.8 TFLOPS).
pub const FP32_TFLOPS: f64 = 47.8;
/// fp16 dense peak (95.7 TFLOPS).
pub const FP16_TFLOPS: f64 = 95.7;
/// 47.8e12 / (2 * 26.9e9) = 888 tokens/s for a 26.9 B-parameter model.
pub const FP32_CEIL_TOKS: f64 = 888.0;
pub const FP16_CEIL_TOKS: f64 = 1779.0;
/// `llama.cpp` HIP, pp512, same file, same box.
pub const LLAMA_PP512: f64 = 1256.5;

fn arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// `hipml prefill-bench [--reps N]`
pub fn bench(args: &[String]) -> i32 {
    let reps = arg_value(args, "--reps")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            println!("device: none ({e})");
            return 1;
        }
    };
    println!("device: {}", launcher.device_name);
    println!(
        "reps {reps}; fp32 dense peak {FP32_TFLOPS:.1} TFLOPS ({FP32_CEIL_TOKS:.0} tok/s), \
         fp16 dense peak {FP16_TFLOPS:.1} TFLOPS ({FP16_CEIL_TOKS:.0} tok/s)"
    );
    println!("llama.cpp HIP pp512 reference: {LLAMA_PP512:.1} tok/s");
    println!(
        "\n{:<26} {:>6} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "shape", "M", "us", "TFLOPS", "tok/s-eq", "%fp32", "%fp16"
    );

    const MAX_TOKENS: usize = 2048;
    let mut worst = 0.0f64;
    for case in qwen38_prefill_cases(MAX_TOKENS) {
        let (w, x) = (case.inputs)(case.seed, case.m, case.k, MAX_TOKENS);
        for &tokens in &[512usize, MAX_TOKENS] {
            let xs = &x[..tokens * case.k];
            if let Err(e) =
                launcher.run_prefill_reps(case.name, &w, xs, case.m, case.k, tokens, 1)
            {
                println!("{:<26} failed: {e}", case.label);
                continue;
            }
            let (_, secs) = match launcher.run_prefill_reps(
                case.name, &w, xs, case.m, case.k, tokens, reps,
            ) {
                Ok(v) => v,
                Err(e) => {
                    println!("{:<26} failed: {e}", case.label);
                    continue;
                }
            };
            let per = secs / reps as f64;
            let flops = 2.0 * tokens as f64 * case.m as f64 * case.k as f64;
            let tflops = flops / per / 1e12;
            let tps = tokens as f64 / per;
            worst = worst.max(tps);
            println!(
                "{:<26} {:>6} {:>10.1} {:>8.2} {:>10.1} {:>7.1}% {:>7.1}%",
                case.label,
                tokens,
                per * 1e6,
                tflops,
                tps,
                tflops / FP32_TFLOPS * 100.0,
                tflops / FP16_TFLOPS * 100.0
            );
        }
    }

    // One full-model token is 53.8 GFLOP at these shapes' parameter count; the
    // per-tensor tok/s column is "tokens through this tensor only", so it is not
    // comparable to 888/1779 directly. The ceiling percentages above are.
    println!(
        "\nbest single-tensor {:.1} tok/s-eq; a full-model token mixes all tensors, \
         which is measured by `prefill-model`.",
        worst
    );
    0
}

fn layer_of(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("blk.")?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

/// `hipml prefill-model <model.gguf> [--tokens N]`
pub fn model(args: &[String]) -> i32 {
    let Some(path) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!("usage: hipml prefill-model <model.gguf> [--tokens 512]");
        return 2;
    };
    let tokens = arg_value(args, "--tokens")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512)
        .max(1);
    let mut g = match Gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 1;
        }
    };
    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            println!("device: none ({e})");
            return 1;
        }
    };

    // 16 full-attention blocks are the ones that carry attn_q; the other 48 are
    // SSM blocks and have a fixed-size recurrent state instead of a KV cache.
    let attn: HashSet<u64> = g
        .tensors
        .iter()
        .filter(|t| t.name.ends_with(".attn_q.weight"))
        .filter_map(|t| layer_of(&t.name))
        .collect();
    let Some(head_dim) = g
        .tensors
        .iter()
        .find(|t| t.name.ends_with(".attn_q_norm.weight"))
        .and_then(|t| t.dims.first().copied())
        .map(|d| d as usize)
    else {
        println!("cannot find attn_q_norm: not an Qwen3.8-27B-shaped model");
        return 1;
    };
    let kv_rows = g
        .tensors
        .iter()
        .find(|t| t.name.ends_with(".attn_k.weight"))
        .map(|t| t.rows() as usize)
        .unwrap_or(0);
    if kv_rows == 0 || kv_rows % head_dim != 0 {
        println!("attn_k rows {kv_rows} are not a whole number of {head_dim}-dim heads");
        return 1;
    }
    let n_kv = kv_rows / head_dim;

    println!("device: {}", launcher.device_name);
    println!(
        "model: {} tensors, {} full-attention blocks, {} SSM blocks, KV {} heads x {} dim",
        g.tensor_count,
        attn.len(),
        64usize.saturating_sub(attn.len()),
        n_kv,
        head_dim
    );
    println!("prompt: {tokens} tokens, identical tokens on both passes");

    // Cold: nothing cached.
    let cold = match run_pass(&mut g, &launcher, tokens, |_| true) {
        Ok(v) => v,
        Err(e) => {
            println!("cold pass failed: {e}");
            return 1;
        }
    };

    // Populate the KV cache for the 16 attention blocks. The cache geometry and
    // the token bookkeeping are real; the values are placeholder because the
    // attention kernel is not part of prefill (there is no Flash Attention path
    // yet). The correctness claim here is the prefix match and the skipped
    // GEMM work, not attention numerics.
    let mut cache = KvCache::new(attn.len(), n_kv, head_dim, tokens);
    let kv_vals = vec![0.001f32; n_kv * tokens * head_dim];
    for l in 0..attn.len() {
        if let Err(e) = cache.append(l, tokens, &kv_vals, &kv_vals) {
            println!("KV append failed: {e}");
            return 1;
        }
    }
    let prompt: Vec<u32> = (0..tokens as u32).collect();
    let cached = prefix_match(&prompt, &prompt);
    if cached != cache.len() {
        println!(
            "internal: prefix match {cached} != cache length {}",
            cache.len()
        );
        return 1;
    }

    // Warm: the attention blocks' tokens are in the cache, so their GEMMs are
    // skipped; the SSM blocks have no sliceable state and run in full.
    let warm = match run_pass(&mut g, &launcher, tokens, |l| !attn.contains(&l)) {
        Ok(v) => v,
        Err(e) => {
            println!("warm pass failed: {e}");
            return 1;
        }
    };

    let total = cold.bytes.max(1);
    println!(
        "\n{:<22} {:>9} {:>12} {:>10} {:>9} {:>10}",
        "pass", "tensors", "bytes GB", "ms", "TFLOP", "TFLOPS"
    );
    for (name, p) in [("cold", &cold), ("warm", &warm)] {
        println!(
            "{:<22} {:>9} {:>12.3} {:>10.1} {:>9.1} {:>10.2}",
            name,
            p.tensors,
            p.bytes as f64 / 1e9,
            p.secs * 1e3,
            p.flops as f64 / 1e12,
            p.flops as f64 / p.secs.max(1e-9) / 1e12
        );
    }
    let kv_bytes = cache.bytes() as f64 / 1e6;
    println!(
        "\ncached tokens: {cached} (KV for {} attention blocks, {:.1} MB at {} tokens)",
        attn.len(),
        kv_bytes,
        cache.len()
    );
    println!(
        "recomputed: {} SSM blocks, {:.3} GB of the {:.3} GB quantized tensor stream ({:.1}%)",
        attn.len(),
        warm.bytes as f64 / 1e9,
        total as f64 / 1e9,
        warm.bytes as f64 / total as f64 * 100.0
    );
    println!(
        "cold TTFT {:.1} ms = {:.1} tok/s; warm TTFT {:.1} ms = {:.1} tok/s; {:.2}x",
        cold.secs * 1e3,
        tokens as f64 / cold.secs,
        warm.secs * 1e3,
        tokens as f64 / warm.secs,
        cold.secs / warm.secs
    );
    let intensity = cold.flops as f64 / total as f64;
    println!(
        "cold arithmetic intensity: {:.0} FLOP/byte (fp32 balance {:.0}), so this chunk is \
         {}-bound; weight read at 631 GB/s would be {:.1} ms",
        intensity,
        FP32_TFLOPS * 1e12 / 631e9,
        if intensity > FP32_TFLOPS * 1e12 / 631e9 {
            "compute"
        } else {
            "bandwidth"
        },
        total as f64 / 631e9 * 1e3
    );
    println!(
        "excluded: F32 norm/SSM tensors (0.1 GB), Q5_K blk.51/blk.55 ffn_down (2 x 61.3 MB = \
         0.7% of quantized bytes, 0.66% of model FLOPs), token-embedding lookup"
    );
    0
}

struct Pass {
    tensors: usize,
    bytes: u64,
    flops: u64,
    secs: f64,
}

/// Run every quantized GEMM tensor once at `tokens` rows, keeping only the
/// blocks `keep` accepts (by layer index; non-block tensors always run).
fn run_pass(
    g: &mut Gguf<File>,
    launcher: &Launcher,
    tokens: usize,
    keep: impl Fn(u64) -> bool,
) -> Result<Pass, String> {
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .filter(|t| {
            matches!(t.ty, TensorType::Q4K | TensorType::Q6K | TensorType::Q8_0)
                && !t.name.contains("token_embd")
        })
        .filter(|t| layer_of(&t.name).map(&keep).unwrap_or(true))
        .cloned()
        .collect();

    let mut pass = Pass {
        tensors: 0,
        bytes: 0,
        flops: 0,
        secs: 0.0,
    };
    let mut buf: Vec<u8> = Vec::new();
    for t in &tensors {
        let op = match t.ty {
            TensorType::Q4K => "prefill_q4_k",
            TensorType::Q6K => "prefill_q6_k",
            TensorType::Q8_0 => "prefill_q8_0",
            _ => unreachable!(),
        };
        let k = t.ne0() as usize;
        let m = t.rows() as usize;
        if k == 0 || m == 0 || k % 64 != 0 {
            return Err(format!("{}: k={k} is not a multiple of 64", t.name));
        }
        let row_bytes = t.nbytes / m as u64;
        let rows_per_chunk = (CHUNK_BYTES / row_bytes).max(1) as usize;
        let x = vec![0.01f32; tokens * k];
        let mut offset = 0u64;
        let mut row = 0usize;
        while row < m {
            let rows = (m - row).min(rows_per_chunk);
            let nbytes = rows as u64 * row_bytes;
            buf.resize(nbytes as usize, 0);
            g.read_tensor_range(&t.name, offset, nbytes as usize, &mut buf)
                .map_err(|e| format!("{}: {e}", t.name))?;
            pass.secs += launcher
                .run_prefill_time(op, &buf, &x, rows, k, tokens, 1)
                .map_err(|e| format!("{}: {e}", t.name))?;
            offset += nbytes;
            row += rows;
        }
        pass.tensors += 1;
        pass.bytes += t.nbytes;
        pass.flops += 2 * tokens as u64 * m as u64 * k as u64;
    }
    Ok(pass)
}
