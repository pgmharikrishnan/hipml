// SPDX-License-Identifier: Apache-2.0
//! `hipml cache-gate`: the undersized-cap correctness gate.
//!
//! One prompt is snapshotted with a cache byte bound smaller than the snapshot
//! it needs. The capture must be *refused*: no entry, no truncated/zero-byte
//! placeholder. An exact repeat must then either miss the cache (full prefill)
//! or fall back to the shorter admitted prefix, but it must never restore a
//! shorter snapshot and call it exact. The old `cache-bench` consumer did
//! exactly that when the end snapshot was refused, and diverged from the
//! uncached ids.
//!
//! The `legacy` check below replays that consumer deliberately: it restores the
//! longest prefix (not an exact hit) and treats its logits as the answer. That
//! path must diverge, which is what makes this gate able to catch the defect.

use std::time::Instant;

use hipml_core::gen::argmax_f32;
use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;
use hipml_model::gpu::{GpuEngine, KvMode, PrefixHit};
use hipml_model::Model;

use crate::cache_cmd::{decode, fill, run, SYSTEM};

struct Args {
    gguf: String,
    len: usize,
    shared: usize,
    chunk: usize,
    tokens: usize,
    bytes: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        len: 8192,
        shared: 4096,
        chunk: 2048,
        tokens: 20,
        bytes: 1024 * 1024 * 1024,
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
            "--len" => {
                a.len = need(i)?.parse().map_err(|e| format!("bad --len: {e}"))?;
                i += 2;
            }
            "--shared" => {
                a.shared = need(i)?.parse().map_err(|e| format!("bad --shared: {e}"))?;
                i += 2;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--tokens" => {
                a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--bytes-mib" => {
                let mib: usize = need(i)?
                    .parse()
                    .map_err(|e| format!("bad --bytes-mib: {e}"))?;
                a.bytes = mib.max(1) * 1024 * 1024;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.shared == 0 || a.shared >= a.len {
        return Err("--shared must be in 1..--len".to_string());
    }
    Ok(a)
}

/// The old exact-repeat consumer: any longest-prefix hit is treated as exact.
fn legacy_ids(engine: &mut GpuEngine<'_>, hit: PrefixHit, n: usize) -> Result<Vec<u32>, String> {
    let mut st = engine.new_state();
    let cached = engine
        .prefix_restore(hit, &mut st)?
        .ok_or("legacy restore has no cached logits")?;
    let first = argmax_f32(&cached) as u32;
    decode(engine, &mut st, first, n)
}

fn gate(engine: &mut GpuEngine<'_>, a: &Args) -> Result<(), String> {
    let g = Gguf::open(&a.gguf).map_err(|e| format!("{}: {e}", a.gguf))?;
    let tok = Tokenizer::from_gguf(&g).map_err(|e| format!("tokenizer: {e}"))?;
    let mut p = fill(&tok, SYSTEM, a.shared);
    let suffix = fill(
        &tok,
        "Question one: why does a KV-only prefix cache stop at 1.38x on this \
         architecture, and what else has to be restored at the boundary?",
        a.len - a.shared,
    );
    p.extend_from_slice(&suffix);

    // 1. Reference: the same prompt with no cache involvement.
    let t = Instant::now();
    let (_, ids_cold) = run(engine, &p, a.chunk, a.tokens)?;
    let cold_ms = t.elapsed().as_secs_f64() * 1e3;

    // 2. Capture with a cap below the full snapshot. The shared prefix is
    //    admitted first; that shorter entry is what makes the legacy path
    //    dangerous when the full snapshot is refused.
    engine.reset_state()?;
    let mut st = engine.new_state();
    engine.prefill_chunked(&p[..a.shared], &mut st, a.chunk)?;
    let logits_shared = engine.logits_host()?;
    let shared = engine.prefix_capture(&p, a.shared, Some(&logits_shared))?;
    if shared.refused {
        return Err(format!(
            "the shared P@{} half ({:.1} MiB) must fit a {:.0} MiB cap",
            a.shared,
            shared.bytes as f64 / 1048576.0,
            a.bytes as f64 / 1048576.0
        ));
    }
    engine.prefill_chunked(&p[a.shared..], &mut st, a.chunk)?;
    let logits = engine.logits_host()?;
    let cap = engine.prefix_capture(&p, p.len(), Some(&logits))?;
    drop(st);
    if !cap.refused {
        return Err(format!(
            "cap {:.0} MiB admitted a {:.1} MiB snapshot; the gate needs a cap below one entry",
            a.bytes as f64 / 1048576.0,
            cap.bytes as f64 / 1048576.0
        ));
    }
    println!(
        "capture @{} refused: needs {:.1} MiB, cap {:.0} MiB",
        p.len(),
        cap.bytes as f64 / 1048576.0,
        a.bytes as f64 / 1048576.0
    );

    // 3. An exact repeat must not be an exact hit; it must fall back.
    if let Some(h) = engine.prefix_lookup_exact(&p) {
        return Err(format!(
            "undersized cap served an exact hit at {} tokens",
            h.pos
        ));
    }
    let (_, ids_fallback) = run(engine, &p, a.chunk, a.tokens)?;
    if ids_fallback != ids_cold {
        return Err("refused-snapshot fallback ids differ from the uncached run".to_string());
    }

    // 4. Prove the gate catches the old consumer. A shorter prefix must exist,
    //    and restoring it as if it were exact must diverge.
    let partial = engine
        .prefix_lookup(&p)
        .filter(|h| h.pos < p.len())
        .ok_or("gate needs a shorter admitted prefix to replay the old consumer")?;
    let ids_legacy = legacy_ids(engine, partial, a.tokens)?;
    let diverge = ids_legacy
        .iter()
        .zip(ids_cold.iter())
        .position(|(a, b)| a != b);
    if diverge.is_none() {
        return Err(format!(
            "legacy consumer did not diverge at a partial hit of {} tokens",
            partial.pos
        ));
    }
    println!(
        "legacy consumer (partial hit @{} used as exact) diverges at id {}",
        partial.pos,
        diverge.unwrap()
    );

    let s = engine.prefix_stats().ok_or("cache disabled")?;
    if s.refused == 0 {
        return Err("refusal was not counted in the cache stats".to_string());
    }
    println!(
        "fallback ids equal cold ids ({} tokens); stats: {} entries, {} refused, {} hits, {} misses",
        a.tokens, s.entries, s.refused, s.hits, s.misses
    );
    println!("cold {cold_ms:.1} ms; an undersized cap is a miss, not a divergent hit: pass");
    Ok(())
}

pub fn run_gate(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("cache-gate: {e}");
            return 2;
        }
    };
    let (model, _) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cache-gate: load: {e}");
            return 1;
        }
    };
    let n_ctx = a.len + a.tokens + 8;
    let mut engine = match GpuEngine::new(&model, n_ctx, KvMode::F32) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cache-gate: graph: {e}");
            return 1;
        }
    };
    engine.enable_prefix_cache(2, a.bytes);
    let _ = engine.chunk_bytes(a.chunk);
    let warm: Vec<u32> = (0..8).map(|i| 100 + i).collect();
    let mut st = engine.new_state();
    if let Err(e) = engine.prefill_chunked(&warm, &mut st, a.chunk.min(8)) {
        eprintln!("cache-gate: warmup: {e}");
        return 1;
    }
    match gate(&mut engine, &a) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("cache-gate: {e}");
            1
        }
    }
}
