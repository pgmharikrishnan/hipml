//! `hipml cache-bench`: cold, warm-exact and warm-shared TTFT on the real
//! hybrid graph, with the 48 SSM recurrent states snapshotted alongside the 16
//! f32 attention KV caches.
//!
//! Three prompts drive it:
//!
//! * `P` = shared system prefix + suffix A
//! * `Q` = the same system prefix + suffix B
//! * `R` = an unrelated prompt
//!
//! The seed pass snapshots `P` at the shared boundary and at its end. An exact
//! repeat of `P` restores the end snapshot and takes the cached next-token
//! logits; `Q` restores the shared boundary and prefills only its suffix. A
//! small cache bound then shows LRU eviction and the resulting partial hit.

use std::time::Instant;

use hipml_core::gen::argmax_f32;
use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;
use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

const SYSTEM: &str = "You are a careful technical assistant. Answer with \
    measured numbers and state what is unverified. Read the whole question, \
    keep the reasoning short, and never invent a benchmark that was not run. \
    Prefer an exact token match over a guess, and say when a cache cannot be \
    invalidated safely. The context below is shared by every request in this \
    batch, so it is the part a prefix cache is expected to reuse.";

struct Args {
    gguf: String,
    len: usize,
    shared: usize,
    chunk: usize,
    tokens: usize,
    entries: usize,
    bytes: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        len: 2048,
        shared: 1024,
        chunk: 512,
        tokens: 20,
        entries: 2,
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
            "--entries" => {
                a.entries = need(i)?.parse().map_err(|e| format!("bad --entries: {e}"))?;
                i += 2;
            }
            "--bytes-mib" => {
                let mib: usize = need(i)?.parse().map_err(|e| format!("bad --bytes-mib: {e}"))?;
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
    if a.chunk == 0 {
        return Err("--chunk must be positive (this command uses chunked prefill)".to_string());
    }
    Ok(a)
}

/// Repeat `base` until it is exactly `n` tokens. The result is a valid token
/// sequence; repeating the source text keeps it realistic without inventing
/// ids.
fn fill(tok: &Tokenizer, text: &str, n: usize) -> Vec<u32> {
    let mut base = tok.encode(text);
    if base.is_empty() {
        base = vec![220];
    }
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        out.extend_from_slice(&base);
    }
    out.truncate(n);
    out
}

fn decode(
    engine: &mut GpuEngine<'_>,
    st: &mut hipml_model::gpu::GpuState,
    first: u32,
    n: usize,
) -> Result<Vec<u32>, String> {
    let mut ids = Vec::with_capacity(n);
    ids.push(first);
    while ids.len() < n {
        let next = engine.forward_token_id(*ids.last().unwrap(), st)?;
        ids.push(next);
    }
    Ok(ids)
}

/// Prefill `tokens` (from `st.pos`), then emit `n` greedy ids.
fn run(
    engine: &mut GpuEngine<'_>,
    tokens: &[u32],
    chunk: usize,
    n: usize,
) -> Result<(f64, Vec<u32>), String> {
    engine.reset_state()?;
    let mut st = engine.new_state();
    let t = Instant::now();
    let first = engine.prefill_chunked(tokens, &mut st, chunk)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    Ok((ms, decode(engine, &mut st, first, n)?))
}

fn main_bench(engine: &mut GpuEngine<'_>, a: &Args) -> Result<(), String> {
    let g = Gguf::open(&a.gguf).map_err(|e| format!("{}: {e}", a.gguf))?;
    let tok = Tokenizer::from_gguf(&g).map_err(|e| format!("tokenizer: {e}"))?;

    let prefix = fill(&tok, SYSTEM, a.shared);
    let suffix_a = fill(
        &tok,
        "Question one: why does a KV-only prefix cache stop at 1.38x on this \
         architecture, and what else has to be restored at the boundary?",
        a.len - a.shared,
    );
    let suffix_b = fill(
        &tok,
        "Question two: compare the memory cost of one cached prefix with the \
         prefill time it saves, then choose an eviction bound for a 32 GB card.",
        a.len - a.shared,
    );
    let mut p = prefix.clone();
    p.extend_from_slice(&suffix_a);
    let mut q = prefix.clone();
    q.extend_from_slice(&suffix_b);
    let r = fill(
        &tok,
        "Write a short poem about the ocean at dawn, then explain the tides.",
        a.len,
    );
    if p[..a.shared] != q[..a.shared] {
        return Err("internal: the two prompts do not share the prefix".to_string());
    }

    println!(
        "prompts: P and Q {} tokens, shared prefix {} ({:.0}%), unrelated R {}",
        a.len,
        a.shared,
        100.0 * a.shared as f64 / a.len as f64,
        r.len()
    );
    println!(
        "cache: {} entries / {:.0} MiB, chunk {}; decode {} ids per case",
        a.entries,
        a.bytes as f64 / (1024.0 * 1024.0),
        a.chunk,
        a.tokens
    );

    // 1. Cold P: the baseline TTFT and the reference ids.
    let (cold_p, ids_p) = run(engine, &p, a.chunk, a.tokens)?;

    // 2. Seed: snapshot at the shared boundary and at the end. Not timed.
    engine.reset_state()?;
    let mut st = engine.new_state();
    engine.prefill_chunked(&p[..a.shared], &mut st, a.chunk)?;
    let logits_shared = engine.logits_host()?;
    let at_shared = engine.prefix_capture(&p, a.shared, Some(&logits_shared))?;
    engine.prefill_chunked(&p[a.shared..], &mut st, a.chunk)?;
    let logits_end = engine.logits_host()?;
    let at_end = engine.prefix_capture(&p, p.len(), Some(&logits_end))?;
    drop(st);
    let seeded = engine.prefix_stats().ok_or("cache disabled")?;
    let entries = engine.prefix_entries().unwrap_or_default();
    let bytes_at = |i: usize| entries.get(i).map(|e| e.1).unwrap_or(0) as f64 / 1048576.0;
    println!(
        "seed: captured P@{} ({:.1} MiB) and P@{} ({:.1} MiB); cache {} entries / {:.1} MiB{}",
        a.shared,
        bytes_at(0),
        a.len,
        bytes_at(1),
        seeded.entries,
        seeded.bytes as f64 / 1048576.0,
        if at_shared.evicted + at_end.evicted > 0 {
            " (evicted)"
        } else {
            ""
        }
    );

    // 3. Hit latency: lookup + restore alone, no forward pass.
    let mut st = engine.new_state();
    let t = Instant::now();
    let hit = engine.prefix_lookup(&p).ok_or("exact lookup missed")?;
    let cached = engine
        .prefix_restore(hit, &mut st)?
        .ok_or("end snapshot has no cached logits")?;
    let restore_ms = t.elapsed().as_secs_f64() * 1e3;

    // 4. Warm exact P: restore + cached logits, then decode.
    let mut st = engine.new_state();
    let t = Instant::now();
    let hit = engine.prefix_lookup(&p).ok_or("exact lookup missed")?;
    let cached = engine
        .prefix_restore(hit, &mut st)?
        .ok_or("end snapshot has no cached logits")?;
    let first = argmax_f32(&cached) as u32;
    let warm_exact = t.elapsed().as_secs_f64() * 1e3;
    let ids_exact = decode(engine, &mut st, first, a.tokens)?;

    // 5. Cold Q: a shared prefix but a different suffix, computed from zero.
    let (cold_q, ids_q) = run(engine, &q, a.chunk, a.tokens)?;

    // 6. Warm shared Q: restore the shared boundary, prefill only the suffix.
    let mut st = engine.new_state();
    let t = Instant::now();
    let hit = engine.prefix_lookup(&q).ok_or("shared lookup missed")?;
    if hit.pos != a.shared {
        return Err(format!(
            "shared lookup matched {} tokens, expected {}",
            hit.pos, a.shared
        ));
    }
    let _cached = engine.prefix_restore(hit, &mut st)?;
    let computed = q.len() - hit.pos;
    let first_shared = engine.prefill_chunked(&q[hit.pos..], &mut st, a.chunk)?;
    let warm_shared = t.elapsed().as_secs_f64() * 1e3;
    let ids_shared = decode(engine, &mut st, first_shared, a.tokens)?;
    if ids_shared != ids_q {
        return Err("shared-prefix ids differ from the cold path".to_string());
    }

    // 7. Unrelated R: a real miss, then a capture that exercises the bound.
    engine.reset_state()?;
    let mut st = engine.new_state();
    let miss = engine.prefix_lookup(&r).is_none();
    let t = Instant::now();
    let first_r = engine.prefill_chunked(&r, &mut st, a.chunk)?;
    let cold_r = t.elapsed().as_secs_f64() * 1e3;
    // Capture before decoding: the state must be exactly `r`, not `r + ids`.
    let logits_r = engine.logits_host()?;
    let inserted = engine.prefix_capture(&r, r.len(), Some(&logits_r))?;
    let ids_r = decode(engine, &mut st, first_r, a.tokens)?;
    let final_stats = engine.prefix_stats().ok_or("cache disabled")?;
    let after_evict = engine.prefix_lookup(&p);

    if ids_p != ids_exact {
        return Err("exact-repeat ids differ from the cold path".to_string());
    }

    println!(
        "\n{:<14} {:>6} {:>8} {:>8} {:>10} {:>9} {:>10} {:>7}",
        "case", "cache", "matched", "computed", "TTFT ms", "vs cold", "prefill%", "ids"
    );
    let row = |label: &str, kind: &str, matched: usize, computed: usize, ms: f64, cold: f64, ok: bool| {
        println!(
            "{:<14} {:>6} {:>8} {:>8} {:>10.1} {:>8.2}x {:>9.0}% {:>7}",
            label,
            kind,
            matched,
            computed,
            ms,
            cold / ms.max(1e-9),
            100.0 * (1.0 - computed as f64 / a.len as f64),
            if ok { "ok" } else { "DIFF" }
        );
    };
    row("cold P", "miss", 0, a.len, cold_p, cold_p, true);
    row("warm exact P", "hit", p.len(), 0, warm_exact, cold_p, ids_exact == ids_p);
    row("cold Q", "miss", 0, a.len, cold_q, cold_q, true);
    row(
        "warm shared Q",
        "hit",
        a.shared,
        computed,
        warm_shared,
        cold_q,
        ids_shared == ids_q,
    );
    row("cold R", "miss", 0, a.len, cold_r, cold_r, true);

    println!(
        "\nhit latency (lookup + restore + argmax, no compute): {:.2} ms for {} tokens",
        restore_ms, p.len()
    );
    println!(
        "first 20 ids: cold P {:?}\n              warm exact {:?} (equal: {})\n              warm shared Q {:?} (equal to cold: {})",
        &ids_p[..ids_p.len().min(20)],
        &ids_exact[..ids_exact.len().min(20)],
        ids_p == ids_exact,
        &ids_shared[..ids_shared.len().min(20)],
        ids_shared == ids_q
    );
    println!(
        "unrelated ids: {:?} (its own cold path; not compared)",
        &ids_r[..ids_r.len().min(8)]
    );

    println!("\nentries after the seed:");
    for (pos, bytes, hash) in engine.prefix_entries().unwrap_or_default() {
        println!("  P@{pos:<6} {:.1} MiB  hash {:016x}", bytes as f64 / 1048576.0, hash);
    }
    println!(
        "memory per entry: {} SSM layers x {:.1} MiB + {} KV rows x {} B + logits",
        48,
        156.9 / 48.0,
        16,
        a.len * 1024 * 4
    );
    println!(
        "3-request sequence: exact repeat -> hit @{}; shared prefix -> hit @{}; unrelated -> {} ({} at {:.1} MiB, {} evicted)",
        p.len(),
        a.shared,
        if miss { "miss" } else { "HIT" },
        if inserted.replaced { "replaced" } else { "inserted" },
        final_stats.bytes as f64 / 1048576.0,
        inserted.evicted
    );
    println!(
        "cache bound: {} entries / {:.0} MiB -> {} entries / {:.1} MiB, {} hits, {} misses, {} evictions, {} too large",
        a.entries,
        a.bytes as f64 / 1048576.0,
        final_stats.entries,
        final_stats.bytes as f64 / 1048576.0,
        final_stats.hits,
        final_stats.misses,
        final_stats.evictions,
        final_stats.too_large
    );
    match after_evict {
        Some(h) => println!(
            "post-eviction exact repeat: partial hit at {} of {} tokens (the full entry was the LRU victim)",
            h.pos,
            p.len()
        ),
        None => println!("post-eviction exact repeat: miss (the whole prompt was evicted)"),
    }
    Ok(())
}

pub fn bench(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("cache-bench: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cache-bench: load: {e}");
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
    let n_ctx = a.len + a.tokens + 8;
    let mut engine = match GpuEngine::new(&model, n_ctx, KvMode::F32) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cache-bench: graph: {e}");
            return 1;
        }
    };
    engine.enable_prefix_cache(a.entries, a.bytes);
    // Allocate the chunk scratch and resolve every entry point before timing.
    let _ = engine.chunk_bytes(a.chunk);
    let warm: Vec<u32> = (0..8).map(|i| 100 + i).collect();
    let mut st = engine.new_state();
    if let Err(e) = engine.prefill_chunked(&warm, &mut st, a.chunk.min(8)) {
        eprintln!("cache-bench: warmup: {e}");
        return 1;
    }
    println!("build: {:.1}s", t0.elapsed().as_secs_f64());

    if let Err(e) = main_bench(&mut engine, &a) {
        eprintln!("cache-bench: {e}");
        return 1;
    }
    0
}
