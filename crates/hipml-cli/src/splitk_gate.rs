//! `hipml splitk-gate`: the split-K ffn scratch against the single-pass path.
//!
//! The split is chosen **per tile**, so a prompt that ends in a short tail runs
//! the wide ffn shapes through `prefill_*_sk` even when every full chunk stayed
//! single-pass. Allocation used to be sized from the chunk, which left a
//! one-token tail with one value of scratch and failed an 8,193-token prompt
//! with `split scratch has 1 values, needs 17408`.
//!
//! This gate runs the same ids twice on one engine — split enabled, then
//! `HIPML_SPLITK=0` — at every length that can expose the tail: one token
//! either side of each chunk boundary, the 2x/4x boundaries, and every tail
//! 1..=N of a large prompt. Ids must match exactly. Logits are compared with
//! `SPLIT_LOGIT_TOL` on the deterministic synthetic prompt. A `--ids-file`
//! fixture is also supported: its ids must match, but its recurrence can
//! amplify last-bit ffn differences, so that mode uses the wider
//! `NATURAL_LOGIT_TOL` sanity bound.
//!
//! The split sums two k-partials in a fixed but different order from the
//! single-pass chain: deterministic, not bit-identical.
//!
//! Exit 1 on a launch error, an id mismatch, or a logit delta above the bound.

use std::time::Instant;

use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

/// Tight bound for the deterministic synthetic prompt. The two-partial ffn sum
/// moves a logit by ~1e-2 at worst over the full 44-length sweep.
const SPLIT_LOGIT_TOL: f32 = 0.05;

/// Bound for a caller-supplied real-token fixture. Tiny ffn differences feed the
/// gated-delta-net recurrence, which can amplify them across a long prompt (the
/// worst measured case is 9.31 on a row whose peak logit is ~20). The id stream
/// is the assertion there; this bound only catches a gross arithmetic break.
const NATURAL_LOGIT_TOL: f32 = 16.0;

struct Args {
    gguf: String,
    ids_file: Option<String>,
    chunk: usize,
    tails: usize,
    sweep_base: Option<usize>,
    ctx: Option<usize>,
    kv: KvMode,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        ids_file: None,
        chunk: 2048,
        tails: 40,
        sweep_base: None,
        ctx: None,
        kv: KvMode::Q8,
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
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--tails" => {
                a.tails = need(i)?.parse().map_err(|e| format!("bad --tails: {e}"))?;
                i += 2;
            }
            "--sweep-base" => {
                a.sweep_base = Some(
                    need(i)?
                        .parse()
                        .map_err(|e| format!("bad --sweep-base: {e}"))?,
                );
                i += 2;
            }
            "--ctx" => {
                a.ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.chunk == 0 {
        return Err("--chunk must be positive".to_string());
    }
    Ok(a)
}

/// The lengths this gate runs. `sweep_base + 1..=sweep_base + tails` is the
/// tail sweep; the boundaries around `chunk`, `2*chunk` and `4*chunk` cover
/// the exact and one-token-over cases, and `4*chunk+1` crosses the 8,192
/// attention split gate with a one-token tile.
fn lengths(chunk: usize, tails: usize, sweep_base: Option<usize>) -> Vec<usize> {
    let base = sweep_base.unwrap_or(2 * chunk);
    let mut v = vec![
        chunk.saturating_sub(1).max(1),
        chunk,
        chunk + 1,
        2 * chunk + 1,
        4 * chunk + 1,
    ];
    for tail in 1..=tails {
        v.push(base + tail);
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// `len` ids: the fixture cycled, or a deterministic in-vocabulary pattern.
fn prompt(len: usize, fixture: Option<&[u32]>, vocab: u32) -> Vec<u32> {
    match fixture {
        Some(ids) if !ids.is_empty() => (0..len).map(|i| ids[i % ids.len()]).collect(),
        _ => {
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            (0..len)
                .map(|_| {
                    x = x
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((x >> 33) as u32) % vocab.max(1)
                })
                .collect()
        }
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("splitk-gate: {e}");
            return 2;
        }
    };
    let fixture = match &a.ids_file {
        Some(p) => match crate::device_cmd::read_ids_file(p) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("splitk-gate: {e}");
                return 2;
            }
        },
        None => None,
    };

    let (model, _) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("splitk-gate: resident load: {e}");
            return 1;
        }
    };
    let lens = lengths(a.chunk, a.tails, a.sweep_base);
    let longest = *lens.last().unwrap_or(&a.chunk);
    let n_ctx = a.ctx.unwrap_or(longest + 8);
    let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("splitk-gate: graph: {e}");
            return 1;
        }
    };
    let vocab = engine.cfg.n_vocab as u32;
    // A real-token fixture can amplify last-bit ffn differences through the
    // recurrence; the synthetic prompt is chosen so the bound stays tight.
    let tol = if fixture.is_some() {
        NATURAL_LOGIT_TOL
    } else {
        SPLIT_LOGIT_TOL
    };
    println!(
        "splitk-gate: chunk {}, {} lengths ({}..{}), {} KV, {} prompt, logit tol {tol}",
        a.chunk,
        lens.len(),
        lens.first().copied().unwrap_or(0),
        longest,
        a.kv.name(),
        if fixture.is_some() {
            "fixture"
        } else {
            "synthetic"
        }
    );

    let mut failed = 0usize;
    let mut worst = 0.0f32;
    for &len in &lens {
        let ids = prompt(len, fixture.as_deref(), vocab);
        // Split on: the default unless the caller's environment disabled it.
        std::env::remove_var("HIPML_SPLITK");
        if let Err(e) = engine.reset_state() {
            println!("  {len:6}  FAIL  reset: {e}");
            failed += 1;
            continue;
        }
        let mut st = engine.new_state();
        let t = Instant::now();
        let split = engine.prefill_chunked_logits(&ids, &mut st, a.chunk);
        let split_secs = t.elapsed().as_secs_f64();
        let (split_id, split_logits) = match split {
            Ok(v) => v,
            Err(e) => {
                println!("  {len:6}  FAIL  split path: {e}");
                failed += 1;
                continue;
            }
        };

        std::env::set_var("HIPML_SPLITK", "0");
        if let Err(e) = engine.reset_state() {
            println!("  {len:6}  FAIL  reset: {e}");
            failed += 1;
            std::env::remove_var("HIPML_SPLITK");
            continue;
        }
        let mut st = engine.new_state();
        let mono = engine.prefill_chunked_logits(&ids, &mut st, a.chunk);
        std::env::remove_var("HIPML_SPLITK");
        let (mono_id, mono_logits) = match mono {
            Ok(v) => v,
            Err(e) => {
                println!("  {len:6}  FAIL  single-pass path: {e}");
                failed += 1;
                continue;
            }
        };

        let delta = max_abs(&split_logits, &mono_logits);
        worst = worst.max(delta);
        let ok = split_id == mono_id && delta <= tol;
        if !ok {
            failed += 1;
        }
        println!(
            "  {len:6}  {}  id {split_id} = {mono_id}, max|dlogit| {delta:.6}, split {split_secs:.2}s",
            if ok { "ok  " } else { "FAIL" }
        );
    }

    if failed == 0 {
        println!(
            "PASS: {} lengths, ids identical, worst |dlogit| {worst:.6} <= {tol}",
            lens.len()
        );
        0
    } else {
        println!("FAIL: {failed}/{} lengths failed", lens.len());
        1
    }
}
