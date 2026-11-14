// SPDX-License-Identifier: Apache-2.0
//! `hipml batch-bench`: aggregate decode throughput for B concurrent slots.
//!
//! This is the multi-sequence counterpart of `gpu-bench`. It builds a
//! `BatchEngine` with `--slots` slots, optionally fills each slot's KV cache to
//! a depth (uniform or mixed), and times the batched decode step. `--check`
//! runs the same token stream through the single-stream engine and requires the
//! greedy ids to match slot by slot.

use std::time::Instant;

use hipml_model::gpu::{BatchEngine, GpuEngine, KvMode};
use hipml_model::Model;

#[path = "batch_probe.rs"]
mod probe;

struct Args {
    gguf: String,
    slots: usize,
    active: Option<usize>,
    tokens: usize,
    depths: Vec<usize>,
    kv: KvMode,
    check: bool,
    sweep: bool,
    distinct: bool,
    split: bool,
    probe: bool,
    /// Print the per-step logit agreement between this slot's batched row and
    /// the single-sequence engine instead of the pass/fail id gate.
    diff_slot: Option<usize>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        slots: 4,
        active: None,
        tokens: 16,
        depths: vec![0],
        kv: KvMode::Q8,
        check: false,
        sweep: false,
        distinct: false,
        split: false,
        probe: false,
        diff_slot: None,
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
            "--slots" => {
                a.slots = need(i)?.parse().map_err(|e| format!("bad --slots: {e}"))?;
                i += 2;
            }
            "--active" => {
                a.active = Some(need(i)?.parse().map_err(|e| format!("bad --active: {e}"))?);
                i += 2;
            }
            "--tokens" => {
                a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--depth" => {
                a.depths = vec![need(i)?.parse().map_err(|e| format!("bad --depth: {e}"))?];
                i += 2;
            }
            "--depths" => {
                a.depths = need(i)?
                    .split(',')
                    .map(|s| s.parse::<usize>().map_err(|e| format!("bad --depths: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--check" => {
                a.check = true;
                i += 1;
            }
            "--sweep" => {
                a.sweep = true;
                i += 1;
            }
            "--distinct" => {
                a.distinct = true;
                i += 1;
            }
            "--split" => {
                a.split = true;
                i += 1;
            }
            "--probe" => {
                a.probe = true;
                i += 1;
            }
            "--diff-slot" => {
                a.diff_slot = Some(
                    need(i)?
                        .parse()
                        .map_err(|e| format!("bad --diff-slot: {e}"))?,
                );
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.depths.len() == 1 {
        a.depths = vec![a.depths[0]; a.slots];
    }
    if a.depths.len() != a.slots {
        return Err(format!("{} depths for {} slots", a.depths.len(), a.slots));
    }
    Ok(a)
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("batch-bench: {e}");
            return 2;
        }
    };
    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("batch-bench: load: {e}");
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
    let max_depth = *a.depths.iter().max().unwrap();
    let n_ctx = max_depth + a.tokens + 8;
    let mut eng = match BatchEngine::new(&model, n_ctx, a.kv, a.slots) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("batch-bench: engine: {e}");
            return 1;
        }
    };
    let active = a.active.unwrap_or(a.slots);
    if let Err(e) = eng.activate(active) {
        eprintln!("batch-bench: activate: {e}");
        return 1;
    }
    let per_slot = (eng.kv_bytes() + eng.ssm_state_bytes()) / a.slots;
    println!(
        "batch: slots={} kv={} n_ctx={} | {:.2} GiB streamed/step | per-slot state {:.1} MiB (KV {:.1} + SSM {:.1})",
        a.slots,
        eng.kv_mode.name(),
        n_ctx,
        eng.weight_bytes() as f64 / (1u64 << 30) as f64,
        per_slot as f64 / (1024.0 * 1024.0),
        (eng.kv_bytes() / a.slots) as f64 / (1024.0 * 1024.0),
        (eng.ssm_state_bytes() / a.slots) as f64 / (1024.0 * 1024.0),
    );

    if a.probe {
        probe::probe_kernels(&model, a.slots);
    }

    if let Err(e) = eng.fill_synthetic(&a.depths[..active]) {
        eprintln!("batch-bench: synthetic fill: {e}");
        return 1;
    }
    if let Err(e) = eng.set_positions(&a.depths[..active]) {
        eprintln!("batch-bench: positions: {e}");
        return 1;
    }

    // Warm: one step per slot, so clocks and caches settle. Then rewind the
    // positions and clear SSM state, so the measured run and the B=1 check both
    // start at the same depth.
    if let Err(e) = eng.forward(&vec![1u32; active]) {
        eprintln!("batch-bench: warm: {e}");
        return 1;
    }
    if let Err(e) = eng.reset_state() {
        eprintln!("batch-bench: reset: {e}");
        return 1;
    }
    if let Err(e) = eng.set_positions(&a.depths[..active]) {
        eprintln!("batch-bench: rewind: {e}");
        return 1;
    }

    let n = a.tokens;
    eng.trace = a.split;
    // `--distinct` feeds each slot its own token stream, so the routed FFN
    // sees the routing pattern a real fleet produces instead of one expert set
    // shared by every slot (which is the best case for a cache, and the reason
    // the same-token default is not the number to price a gather against).
    let vocab = eng.cfg.n_vocab as u32;
    let stream = |b: usize, step: usize| (7 + b as u32 * 97 + step as u32 * 13) % vocab;
    let t = Instant::now();
    let mut last = vec![0u32; a.slots];
    for step in 0..n {
        let toks: Vec<u32> = if a.distinct {
            (0..active).map(|b| stream(b, step)).collect()
        } else {
            vec![1u32; active]
        };
        match eng.forward(&toks) {
            Ok(ids) => last = ids,
            Err(e) => {
                eprintln!("batch-bench: decode: {e}");
                return 1;
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let agg = (active * n) as f64 / secs;
    let per_stream = agg / active as f64;
    let step_ms = secs * 1e3 / n as f64;
    let gbs = eng.weight_bytes() as f64 * n as f64 / secs / 1e9;
    println!(
        "decode {}: {:.2} tok/s aggregate | {:.2} tok/s per stream | {:.2} ms/step | {:.0} GB/s weight | ids {:?}",
        depth_label(&a.depths),
        agg,
        per_stream,
        step_ms,
        gbs,
        &last[..last.len().min(4)]
    );
    if a.split {
        const NAMES: [&str; 9] = [
            "embed",
            "attn proj",
            "attn kv+core",
            "attn out",
            "ssm proj",
            "ssm rec",
            "ssm out",
            "ffn",
            "final",
        ];
        for (i, name) in NAMES.iter().enumerate() {
            println!(
                "  {name:<13} {:8.2} ms/step ({:5.1}%)",
                eng.stages[i] * 1e3 / n as f64,
                100.0 * eng.stages[i] / secs
            );
        }
    }

    // Single-stream reference on the same engine shape, so the efficiency is
    // measured on this run rather than copied from a doc.
    let mut single = match GpuEngine::new(&model, a.depths[0] + n + 8, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("batch-bench: single engine: {e}");
            return 1;
        }
    };
    if a.depths.iter().all(|d| *d == a.depths[0]) {
        if let Err(e) = single.fill_synthetic(a.depths[0]) {
            eprintln!("batch-bench: single fill: {e}");
            return 1;
        }
    }
    let mut st = single.state_at(a.depths[0]);
    for _ in 0..2 {
        if let Err(e) = single.forward_token_id(1, &mut st) {
            eprintln!("batch-bench: single warm: {e}");
            return 1;
        }
    }
    let t = Instant::now();
    for _ in 0..n {
        if let Err(e) = single.forward_token_id(1, &mut st) {
            eprintln!("batch-bench: single: {e}");
            return 1;
        }
    }
    let ssecs = t.elapsed().as_secs_f64();
    let single_tps = n as f64 / ssecs;

    println!(
        "single stream at depth {}: {:.2} tok/s ({:.2} ms/step) | batch efficiency {:.2}x of {}x",
        a.depths[0],
        single_tps,
        ssecs * 1e3 / n as f64,
        agg / (active as f64 * single_tps),
        active
    );

    if a.check {
        if let Some(slot) = a.diff_slot {
            if let Err(e) = diff_logits(&mut eng, &mut single, n, active, slot) {
                eprintln!("batch-bench: diff: {e}");
                return 1;
            }
            return 0;
        }
        if a.depths.iter().any(|d| *d != 0) {
            println!(
                "check: token parity is only defined at depth 0 here (synthetic KV seeds differ)"
            );
        } else {
            let ok = match check_slots(&mut eng, &mut single, n, active) {
                Ok(ok) => ok,
                Err(e) => {
                    eprintln!("batch-bench: check: {e}");
                    return 1;
                }
            };
            println!(
                "check: {}",
                if ok {
                    "all slots reproduce the B=1 path over the full stream"
                } else {
                    "TOKENS DIFFER"
                }
            );
            if !ok {
                return 1;
            }
            // The gate the dense path has always had, at every slot count: a
            // slot must produce the ids it would produce alone. `--sweep` builds
            // a fresh engine at B = 1..=slots and replays each one, so a bug
            // that only appears at some batch width cannot hide behind the
            // capacity the benchmark happened to run at.
            if a.sweep {
                for b in 1..active {
                    let mut e = match BatchEngine::new(&model, n_ctx, a.kv, b) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("batch-bench: sweep engine at B={b}: {e}");
                            return 1;
                        }
                    };
                    if let Err(e) = e.activate(b) {
                        eprintln!("batch-bench: sweep activate at B={b}: {e}");
                        return 1;
                    }
                    let ok_b = match check_slots(&mut e, &mut single, n, b) {
                        Ok(ok) => ok,
                        Err(e) => {
                            eprintln!("batch-bench: sweep check at B={b}: {e}");
                            return 1;
                        }
                    };
                    println!(
                        "sweep B={b}: {}",
                        if ok_b {
                            "all slots reproduce the B=1 path"
                        } else {
                            "TOKENS DIFFER"
                        }
                    );
                    if !ok_b {
                        return 1;
                    }
                }
                println!("sweep: B=1..={active} all reproduce the B=1 path");
            }
        }
    }

    vram_note(&eng, a.slots, 8192);
    0
}

/// Replay a token stream that differs per slot and per step through `eng`, then
/// run each slot's stream through the single-sequence engine and compare ids.
///
/// Distinct streams catch state bleeding between slots, which a shared token
/// would hide (all slots would then be identical by construction). Returns
/// `Ok(true)` when every slot reproduces the B=1 ids over the whole stream.
fn check_slots(
    eng: &mut BatchEngine<'_>,
    single: &mut GpuEngine<'_>,
    n: usize,
    active: usize,
) -> Result<bool, String> {
    let vocab = eng.cfg.n_vocab as u32;
    let stream = |b: usize, step: usize| (7 + b as u32 * 97 + step as u32 * 13) % vocab;
    let mut batch_ids = vec![Vec::with_capacity(n); active];
    eng.reset_state()?;
    eng.set_positions(&vec![0; active])?;
    for step in 0..n {
        let toks: Vec<u32> = (0..active).map(|b| stream(b, step)).collect();
        let ids = eng.forward(&toks)?;
        for (b, id) in ids.into_iter().enumerate() {
            batch_ids[b].push(id);
        }
    }
    let mut ok = true;
    for b in 0..active {
        single.reset_state()?;
        let mut st = single.state_at(0);
        let mut ok_b = true;
        for (step, want) in batch_ids[b].iter().enumerate() {
            let id = single.forward_token_id(stream(b, step), &mut st)?;
            if id != *want {
                ok = false;
                ok_b = false;
                println!("check slot {b} step {step}: batch {want} vs single {id} -> MISMATCH");
                break;
            }
        }
        if ok_b {
            println!("check slot {b}: {n}/{n} ids match the B=1 path");
        }
    }
    Ok(ok)
}

/// The same replay as [`check_slots`], but printing the full logit vector's
/// agreement per step for one slot: how far apart the batch and the single
/// stream are, and how much room the argmax had. Diagnostic only.
fn diff_logits(
    eng: &mut BatchEngine<'_>,
    single: &mut GpuEngine<'_>,
    n: usize,
    active: usize,
    slot: usize,
) -> Result<(), String> {
    let vocab = eng.cfg.n_vocab as usize;
    let stream = |b: usize, step: usize| (7 + b as u32 * 97 + step as u32 * 13) % vocab as u32;
    let mut batch: Vec<Vec<f32>> = Vec::with_capacity(n);
    eng.reset_state()?;
    eng.set_positions(&vec![0; active])?;
    for step in 0..n {
        let toks: Vec<u32> = (0..active).map(|b| stream(b, step)).collect();
        eng.forward(&toks)?;
        let lg = eng.logits_host()?;
        batch.push(lg[slot * vocab..(slot + 1) * vocab].to_vec());
    }
    let top = |v: &[f32]| -> (usize, f32, f32) {
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut second = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > best.1 {
                second = best.1;
                best = (i, x);
            } else if x > second {
                second = x;
            }
        }
        (best.0, best.1, second)
    };
    single.reset_state()?;
    let mut st = single.state_at(0);
    for (step, a) in batch.iter().enumerate() {
        let lg = single.forward_token(stream(slot, step), &mut st)?;
        let (mut diff, mut worst) = (0usize, 0.0f32);
        for i in 0..vocab {
            if a[i].to_bits() != lg[i].to_bits() {
                diff += 1;
                worst = worst.max((a[i] - lg[i]).abs());
            }
        }
        let (ba, bv, b2) = top(a);
        let (sa, sv, s2) = top(&lg);
        println!(
            "step {step:3}: batch {ba} ({bv:.6}, margin {:.2e}) | single {sa} ({sv:.6}, \
             margin {:.2e}) | {diff} of {vocab} logits differ, max |d| {worst:.3e}",
            bv - b2,
            sv - s2
        );
    }
    Ok(())
}

fn depth_label(d: &[usize]) -> String {
    if d.iter().all(|x| *x == d[0]) {
        format!("depth {}", d[0])
    } else {
        format!("mixed {:?}", d)
    }
}

/// Per-slot VRAM and how many slots fit at 8K, from the allocation that is
/// live now (weights + this run's slots).
fn vram_note(eng: &BatchEngine<'_>, slots: usize, ctx: usize) {
    let cfg = &eng.cfg;
    let kv_row = match eng.kv_mode {
        KvMode::Q8 => hipml_core::kv::KvDtype::Q8.row_bytes(cfg.head_dim),
        KvMode::Q4 => hipml_core::kv::KvDtype::Q4.row_bytes(cfg.head_dim),
        KvMode::F32 => cfg.head_dim * 4,
    };
    let n_attn = (0..cfg.n_layer).filter(|i| !cfg.is_recr[*i]).count();
    let kv_8k = n_attn * 2 * cfg.n_head_kv * ctx * kv_row;
    let ssm = eng.ssm_state_bytes() / slots;
    let per = kv_8k + ssm;
    println!(
        "vram per slot at {ctx}: {:.1} MiB (KV {:.1} + SSM {:.1})",
        per as f64 / (1024.0 * 1024.0),
        kv_8k as f64 / (1024.0 * 1024.0),
        ssm as f64 / (1024.0 * 1024.0)
    );
    #[cfg(feature = "hip")]
    if let Ok(d) = hipml_hip::Device::first() {
        if let Ok((free, total)) = d.mem_info() {
            let extra = free / per.max(1);
            let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "free now {:.2} GiB of {:.2}: ~{} more slots at {ctx}, {} total",
                gib(free),
                gib(total),
                extra,
                slots + extra
            );
        }
    }
    #[cfg(not(feature = "hip"))]
    let _ = (cfg, ctx);
}
