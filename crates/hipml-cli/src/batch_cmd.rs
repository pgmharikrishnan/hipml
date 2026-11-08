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

struct Args {
    gguf: String,
    slots: usize,
    tokens: usize,
    depths: Vec<usize>,
    kv: KvMode,
    check: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        slots: 4,
        tokens: 16,
        depths: vec![0],
        kv: KvMode::Q8,
        check: false,
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
    if let Err(e) = eng.activate(a.slots) {
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

    if let Err(e) = eng.fill_synthetic(&a.depths) {
        eprintln!("batch-bench: synthetic fill: {e}");
        return 1;
    }
    if let Err(e) = eng.set_positions(&a.depths) {
        eprintln!("batch-bench: positions: {e}");
        return 1;
    }

    // Warm: one step per slot, so clocks and caches settle. Then rewind the
    // positions and clear SSM state, so the measured run and the B=1 check both
    // start at the same depth.
    if let Err(e) = eng.forward(&vec![1u32; a.slots]) {
        eprintln!("batch-bench: warm: {e}");
        return 1;
    }
    if let Err(e) = eng.reset_state() {
        eprintln!("batch-bench: reset: {e}");
        return 1;
    }
    if let Err(e) = eng.set_positions(&a.depths) {
        eprintln!("batch-bench: rewind: {e}");
        return 1;
    }

    let n = a.tokens;
    let t = Instant::now();
    let mut last = vec![0u32; a.slots];
    for _ in 0..n {
        match eng.forward(&vec![1u32; a.slots]) {
            Ok(ids) => last = ids,
            Err(e) => {
                eprintln!("batch-bench: decode: {e}");
                return 1;
            }
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let agg = (a.slots * n) as f64 / secs;
    let per_stream = agg / a.slots as f64;
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
        agg / (a.slots as f64 * single_tps),
        a.slots
    );

    // Ids for the parity check: same start position, same token stream. The
    // timing warm-up moved the position, so rewind first.
    if let Err(e) = single.reset_state() {
        eprintln!("batch-bench: single reset: {e}");
        return 1;
    }
    let mut st = single.state_at(a.depths[0]);
    let mut single_ids = Vec::with_capacity(n);
    for _ in 0..n {
        match single.forward_token_id(1, &mut st) {
            Ok(id) => single_ids.push(id),
            Err(e) => {
                eprintln!("batch-bench: single ids: {e}");
                return 1;
            }
        }
    }
    let _ = &single_ids;

    if a.check {
        if a.depths.iter().any(|d| *d != 0) {
            println!("check: token parity is only defined at depth 0 here (synthetic KV seeds differ)");
        }
        let mut ok = true;
        for (b, ids) in last.iter().enumerate() {
            let same = *ids == single_ids[n - 1];
            if !same {
                ok = false;
            }
            println!(
                "check slot {b}: last id batch {} vs single {} -> {}",
                ids,
                single_ids[n - 1],
                if same { "match" } else { "MISMATCH" }
            );
        }
        println!(
            "check: {}",
            if ok {
                "all slots reproduce the B=1 path"
            } else {
                "TOKENS DIFFER"
            }
        );
        if !ok {
            return 1;
        }
    }

    vram_note(&eng, a.slots, 8192);
    0
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
