// SPDX-License-Identifier: Apache-2.0
//! `hipml gpu-profile`: where a decode token's non-weight time goes.
//!
//! Four measurements, all on the resident graph's own buffers:
//!
//! 1. **launch cost** — the same elementwise kernel launched `n` times, once
//!    with a fresh `module.kernel()` lookup per launch and once with the handle
//!    hoisted. The difference is the driver lookup the graph pays per op.
//! 2. **embedding row** — `Model::embedding_row`, the graph's one host
//!    round-trip on the short path (dequant launch + device sync + row copy).
//! 3. **logits readback** — copying `n_vocab` f32 back and running the host
//!    argmax, which is what the graph pays once per token.
//! 4. **graph launches** — how many launches one token issues, counted from the
//!    layer plan rather than instrumented, so it cannot drift from the code.

use std::ffi::c_void;
use std::time::Instant;

use hipml_core::config::Config;
use hipml_core::gen::argmax_f32;
use hipml_core::gguf::Gguf;
use hipml_hip::{Buffer, LaunchConfig};
use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

/// Launches one token issues, from the graph's structure.
///
/// Short (f32) path: 1 embedding dequant, then per layer
/// `rmsnorm + block + residual + rmsnorm + gate/up/silu/down/residual`, then
/// `output_norm + lm_head`.
pub fn launches_per_token(n_layer: usize, is_recr: &[bool]) -> usize {
    let mut n = 1; // embedding dequant
    for &recr in is_recr.iter().take(n_layer) {
        n += 1; // attn_norm rmsnorm
        n += if recr {
            // qkv/gate/alpha/beta (one job-list launch), gates, conv, silu,
            // 2x l2norm, gdn, gate_norm, out
            9
        } else {
            // q/k/v (one job-list launch), qg_split, 2x head norm, 2x rope,
            // kv_append, split, reduce, q_gate, wo
            11
        };
        // residual, rmsnorm, gate+up (one job-list launch), silu, down,
        // residual
        n += 6;
    }
    n + 2 // output rmsnorm, lm_head
}

pub fn profile(args: &[String]) -> i32 {
    let mut gguf = String::new();
    let mut reps = 2000usize;
    let mut trace = false;
    let mut depth = 0usize;
    let mut kv = KvMode::F32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--trace" => {
                trace = true;
                i += 1;
            }
            "--gguf" => {
                gguf = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--reps" => {
                reps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2000);
                i += 2;
            }
            "--depth" => {
                depth = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--kv" => {
                kv = match args.get(i + 1).map(|s| KvMode::parse(s)) {
                    Some(Ok(m)) => m,
                    _ => {
                        eprintln!("gpu-profile: --kv needs f32, q8 or q4");
                        return 2;
                    }
                };
                i += 2;
            }
            other => {
                eprintln!("gpu-profile: unknown argument {other}");
                return 2;
            }
        }
    }
    if depth > 0 && kv == KvMode::F32 {
        eprintln!("gpu-profile: --depth needs a quantized cache (--kv q8 or q4)");
        return 2;
    }

    let (model, _report) = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gpu-profile: load: {e}");
            return 1;
        }
    };
    let cfg = match Gguf::open(&gguf).and_then(|mut g| Config::from_gguf(&mut g)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gpu-profile: config: {e}");
            return 1;
        }
    };
    let l = model.launcher();
    let n = cfg.n_embd;

    // 1. Launch cost, with and without the per-launch handle lookup.
    let a = Buffer::alloc(n).unwrap();
    let b = Buffer::alloc(n).unwrap();
    let y = Buffer::alloc(n).unwrap();
    let per_launch = match launch_cost(&l, &a, &b, &y, n, reps) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("gpu-profile: launch cost: {e}");
            return 1;
        }
    };

    // 2. Embedding row: dequant launch + sync + copy, exactly as the graph does.
    let t = Instant::now();
    let mut sink = 0.0f32;
    for k in 0..64u32 {
        match model.embedding_row(100 + k) {
            Ok(v) => sink += v[0],
            Err(e) => {
                eprintln!("gpu-profile: embedding: {e}");
                return 1;
            }
        }
    }
    let embed_ms = t.elapsed().as_secs_f64() * 1e3 / 64.0;

    // 3. Logits readback and host argmax.
    let logits = Buffer::alloc(cfg.n_vocab).unwrap();
    let _ = logits.write(&vec![0.25f32; cfg.n_vocab]);
    let t = Instant::now();
    let mut id = 0usize;
    for _ in 0..32 {
        match logits.to_host() {
            Ok(v) => id += argmax_f32(&v),
            Err(e) => {
                eprintln!("gpu-profile: readback: {e}");
                return 1;
            }
        }
    }
    let readback_ms = t.elapsed().as_secs_f64() * 1e3 / 32.0;

    let n_launch = launches_per_token(cfg.n_layer, &cfg.is_recr);
    let _ = sink;
    let _ = id;
    println!("device:            {}", model.device_name());
    println!("launch (lookup):   {:.3} us", per_launch.0);
    println!("launch (cached):   {:.3} us", per_launch.1);
    println!("launches/token:    {n_launch}");
    println!(
        "  => launch floor: {:.3} ms (lookup) / {:.3} ms (cached)",
        n_launch as f64 * per_launch.0 / 1e3,
        n_launch as f64 * per_launch.1 / 1e3
    );
    // 1b. The 32-byte step upload the position-aware kernels read. The graph
    // issues one per token; the verify's per-column loops issue several per
    // layer, so its cost matters there.
    let t = Instant::now();
    for i in 0..512 {
        if let Err(e) = l.set_step(i, i + 1, 0) {
            eprintln!("gpu-profile: set_step: {e}");
            return 1;
        }
    }
    if let Err(e) = l.synchronize() {
        eprintln!("gpu-profile: set_step sync: {e}");
        return 1;
    }
    let step_us = t.elapsed().as_secs_f64() * 1e6 / 512.0;
    println!("step upload:       {step_us:.3} us (async, 512 queued)");

    println!("embedding row:     {embed_ms:.3} ms");
    println!(
        "logits readback:   {readback_ms:.3} ms ({} values)",
        cfg.n_vocab
    );

    match stream_weights(&model) {
        Ok((secs, bytes, count)) => println!(
            "weight stream:     {:.3} ms for {:.2} GB in {count} GEMVs ({:.1} GB/s, one launch each)",
            secs * 1e3,
            bytes as f64 / 1e9,
            bytes as f64 / secs / 1e9
        ),
        Err(e) => println!("weight stream:     failed ({e})"),
    }

    // The graph itself, so the profile is against the same build it measures.
    let n_ctx = depth + 64;
    let mut engine = match GpuEngine::new(&model, n_ctx, kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("gpu-profile: graph: {e}");
            return 1;
        }
    };
    if let Err(e) = engine.fill_synthetic(depth) {
        eprintln!("gpu-profile: synthetic fill: {e}");
        return 1;
    }
    let mut st = engine.state_at(depth);
    let mut t = Instant::now();
    for _ in 0..4 {
        if let Err(e) = engine.forward_token(1, &mut st) {
            eprintln!("gpu-profile: graph: {e}");
            return 1;
        }
    }
    let warm = t.elapsed().as_secs_f64() / 4.0;
    t = Instant::now();
    for _ in 0..16 {
        if let Err(e) = engine.forward_token(1, &mut st) {
            eprintln!("gpu-profile: graph: {e}");
            return 1;
        }
    }
    let full = t.elapsed().as_secs_f64() / 16.0;
    println!(
        "graph token:       {:.3} ms (warm {:.3} ms), weight stream {:.3} ms at 632 GB/s",
        full * 1e3,
        warm * 1e3,
        engine.weight_bytes() as f64 / 632e9 * 1e3
    );
    println!(
        "overhead:          {:.3} ms",
        (full - engine.weight_bytes() as f64 / 632e9) * 1e3
    );

    if trace {
        let mut st = engine.state_at(depth);
        for _ in 0..2 {
            let _ = engine.forward_token_id(1, &mut st);
        }
        hipml_hip::probe::set_enabled(true);
        let n = 4;
        engine.trace = Some(hipml_model::gpu::Trace::default());
        for _ in 0..n {
            if let Err(e) = engine.forward_token_id(1, &mut st) {
                eprintln!("gpu-profile: trace: {e}");
                return 1;
            }
        }
        if let Err(e) = engine.model().launcher().synchronize() {
            eprintln!("gpu-profile: trace sync: {e}");
            return 1;
        }
        let bytes: std::collections::HashMap<&str, usize> = engine
            .stage_bytes()
            .into_iter()
            .map(|(n, b)| (n, b))
            .collect();
        // Weight bytes of the stage that runs the GEMVs of that name.
        let attr = |name: &str| -> usize {
            match name {
                "ffn_gate" => bytes["gate"],
                "ffn_up" => bytes["up"],
                "ffn_down" => bytes["down"],
                "attn_qkv" => bytes["attn"],
                "ssm_proj" => bytes["ssm_proj"],
                "ssm_out" => bytes["ssm_step"],
                "lm_head" => bytes["lm_head"],
                _ => 0,
            }
        };
        let rows: Vec<hipml_hip::probe::StageRow> = match engine.trace.as_mut() {
            Some(tr) => tr.fold().to_vec(),
            None => Vec::new(),
        };
        let total: f64 = rows.iter().map(|r| r.span_us).sum();
        let launches: usize = rows.iter().map(|r| r.launches).sum();
        println!(
            "trace at depth {depth} ({n} tokens, ranked by GPU time; launches per token):\n  \
             {:<12} {:>10} {:>7} {:>7} {}",
            "stage", "ms/token", "share", "launch", "GB/s"
        );
        for r in &rows {
            let b = attr(r.name);
            let ms = r.span_us / 1e3 / n as f64;
            let gbs = if b > 0 && ms > 0.0 {
                format!("{:6.1}", b as f64 / (ms / 1e3) / 1e9)
            } else {
                String::new()
            };
            println!(
                "  {:<12} {:10.3} {:6.1}% {:7} {}",
                r.name,
                ms,
                r.span_us / total * 100.0,
                r.launches / n,
                gbs
            );
        }
        println!(
            "  {:<12} {:10.3} {:6.1}% {:7}",
            "sum",
            total / 1e3 / n as f64,
            100.0,
            launches / n
        );
        if hipml_hip::probe::span_enabled() {
            let krows = hipml_hip::probe::kernel_report();
            let ktotal: f64 = krows.iter().map(|r| r.span_us).sum();
            println!(
                "per-kernel, depth {depth} ({n} tokens, ranked by GPU time):\n  \
                 {:<24} {:>10} {:>7} {:>7} {:>9}",
                "kernel", "ms/token", "share", "launch", "us/launch"
            );
            for r in krows.iter().take(30) {
                let ms = r.span_us / 1e3 / n as f64;
                println!(
                    "  {:<24} {:10.3} {:6.1}% {:7} {:9.2}",
                    r.name,
                    ms,
                    r.span_us / ktotal * 100.0,
                    r.launches / n,
                    r.span_us / r.launches.max(1) as f64
                );
            }
            println!(
                "  {:<24} {:10.3} {:6.1}% {:7}",
                "sum",
                ktotal / 1e3 / n as f64,
                100.0,
                krows.iter().map(|r| r.launches).sum::<usize>() / n
            );
        }
    }
    0
}

/// `(us per launch with the handle hoisted, us per launch with a fresh lookup)`.
fn launch_cost(
    l: &hipml_kernels::Launcher,
    a: &Buffer,
    b: &Buffer,
    y: &Buffer,
    n: usize,
    reps: usize,
) -> Result<(f64, f64), String> {
    let mut p_a = a.as_ptr();
    let mut p_b = b.as_ptr();
    let mut p_y = y.as_ptr();
    let mut v_n = n as i32;
    let mut params: [*mut c_void; 4] = [
        &mut p_a as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_y as *mut _ as *mut c_void,
        &mut v_n as *mut _ as *mut c_void,
    ];
    let cfg = LaunchConfig::linear(n, 256);
    let stream = l.stream();

    // Cached first: the very first launches of a kernel in a freshly loaded
    // module pay code-object residency, and averaging that over a loop makes
    // whichever loop runs first look slower by about 1.8 us a launch.
    let k = l.module().kernel("residual_add_f32")?;
    for _ in 0..64 {
        k.launch(cfg, stream, &mut params)?;
    }
    l.synchronize()?;
    let t = Instant::now();
    for _ in 0..reps {
        k.launch(cfg, stream, &mut params)?;
    }
    l.synchronize()?;
    let cached = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let t = Instant::now();
    for _ in 0..reps {
        let k = l.module().kernel("residual_add_f32")?;
        k.launch(cfg, stream, &mut params)?;
    }
    l.synchronize()?;
    let lookup = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!(
        "  ({} entries resolved, lookup-only loop below)",
        l.module().resolved()
    );

    // Lookup alone, no launch.
    let t = Instant::now();
    for _ in 0..reps {
        let _ = l.module().kernel("residual_add_f32")?;
    }
    let only = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("  (entry lookup alone: {only:.3} us)");
    Ok((cached, lookup))
}

/// Stream every resident quantized tensor through its GEMV once, back to back,
/// with no elementwise kernels between them. This is the graph's weight stream
/// without the per-layer dependencies, so the difference against the token time
/// is what the interleaving costs.
fn stream_weights(model: &Model) -> Result<(f64, usize, usize), String> {
    let l = model.launcher();
    let cfg = Gguf::open(model.path()).and_then(|mut g| Config::from_gguf(&mut g));
    let cfg = cfg.map_err(|e| format!("config: {e}"))?;
    let x = Buffer::alloc(cfg.n_embd)?;
    // One output buffer per distinct width, kept alive so no hipFree lands in
    // the timed loop (hipFree synchronizes).
    let mut outs: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    let mut bytes = 0usize;
    let mut count = 0usize;
    let t0 = Instant::now();
    for t in model.tensors() {
        let entry = match t.ty() {
            hipml_core::quant::TensorType::Q4K => "gemv_q4_k",
            hipml_core::quant::TensorType::Q6K => "gemv_q6_k",
            hipml_core::quant::TensorType::Q8_0 => "gemv_q8_0",
            _ => continue,
        };
        let m = t.rows() as usize;
        let k = t.ne0() as usize;
        if k != cfg.n_embd || m == 0 {
            continue;
        }
        if !outs.contains_key(&m) {
            outs.insert(m, Buffer::alloc(m)?);
        }
        l.run_quant_resident(entry, t.buffer(), &x, &outs[&m], m, k)?;
        bytes += t.nbytes() as usize;
        count += 1;
    }
    l.synchronize()?;
    Ok((t0.elapsed().as_secs_f64(), bytes, count))
}
