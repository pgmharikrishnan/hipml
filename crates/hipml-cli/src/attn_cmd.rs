//! `hipml attn-parity` / `attn-bench`: the GPU full-attention block against the
//! CPU reference, and its per-layer timing at real context depths.
//!
//! Only one layer's tensors are read, so this runs without the 17 GB model in
//! host RAM. The first full-attention block is the default layer.

use std::time::Instant;

use hipml_core::attention::{attn_causal_f32, attn_layer_f32, FullAttnW, KvState};
use hipml_core::config::Config;
use hipml_core::gguf::Gguf;
use hipml_core::model::load_attn_layer;
use hipml_core::ops::{head_rmsnorm_f32, rope_multi_f32};
use hipml_core::parity::{compare, Tol};
use hipml_core::rng::Rng;
use hipml_kernels::{AttnLayer, Launcher};

struct Args {
    gguf: String,
    layer: Option<usize>,
    depths: Vec<usize>,
    reps: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut gguf = None;
    let mut layer = None;
    let mut depths = vec![0usize, 8192, 100_000];
    let mut reps = 5usize;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--gguf" => {
                gguf = Some(need(i)?.to_string());
                i += 2;
            }
            "--layer" => {
                layer = Some(need(i)?.parse().map_err(|e| format!("bad --layer: {e}"))?);
                i += 2;
            }
            "--depths" => {
                depths = need(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("bad --depths: {e}"))?;
                i += 2;
            }
            "--reps" => {
                reps = need(i)?.parse().map_err(|e| format!("bad --reps: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        gguf: gguf.ok_or("--gguf PATH is required")?,
        layer,
        depths,
        reps,
    })
}

struct Loaded {
    cfg: Config,
    idx: usize,
    w: FullAttnW,
}

fn load(a: &Args) -> Result<Loaded, String> {
    let mut g = Gguf::open(&a.gguf).map_err(|e| format!("cannot open {}: {e}", a.gguf))?;
    let cfg = Config::from_gguf(&g).map_err(|e| format!("config: {e}"))?;
    let idx = a
        .layer
        .or_else(|| (0..cfg.n_layer).find(|&i| !cfg.is_recr[i]))
        .ok_or("model has no full-attention layer")?;
    if cfg.is_recr[idx] {
        return Err(format!("layer {idx} is a GDN layer, not full attention"));
    }
    let w = load_attn_layer(&mut g, idx).map_err(|e| format!("layer {idx}: {e}"))?;
    Ok(Loaded { cfg, idx, w })
}

pub fn parity(args: &[String]) -> i32 {
    run(args, true)
}

pub fn bench(args: &[String]) -> i32 {
    run(args, false)
}

fn run(args: &[String], parity: bool) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("attn: {e}");
            return 2;
        }
    };
    let l = match load(&a) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("attn: {e}");
            return 1;
        }
    };
    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("attn: no device: {e}");
            return 1;
        }
    };
    println!("device: {}", launcher.device_name);
    println!(
        "layer {}: {} heads, {} kv heads, head_dim {}, rope {} sections {:?} base {}",
        l.idx,
        l.cfg.n_head,
        l.cfg.n_head_kv,
        l.cfg.head_dim,
        l.cfg.n_rot,
        l.cfg.rope_sections,
        l.cfg.rope_base
    );
    println!(
        "weights: q {:?} {}x{}, k {:?} {}x{}, v {:?} {}x{}, o {:?} {}x{}",
        l.w.wq.kind,
        l.w.wq.rows,
        l.w.wq.cols,
        l.w.wk.kind,
        l.w.wk.rows,
        l.w.wk.cols,
        l.w.wv.kind,
        l.w.wv.rows,
        l.w.wv.cols,
        l.w.wo.kind,
        l.w.wo.rows,
        l.w.wo.cols
    );

    if parity {
        parity_run(&launcher, &l)
    } else {
        bench_run(&launcher, &l, &a)
    }
}

fn parity_run(l: &Launcher, m: &Loaded) -> i32 {
    let cfg = &m.cfg;
    let hd = cfg.head_dim;
    let width = cfg.n_head_kv * hd;
    let tol = Tol::new(1e-4, 1e-3);
    println!("\n{:<30} {:>10} {:>10}  {}", "case", "max_abs", "max_rel", "verdict");

    let mut failed = false;

    // 1. q/k RMS norm + M-RoPE, no attention involved.
    let mut rng = Rng::new(0xF011_A77E_1234_5678);
    let q_raw: Vec<f32> = (0..cfg.n_head * hd).map(|_| rng.normal()).collect();
    let k_raw: Vec<f32> = (0..width).map(|_| rng.normal()).collect();
    let pos = 37usize;
    let p = [pos; 4];
    let cpu_q = rope_multi_f32(
        &head_rmsnorm_f32(&q_raw, &m.w.q_norm, cfg.n_head, hd, cfg.eps),
        cfg.n_head,
        hd,
        cfg.n_rot,
        cfg.rope_sections,
        p,
        cfg.rope_base,
    );
    let cpu_k = rope_multi_f32(
        &head_rmsnorm_f32(&k_raw, &m.w.k_norm, cfg.n_head_kv, hd, cfg.eps),
        cfg.n_head_kv,
        hd,
        cfg.n_rot,
        cfg.rope_sections,
        p,
        cfg.rope_base,
    );
    let layer = match AttnLayer::new(l, &m.w, cfg, 1) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("attn: layer alloc: {e}");
            return 1;
        }
    };
    match layer.norm_rope(l, &q_raw, &k_raw, pos) {
        Err(e) => {
            eprintln!("attn: norm_rope launch: {e}");
            return 1;
        }
        Ok((gq, gk)) => {
            let sq = compare(&cpu_q, &gq, tol);
            let sk = compare(&cpu_k, &gk, tol);
            let (q_ok, qs) = verdict(sq);
            let (k_ok, ks) = verdict(sk);
            failed |= !q_ok || !k_ok;
            println!(
                "{:<30} {:>10.2e} {:>10.2e}  {}",
                "q norm+rope",
                qs.max_abs,
                qs.max_rel,
                if q_ok { "PASS" } else { "FAIL" }
            );
            println!(
                "{:<30} {:>10.2e} {:>10.2e}  {}",
                "k norm+rope",
                ks.max_abs,
                ks.max_rel,
                if k_ok { "PASS" } else { "FAIL" }
            );
        }
    }

    // 2. One layer at depth 1 (empty cache), then 3. depth 1000.
    let mut rng = Rng::new(0xA77E_0001_0000_0001);
    let x0: Vec<f32> = (0..cfg.n_embd).map(|_| rng.normal_scaled(0.5)).collect();
    let mut kv = KvState::new(1, width);
    let cpu_y = attn_layer_f32(&m.w, &mut kv, &x0, cfg, 0);
    let layer = match AttnLayer::new(l, &m.w, cfg, 1) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("attn: layer alloc: {e}");
            return 1;
        }
    };
    if layer.kv_bytes() > 0 && layer.zero_kv().is_err() {
        eprintln!("attn: kv zero failed");
        return 1;
    }
    if let Err(e) = layer.forward(l, &x0, 0) {
        eprintln!("attn: depth 1 launch: {e}");
        return 1;
    }
    match layer.forward(l, &x0, 0) {
        Err(e) => {
            eprintln!("attn: depth 1 download: {e}");
            return 1;
        }
        Ok(gpu_y) => {
            let s = compare(&cpu_y, &gpu_y, tol);
            let (ok, st) = verdict(s);
            failed |= !ok;
            println!(
                "{:<30} {:>10.2e} {:>10.2e}  {}",
                "layer depth 1",
                st.max_abs,
                st.max_rel,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    // 2b. The chunked-prefill attention kernel alone: 37 queries at position
    // 29 over a 66-row cache, compared row by row with the CPU reference.
    {
        const TOK: usize = 37;
        const POS0: usize = 29;
        let seq = POS0 + TOK;
        let mut rng = Rng::new(0xC4A0_1234_5678);
        let qc: Vec<f32> = (0..TOK * cfg.n_head * hd).map(|_| rng.normal()).collect();
        let kc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();
        let vc: Vec<f32> = (0..seq * width).map(|_| rng.normal()).collect();
        let qb = hipml_hip::Buffer::from_host(&qc).unwrap();
        let kb = hipml_hip::Buffer::from_host(&kc).unwrap();
        let vb = hipml_hip::Buffer::from_host(&vc).unwrap();
        let ob = hipml_hip::Buffer::alloc(TOK * cfg.n_head * hd).unwrap();
        let layer = match AttnLayer::new(l, &m.w, cfg, seq + 8) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("attn: chunk layer alloc: {e}");
                return 1;
            }
        };
        if let Err(e) = layer.chunk_attention(l, &qb, &kb, &vb, &ob, POS0, TOK) {
            eprintln!("attn: chunk attention launch: {e}");
            return 1;
        }
        match ob.to_host() {
            Ok(gpu) => {
                let scale = 1.0f32 / (hd as f32).sqrt();
                let mut cpu = vec![0.0f32; TOK * cfg.n_head * hd];
                for t in 0..TOK {
                    let row = attn_causal_f32(
                        &qc[t * cfg.n_head * hd..(t + 1) * cfg.n_head * hd],
                        &kc,
                        &vc,
                        cfg.n_head,
                        cfg.n_head_kv,
                        hd,
                        POS0 + t + 1,
                        scale,
                    );
                    cpu[t * cfg.n_head * hd..(t + 1) * cfg.n_head * hd].copy_from_slice(&row);
                }
                let s = compare(&cpu, &gpu, Tol::new(2e-3, 2e-2));
                let (ok, st) = verdict(s);
                failed |= !ok;
                println!(
                    "{:<30} {:>10.2e} {:>10.2e}  {}",
                    "chunk attention 37@29",
                    st.max_abs,
                    st.max_rel,
                    if ok { "PASS" } else { "FAIL" }
                );
                if !ok {
                    // Where the error lives: d half, head, row, and the two
                    // magnitudes, so a layout regression is one line away.
                    let nh = cfg.n_head * hd;
                    let (mut worst, mut at) = (0.0f32, (0usize, 0usize, 0usize));
                    let (mut half0, mut half1) = (0.0f32, 0.0f32);
                    for t in 0..TOK {
                        for hh in 0..cfg.n_head {
                            for dd in 0..hd {
                                let i = t * nh + hh * hd + dd;
                                let e = (cpu[i] - gpu[i]).abs();
                                if dd < hd / 2 {
                                    half0 = half0.max(e);
                                } else {
                                    half1 = half1.max(e);
                                }
                                if e > worst {
                                    worst = e;
                                    at = (t, hh, dd);
                                }
                            }
                        }
                    }
                    println!(
                        "  diag: max|d| half0 {half0:.3e} half1 {half1:.3e}; worst at row {} head {} d {} \
                         (cpu {:.4} gpu {:.4})",
                        at.0, at.1, at.2,
                        cpu[at.0 * nh + at.1 * hd + at.2],
                        gpu[at.0 * nh + at.1 * hd + at.2]
                    );
                }
            }
            Err(e) => {
                eprintln!("attn: chunk attention download: {e}");
                return 1;
            }
        }
    }

    const DEPTH: usize = 1000;
    let mut rng = Rng::new(0xA77E_0002_0000_0002);
    let xs: Vec<Vec<f32>> = (0..DEPTH)
        .map(|_| (0..cfg.n_embd).map(|_| rng.normal_scaled(0.5)).collect())
        .collect();
    let mut kv = KvState::new(DEPTH, width);
    let mut cpu_y = Vec::new();
    for (pos, x) in xs.iter().enumerate() {
        cpu_y = attn_layer_f32(&m.w, &mut kv, x, cfg, pos);
    }
    let layer = match AttnLayer::new(l, &m.w, cfg, DEPTH) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("attn: layer alloc: {e}");
            return 1;
        }
    };
    if layer.zero_kv().is_err() {
        eprintln!("attn: kv zero failed");
        return 1;
    }
    let mut gpu_y = Vec::new();
    for (pos, x) in xs.iter().enumerate() {
        match layer.forward(l, x, pos) {
            Ok(y) => gpu_y = y,
            Err(e) => {
                eprintln!("attn: depth {DEPTH} launch: {e}");
                return 1;
            }
        }
    }
    let s = compare(&cpu_y, &gpu_y, tol);
    let (ok, st) = verdict(s);
    failed |= !ok;
    println!(
        "{:<30} {:>10.2e} {:>10.2e}  {}",
        "layer depth 1000",
        st.max_abs,
        st.max_rel,
        if ok { "PASS" } else { "FAIL" }
    );
    println!("\ntolerance: abs {:.0e} or rel {:.0e}", tol.abs, tol.rel);
    if failed {
        1
    } else {
        0
    }
}

fn verdict(r: Result<hipml_core::Stats, hipml_core::Stats>) -> (bool, hipml_core::Stats) {
    match r {
        Ok(s) => (true, s),
        Err(s) => (false, s),
    }
}

fn bench_run(l: &Launcher, m: &Loaded, a: &Args) -> i32 {
    let cfg = &m.cfg;
    let width = cfg.n_head_kv * cfg.head_dim;
    let mut rng = Rng::new(0xB3C4_0000_0000_0001);
    let x: Vec<f32> = (0..cfg.n_embd).map(|_| rng.normal_scaled(0.5)).collect();
    let xb = match hipml_hip::Buffer::from_host(&x) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("attn: input upload: {e}");
            return 1;
        }
    };

    println!("\n{:<10} {:>12} {:>12} {:>12} {:>10}", "depth", "seq", "layer_ms", "weight_GB/s", "16x_ms");
    let mut per_layer = Vec::new();
    for &depth in &a.depths {
        let n_ctx = depth + 1;
        let layer = match AttnLayer::new(l, &m.w, cfg, n_ctx) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("attn: depth {depth}: {e}");
                return 1;
            }
        };
        if let Err(e) = layer.zero_kv() {
            eprintln!("attn: depth {depth} zero: {e}");
            return 1;
        }
        // Warm-up launch, then the timed series; one sync for the whole run.
        if let Err(e) = layer.forward_buf(l, &xb, depth) {
            eprintln!("attn: depth {depth} launch: {e}");
            return 1;
        }
        if l.synchronize().is_err() {
            return 1;
        }
        let t0 = Instant::now();
        for _ in 0..a.reps {
            if let Err(e) = layer.forward_buf(l, &xb, depth) {
                eprintln!("attn: depth {depth} launch: {e}");
                return 1;
            }
        }
        if l.synchronize().is_err() {
            return 1;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / a.reps as f64;
        let gbs = layer.weight_bytes() as f64 / (ms * 1e-3) / 1e9;
        per_layer.push(ms);
        println!(
            "{:<10} {:>12} {:>12.3} {:>12.1} {:>10.2}",
            depth,
            depth + 1,
            ms,
            gbs,
            ms * 16.0
        );
        if depth == 100_000 {
            let kv = layer.kv_bytes();
            println!(
                "  KV cache: {:.1} MB/layer, 16 layers {:.2} GB ({:.1} KiB/token over 16 layers)",
                kv as f64 / 1e6,
                kv as f64 * 16.0 / 1e9,
                kv as f64 * 16.0 / (depth as f64) / 1024.0
            );
        }
    }
    println!(
        "\nweights streamed per layer: {:.1} MB; 16 layers {:.1} MB",
        m.w.wq.data.len() as f64 / 1e6 + m.w.wk.data.len() as f64 / 1e6
            + m.w.wv.data.len() as f64 / 1e6 + m.w.wo.data.len() as f64 / 1e6,
        16.0 * (m.w.wq.data.len() + m.w.wk.data.len() + m.w.wv.data.len() + m.w.wo.data.len())
            as f64
            / 1e6
    );
    println!("weight-read reference (whole model decode step): 27.19 ms");
    println!("kv cache width: {} floats/token/layer = {} KiB per token over 16 layers",
        width, 2 * width * 4 * 16 / 1024);
    let _ = per_layer;
    0
}
