//! `hipml gpu-sweep`: is the decode stream launch-bound or bandwidth-bound?
//!
//! The decode graph streams ~17 GB of weights through ~500 GEMV launches and
//! lands 6.4 ms above the 632 GB/s roofline. Two explanations fit: a per-launch
//! ramp/drain in the memory pipeline, or a per-byte limit of the GEMV access
//! pattern. This command reads the *same* model bytes twice — once with one
//! launch per tensor (what the graph does), once with a single launch that
//! walks a device-side job list — so the difference is the launch boundary and
//! nothing else. No dequant math runs, so the measurement is memory-side only.
//!
//! ```
//! hipml gpu-sweep --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf
//! hipml gpu-sweep --gguf ... --reps 3      # repeat each variant, report best
//! ```

use std::time::Instant;

use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_model::Model;

use crate::sweep_ab::{event_pass, gemv_pass, pair_pass, report, variant_pass};
use crate::sweep_merge::{merge_pass, split_pass};

pub(crate) struct Weight<'a> {
    pub(crate) buf: &'a ByteBuffer,
    pub(crate) bytes: usize,
}

struct Args {
    gguf: String,
    reps: usize,
    max_blocks: usize,
    gemv: bool,
    pair: bool,
    variant: bool,
    merge: bool,
    split: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        reps: 2,
        max_blocks: 4096,
        gemv: false,
        pair: false,
        variant: false,
        merge: false,
        split: false,
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
            "--reps" => {
                a.reps = need(i)?.parse().map_err(|e| format!("bad --reps: {e}"))?;
                i += 2;
            }
            "--max-blocks" => {
                a.max_blocks = need(i)?.parse().map_err(|e| format!("bad --max-blocks: {e}"))?;
                i += 2;
            }
            "--gemv" => {
                a.gemv = true;
                i += 1;
            }
            "--pair" => {
                a.pair = true;
                i += 1;
            }
            "--variant" => {
                a.variant = true;
                i += 1;
            }
            "--merge" => {
                a.merge = true;
                i += 1;
            }
            "--split" => {
                a.split = true;
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

/// Every tensor the decode graph streams as a GEMV weight: quantized, 2-D, and
/// not the embedding table (the graph reads one row of that, not the matrix).
fn graph_weights(model: &Model) -> Vec<Weight<'_>> {
    let mut out = Vec::new();
    for t in model.tensors() {
        if t.shape().len() != 2 || t.name().ends_with("token_embd.weight") {
            continue;
        }
        match t.ty() {
            TensorType::Q4K | TensorType::Q6K | TensorType::Q8_0 => {}
            _ => continue,
        }
        let bytes = t.nbytes() as usize;
        if bytes < 1 << 20 {
            continue;
        }
        out.push(Weight {
            buf: t.buffer(),
            bytes,
        });
    }
    out
}

pub fn sweep(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gpu-sweep: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let (model, _report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gpu-sweep: load: {e}");
            return 1;
        }
    };
    let w = graph_weights(&model);
    let total: usize = w.iter().map(|t| t.bytes).sum();
    println!(
        "device: {}  resident {} tensors in {:.1}s",
        model.device_name(),
        model.tensors().len(),
        t0.elapsed().as_secs_f64()
    );
    println!(
        "read set: {} quantized matrix tensors, {:.2} GB (embedding excluded)",
        w.len(),
        total as f64 / 1e9
    );

    let l = model.launcher();
    let out = match Buffer::alloc(a.max_blocks.max(4096).max(1 << 16)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("gpu-sweep: alloc: {e}");
            return 1;
        }
    };

    // Descriptor array for the merged launch, uploaded once.
    let mut raw = vec![0u8; w.len() * 16];
    for (i, t) in w.iter().enumerate() {
        raw[i * 16..i * 16 + 8].copy_from_slice(&(t.buf.device_ptr() as u64).to_le_bytes());
        raw[i * 16 + 8..i * 16 + 16].copy_from_slice(&((t.bytes / 16) as u64).to_le_bytes());
    }
    let jobs = match Buffer::alloc(w.len() * 4 + 4) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("gpu-sweep: alloc jobs: {e}");
            return 1;
        }
    };
    if let Err(e) = jobs.upload_bytes(0, &raw) {
        eprintln!("gpu-sweep: upload jobs: {e}");
        return 1;
    }

    // Warm the code object so the first timed pass does not pay residency.
    if let Some(t) = w.iter().max_by_key(|t| t.bytes) {
        if let Err(e) = l.stream_read(t.buf, t.bytes, 1024, &out) {
            eprintln!("gpu-sweep: warm: {e}");
            return 1;
        }
        if let Err(e) = l.synchronize() {
            eprintln!("gpu-sweep: warm sync: {e}");
            return 1;
        }
    }

    // Calibration: one 768 MB allocation read with one launch per grid size.
    // No tensor boundary, no job list: the kernel's own ceiling.
    let cal_bytes = 768usize << 20;
    match Buffer::alloc(cal_bytes / 4) {
        Ok(big) => {
            if big.zero().is_err() {
                eprintln!("gpu-sweep: calibration zero failed");
            }
            for blocks in [1024usize, 2048, 4096, 8192, 16384, 32768] {
                let mut best = f64::INFINITY;
                for _ in 0..a.reps {
                    let t = Instant::now();
                    if let Err(e) = l.stream_read_ptr(big.device_ptr(), cal_bytes, blocks, &out) {
                        eprintln!("gpu-sweep: calibration: {e}");
                        return 1;
                    }
                    if let Err(e) = l.synchronize() {
                        eprintln!("gpu-sweep: sync: {e}");
                        return 1;
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                report(&format!("768MB buf/{blocks} blocks"), cal_bytes, 1, best);
            }
        }
        Err(e) => println!("calibration buffer: skipped ({e})"),
    }

    // A: one launch per tensor, which is the graph's shape today.
    let mut best_a = f64::INFINITY;
    for _ in 0..a.reps {
        let t = Instant::now();
        for t2 in &w {
            let blocks = blocks_for(t2.bytes, a.max_blocks);
            if let Err(e) = l.stream_read(t2.buf, t2.bytes, blocks, &out) {
                eprintln!("gpu-sweep: read: {e}");
                return 1;
            }
        }
        if let Err(e) = l.synchronize() {
            eprintln!("gpu-sweep: sync: {e}");
            return 1;
        }
        best_a = best_a.min(t.elapsed().as_secs_f64());
    }
    report("one launch/tensor", total, w.len(), best_a);

    // C: where the per-launch time goes. Record an event pair around every
    // launch of the per-tensor pass, then read all pairs back. If the sum of
    // kernel spans is much smaller than the wall clock, the loss is idle time
    // between launches; if it is close, the loss is inside the kernels.
    if let Err(e) = event_pass(&l, &w, &out, total) {
        eprintln!("gpu-sweep: events: {e}");
        return 1;
    }

    if a.gemv {
        if let Err(e) = gemv_pass(&l, &model, total) {
            eprintln!("gpu-sweep: gemv: {e}");
            return 1;
        }
    }

    if a.variant {
        if let Err(e) = variant_pass(&l, &model) {
            eprintln!("gpu-sweep: variant: {e}");
            return 1;
        }
    }

    if a.pair {
        if let Err(e) = pair_pass(&l, &model) {
            eprintln!("gpu-sweep: pair: {e}");
            return 1;
        }
    }

    if a.merge {
        if let Err(e) = merge_pass(&l, &model) {
            eprintln!("gpu-sweep: merge: {e}");
            return 1;
        }
    }

    if a.split {
        if let Err(e) = split_pass(&l, &model) {
            eprintln!("gpu-sweep: split: {e}");
            return 1;
        }
    }

    // B: the whole set in one launch, at several grid sizes.
    for blocks in [2048usize, 4096, 8192, 16384, 32768] {
        let mut best = f64::INFINITY;
        for _ in 0..a.reps {
            let t = Instant::now();
            if let Err(e) = l.stream_read_jobs(&jobs, w.len(), blocks, &out) {
                eprintln!("gpu-sweep: jobs: {e}");
                return 1;
            }
            if let Err(e) = l.synchronize() {
                eprintln!("gpu-sweep: sync: {e}");
                return 1;
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        report(&format!("one launch/{blocks} blocks"), total, 1, best);
    }
    0
}

pub(crate) fn blocks_for(bytes: usize, max: usize) -> usize {
    // One float4 per thread where the tensor is small enough, which is the
    // shape that measured fastest on the calibration buffer.
    (bytes / 16 / 256).clamp(256, max)
}
