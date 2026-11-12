// SPDX-License-Identifier: Apache-2.0
//! `hipml stream-shape`: the pure-read ceiling at the decode stream's own
//! tensor shapes and grid geometries.
//!
//! For every resident weight the decode graph streams, launch a kernel that
//! reads exactly those bytes and does nothing else, once per pass, with one
//! synchronize at the end. Five geometries on the same bytes:
//!
//!   w4       GEMV row tile (16 rows/block, 2 rows/warp), four-byte loads --
//!            the width an unaligned block row forces;
//!   w16      the same tile with sixteen-byte loads -- the width a 16-byte
//!            aligned row stride would allow;
//!   w4r1     one row per warp (8 rows/block), for the under-occupied grids;
//!   flat     the tensor as one span, at the GEMV's own block count;
//!   flatmax  the same span as widely gridded as the reader allows (the
//!            calibration-buffer shape behind the 598 GB/s figure).
//!
//! A GEMV at the same shape can only be as fast as the row-tile stream, so the
//! difference between `w4` and the GEMV is arithmetic, and the difference
//! between `w4` and `flat`/`w16` is the layout.

use std::time::Instant;

use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, Event};
use hipml_model::Model;

struct Args {
    gguf: String,
    reps: usize,
    min_mb: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        reps: 3,
        min_mb: 1,
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
            "--min-mb" => {
                a.min_mb = need(i)?.parse().map_err(|e| format!("bad --min-mb: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf is required".to_string());
    }
    Ok(a)
}

struct Item<'a> {
    buf: &'a hipml_hip::ByteBuffer,
    m: usize,
    k: usize,
    row_bytes: usize,
    bytes: usize,
    ty: &'static str,
}

/// (label, entry, blocks) for one geometry, at this tensor's shape.
fn variant(item: &Item<'_>, name: &str) -> (&'static str, usize) {
    let r2 = item.m.div_ceil(16);
    let r1 = item.m.div_ceil(8);
    match name {
        "w4" => ("stream_shape_w4", r2),
        "w16" => ("stream_shape_w16", r2),
        "w4r1" => ("stream_shape_w4_r1", r1),
        "flat" => ("stream_shape_flat", r2),
        _ => (
            "stream_shape_flat",
            (item.bytes / 16 / 256).clamp(256, 16384),
        ),
    }
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("stream-shape: {e}");
            return 2;
        }
    };
    let (model, _report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("stream-shape: load: {e}");
            return 1;
        }
    };
    let l = model.launcher();

    let mut items: Vec<Item<'_>> = Vec::new();
    for t in model.tensors() {
        let shape = t.shape();
        if shape.len() != 2 || t.name().ends_with("token_embd.weight") {
            continue;
        }
        let bytes = t.nbytes() as usize;
        if bytes < a.min_mb * 1_000_000 {
            continue;
        }
        let (m, k) = (shape[0] as usize, shape[1] as usize);
        if m == 0 || k == 0 || bytes % m != 0 {
            continue;
        }
        let ty = match t.ty() {
            TensorType::Q4K => "Q4_K",
            TensorType::Q5K => "Q5_K",
            TensorType::Q6K => "Q6_K",
            TensorType::Q8_0 => "Q8_0",
            TensorType::IQ4XS => "IQ4_XS",
            TensorType::IQ4NL => "IQ4_NL",
            TensorType::Q3K => "Q3_K",
            TensorType::IQ3S => "IQ3_S",
            _ => continue,
        };
        items.push(Item {
            buf: t.buffer(),
            m,
            k,
            row_bytes: bytes / m,
            bytes,
            ty,
        });
    }
    if items.is_empty() {
        eprintln!("stream-shape: no 2-D quantized tensors matched");
        return 1;
    }
    let total: usize = items.iter().map(|i| i.bytes).sum();
    let aligned = items.iter().filter(|i| i.row_bytes % 16 == 0).count();
    println!(
        "device: {}  {} tensors, {:.2} GB streamed per pass",
        model.device_name(),
        items.len(),
        total as f64 / 1e9
    );
    println!(
        "shape summary: {aligned} of {} tensors have 16-byte aligned rows",
        items.len()
    );

    let max_blocks = items
        .iter()
        .map(|i| variant(i, "flatmax").1)
        .max()
        .unwrap_or(4096);
    let out = match Buffer::alloc(max_blocks * 8 + 64) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("stream-shape: alloc: {e}");
            return 1;
        }
    };

    for name in ["w4", "w16", "w4r1", "flat", "flatmax"] {
        // One warm pass (first touch of each allocation), then `reps` timed
        // passes. Each pass issues one launch per tensor and one synchronize at
        // the end, so the per-tensor spans are the same accounting as
        // `gpu-sweep --gemv`.
        let mut marks = Vec::with_capacity(items.len());
        for _ in 0..items.len() {
            match (Event::new(), Event::new()) {
                (Ok(x), Ok(y)) => marks.push((x, y)),
                _ => {
                    eprintln!("stream-shape: could not create events");
                    return 1;
                }
            }
        }
        let mut best: Option<Vec<(&'static str, usize, f64)>> = None;
        let mut best_sum = f64::MAX;
        let mut idle = 0.0f64;
        for pass in 0..=a.reps {
            let t = Instant::now();
            for (idx, it) in items.iter().enumerate() {
                let (entry, blocks) = variant(it, name);
                if pass > 0 {
                    if let Err(e) = marks[idx].0.record(l.stream()) {
                        eprintln!("stream-shape: {e}");
                        return 1;
                    }
                }
                if let Err(e) =
                    l.stream_shape(entry, it.buf, it.m, it.k, it.row_bytes, blocks, 256, &out)
                {
                    eprintln!("stream-shape: {name}: {e}");
                    return 1;
                }
                if pass > 0 {
                    if let Err(e) = marks[idx].1.record(l.stream()) {
                        eprintln!("stream-shape: {e}");
                        return 1;
                    }
                }
            }
            if let Err(e) = l.synchronize() {
                eprintln!("stream-shape: sync: {e}");
                return 1;
            }
            let wall = t.elapsed().as_secs_f64() * 1e3;
            if pass == 0 {
                continue;
            }
            let mut per = Vec::with_capacity(items.len());
            let mut sum = 0.0f64;
            for (idx, it) in items.iter().enumerate() {
                let ms = match marks[idx].0.since(&marks[idx].1) {
                    Ok(v) => v as f64,
                    Err(e) => {
                        eprintln!("stream-shape: {e}");
                        return 1;
                    }
                };
                sum += ms;
                per.push((it.ty, it.bytes, ms));
            }
            if sum < best_sum {
                best_sum = sum;
                idle = (wall - sum).max(0.0);
                best = Some(per);
            }
        }
        let per = best.unwrap_or_default();
        println!(
            "\n{name}: spans {:.3} ms, idle {:.3} ms ({:.1} us/launch), {:.1} GB/s over {:.2} GB",
            best_sum,
            idle,
            idle / items.len() as f64 * 1e3,
            total as f64 / (best_sum / 1e3) / 1e9,
            total as f64 / 1e9
        );
        let mut agg: std::collections::BTreeMap<&str, (usize, f64)> =
            std::collections::BTreeMap::new();
        for (ty, bytes, ms) in per {
            let e = agg.entry(ty).or_insert((0, 0.0));
            e.0 += bytes;
            e.1 += ms;
        }
        for (ty, (bytes, ms)) in agg {
            println!(
                "  {:6} {:6.3} GB  {:8.3} ms  {:6.1} GB/s",
                ty,
                bytes as f64 / 1e9,
                ms,
                bytes as f64 / (ms / 1e3) / 1e9
            );
        }
    }
    0
}
