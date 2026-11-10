// SPDX-License-Identifier: Apache-2.0
//! The A/B passes behind `gpu-sweep`: real-GEMV spans, a two-stream pair test
//! and the 1-row-per-warp variant test.
//!
//! Split from `sweep_cmd` to keep both readable; the driver decides which pass
//! runs.

use std::time::Instant;

use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, ByteBuffer, Event};
use hipml_model::Model;

use crate::sweep_cmd::{blocks_for, Weight};

/// Every layer's (ffn_gate, ffn_up) pair, run sequentially and then
/// concurrently on the two streams. Same bytes, same kernels: the difference is
/// only whether the two weight streams share the machine.
pub(crate) fn pair_pass(l: &hipml_kernels::Launcher, model: &Model) -> Result<(), String> {
    let mut pairs: Vec<(&ByteBuffer, &ByteBuffer, usize, usize)> = Vec::new();
    for i in 0..model.tensors().len() {
        let name = format!("blk.{i}.ffn_gate.weight");
        let g = match model.tensor(&name) {
            Some(t) => t,
            None => continue,
        };
        let name = format!("blk.{i}.ffn_up.weight");
        let u = match model.tensor(&name) {
            Some(t) => t,
            None => continue,
        };
        let (m, k) = (g.shape()[0] as usize, g.shape()[1] as usize);
        pairs.push((g.buffer(), u.buffer(), m, k));
    }
    let k = pairs[0].3;
    let (m, _) = (pairs[0].2, pairs[0].3);
    let x = Buffer::alloc(k)?;
    let yg = Buffer::alloc(m)?;
    let yu = Buffer::alloc(m)?;
    let bytes: usize = pairs.iter().map(|(g, u, _, _)| g.len() + u.len()).sum();

    // Warm.
    for (g, u, m, k) in &pairs {
        l.run_quant_resident("gemv_q4_k", g, &x, &yg, *m, *k)?;
        l.run_quant_resident("gemv_q6_k", u, &x, &yu, *m, *k)?;
    }
    l.synchronize()?;

    let t = Instant::now();
    for (g, u, m, k) in &pairs {
        l.run_quant_resident("gemv_q4_k", g, &x, &yg, *m, *k)?;
        l.run_quant_resident("gemv_q6_k", u, &x, &yu, *m, *k)?;
    }
    l.synchronize()?;
    let seq = t.elapsed().as_secs_f64();
    report("pair sequential", bytes, pairs.len() * 2, seq);

    let t = Instant::now();
    for (g, u, m, k) in &pairs {
        l.fork()?;
        l.run_quant_resident("gemv_q4_k", g, &x, &yg, *m, *k)?;
        l.run_quant_resident_on("side", "gemv_q6_k", u, &x, &yu, *m, *k)?;
        l.join()?;
    }
    l.synchronize()?;
    let par = t.elapsed().as_secs_f64();
    report("pair two-stream", bytes, pairs.len() * 2, par);
    println!("  speedup: {:.2}x", seq / par);
    Ok(())
}

/// The real GEMV kernel over the same set, one launch per tensor, timed with
/// events. This is the number the graph actually pays: a pure read is not the
/// kernel the decode step runs.
pub(crate) fn gemv_pass(
    l: &hipml_kernels::Launcher,
    model: &Model,
    _total: usize,
) -> Result<(), String> {
    let mut xs: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    let mut outs: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    let mut marks = Vec::new();
    let mut items: Vec<(&ByteBuffer, usize, usize, &'static str)> = Vec::new();
    for t in model.tensors() {
        let shape = t.shape();
        if shape.len() != 2 || t.name().ends_with("token_embd.weight") {
            continue;
        }
        let entry = match t.ty() {
            TensorType::Q4K => "gemv_q4_k",
            TensorType::Q6K => "gemv_q6_k",
            TensorType::Q8_0 => "gemv_q8_0",
            _ => continue,
        };
        let (m, k) = (shape[0] as usize, shape[1] as usize);
        if k > 32768 {
            continue;
        }
        let bytes = t.nbytes() as usize;
        if bytes < 1 << 20 {
            continue;
        }
        if !outs.contains_key(&m) {
            outs.insert(m, Buffer::alloc(m)?);
        }
        if !xs.contains_key(&k) {
            xs.insert(k, Buffer::alloc(k)?);
        }
        items.push((t.buffer(), m, k, entry));
    }
    for _ in 0..items.len() {
        marks.push((Event::new()?, Event::new()?));
    }
    for (w, m, k, entry) in items.iter() {
        l.run_quant_resident(entry, w, &xs[k], &outs[m], *m, *k)?;
    }
    l.synchronize()?;

    let wall = Instant::now();
    for ((w, m, k, entry), (a, b)) in items.iter().zip(marks.iter()) {
        a.record(l.stream())?;
        l.run_quant_resident(entry, w, &xs[k], &outs[m], *m, *k)?;
        b.record(l.stream())?;
    }
    l.synchronize()?;
    let wall = wall.elapsed().as_secs_f64();

    let mut per: std::collections::HashMap<&str, (usize, f64)> = std::collections::HashMap::new();
    let mut sum = 0.0f64;
    let mut bytes_all = 0usize;
    for ((w, _, _, entry), (a, b)) in items.iter().zip(marks.iter()) {
        let ms = a.since(b)? as f64;
        let bytes = w.len();
        sum += ms;
        bytes_all += bytes;
        let e = per.entry(*entry).or_insert((0, 0.0));
        e.0 += bytes;
        e.1 += ms;
    }
    println!(
        "gemv: wall {:7.3} ms, spans {:7.3} ms, idle {:6.3} ms ({:.1} us/launch) in {} launches",
        wall * 1e3,
        sum,
        (wall * 1e3 - sum).max(0.0),
        (wall * 1e3 - sum).max(0.0) / items.len() as f64,
        items.len()
    );
    println!(
        "gemv: {:.2} GB, span bandwidth {:6.1} GB/s",
        bytes_all as f64 / 1e9,
        bytes_all as f64 / (sum / 1e3) / 1e9
    );
    let mut kinds: Vec<_> = per.into_iter().collect();
    kinds.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (entry, (bytes, ms)) in kinds {
        println!(
            "  {:12} {:6.2} GB  {:7.3} ms  {:6.1} GB/s",
            entry,
            bytes as f64 / 1e9,
            ms,
            bytes as f64 / (ms / 1e3) / 1e9
        );
    }
    Ok(())
}

/// Per-launch GPU spans for the one-launch-per-tensor stream.
pub(crate) fn event_pass(
    l: &hipml_kernels::Launcher,
    w: &[Weight<'_>],
    out: &Buffer,
    total: usize,
) -> Result<(), String> {
    let mut marks = Vec::with_capacity(w.len());
    for _ in 0..w.len() {
        marks.push((Event::new()?, Event::new()?));
    }
    // Warm the code object; the first launch in a module pays residency.
    l.stream_read(w[0].buf, w[0].bytes, 1024, out)?;
    l.synchronize()?;

    let wall = Instant::now();
    for (t, (a, b)) in w.iter().zip(marks.iter()) {
        a.record(l.stream())?;
        l.stream_read(t.buf, t.bytes, blocks_for(t.bytes, 8192), out)?;
        b.record(l.stream())?;
    }
    l.synchronize()?;
    let wall = wall.elapsed().as_secs_f64();

    let mut sum = 0.0f64;
    let mut rows: Vec<(usize, f64, f64)> = Vec::with_capacity(w.len());
    for (t, (a, b)) in w.iter().zip(marks.iter()) {
        let ms = a.since(b)? as f64;
        sum += ms;
        let floor = t.bytes as f64 / 632e9 * 1e3;
        rows.push((t.bytes, ms, floor));
    }
    rows.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "events: wall {:7.3} ms, kernel spans {:7.3} ms, idle between {:6.3} ms ({:.1} us/launch)",
        wall * 1e3,
        sum,
        (wall * 1e3 - sum).max(0.0),
        (wall * 1e3 - sum).max(0.0) / w.len() as f64
    );
    println!(
        "events: bytes {:.2} GB, span bandwidth {:.1} GB/s",
        total as f64 / 1e9,
        total as f64 / (sum / 1e3) / 1e9
    );
    for (bytes, ms, floor) in rows.iter().take(5) {
        println!(
            "  {:6.1} MB  span {:8.3} ms  floor {:8.3} ms  overhead {:6.3} ms",
            *bytes as f64 / 1e6,
            ms,
            floor,
            ms - floor
        );
    }
    println!("  ...");
    for (bytes, ms, floor) in rows.iter().rev().take(3) {
        println!(
            "  {:6.1} MB  span {:8.3} ms  floor {:8.3} ms  overhead {:6.3} ms",
            *bytes as f64 / 1e6,
            ms,
            floor,
            ms - floor
        );
    }
    Ok(())
}

/// Two-row (default) versus one-row-per-warp kernels over the same tensors.
/// The 1-row entry doubles the grid for shapes whose row count leaves the
/// machine half empty (5120 rows = 320 blocks at 16 rows/block).
pub(crate) fn variant_pass(l: &hipml_kernels::Launcher, model: &Model) -> Result<(), String> {
    let mut sets: Vec<(
        &str,
        Vec<(
            &ByteBuffer,
            usize,
            usize,
            &'static str,
            &'static str,
            &'static str,
        )>,
    )> = Vec::new();
    for suffix in [
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "attn_qkv",
        "attn_gate",
        "ssm_out",
        "attn_output",
    ] {
        let mut v = Vec::new();
        for i in 0..model.tensors().len() {
            let name = format!("blk.{i}.{suffix}.weight");
            let t = match model.tensor(&name) {
                Some(t) => t,
                None => continue,
            };
            let (a, b, c) = match t.ty() {
                TensorType::Q4K => ("gemv_q4_k", "gemv_q4_k_r1", "gemv_q4_k_u4"),
                TensorType::Q6K => ("gemv_q6_k", "gemv_q6_k_r1", "gemv_q6_k_r1"),
                _ => continue,
            };
            v.push((
                t.buffer(),
                t.shape()[0] as usize,
                t.shape()[1] as usize,
                a,
                b,
                c,
            ));
        }
        sets.push((suffix, v));
    }

    for (label, v) in &sets {
        if v.is_empty() {
            continue;
        }
        let k = v[0].2;
        let out_max = v.iter().map(|t| t.1).max().unwrap_or(1);
        let x = Buffer::alloc(k)?;
        let ya = Buffer::alloc(out_max)?;
        let yb = Buffer::alloc(out_max)?;
        let bytes: usize = v.iter().map(|t| t.0.len()).sum();

        // Exactness: every entry must write the same bits.
        let mut worst = 0usize;
        for (w, m, k, a, b, c) in v.iter().take(4) {
            l.run_quant_resident(a, w, &x, &ya, *m, *k)?;
            l.synchronize()?;
            let ha = ya.download_bytes(0, m * 4)?;
            for e in [b, c] {
                l.run_quant_resident(e, w, &x, &yb, *m, *k)?;
                l.synchronize()?;
                if yb.download_bytes(0, m * 4)? != ha {
                    worst += 1;
                }
            }
        }

        let names = ["2 rows/warp", "1 row/warp ", "1 row + x4 "];
        for pass in 0..3 {
            l.synchronize()?;
            let t = Instant::now();
            for (w, m, k, a, b, c) in v.iter() {
                let e = match pass {
                    0 => a,
                    1 => b,
                    _ => c,
                };
                l.run_quant_resident(e, w, &x, if pass == 0 { &ya } else { &yb }, *m, *k)?;
            }
            l.synchronize()?;
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{label:<12} {} {:6.2} GB  {:7.3} ms  {:6.1} GB/s  ({} launches, {} blocks/launch avg)",
                names[pass],
                bytes as f64 / 1e9,
                secs * 1e3,
                bytes as f64 / secs / 1e9,
                v.len(),
                v.iter().map(|t| t.1 / 16).sum::<usize>() / v.len()
            );
        }
        if worst > 0 {
            return Err(format!(
                "{label}: {worst} of 8 outputs differ between entries"
            ));
        }
        println!("{label:<12} entries write identical bytes (4 sampled tensors)");
    }
    Ok(())
}

/// One line of the sweep table, shared with the driver.
pub(crate) fn report(label: &str, bytes: usize, launches: usize, secs: f64) {
    println!(
        "read {label:<22} {:.2} GB in {:7.3} ms  {:6.1} GB/s  ({} launches)",
        bytes as f64 / 1e9,
        secs * 1e3,
        bytes as f64 / secs / 1e9,
        launches
    );
}
