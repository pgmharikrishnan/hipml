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
    let bytes: usize = pairs
        .iter()
        .map(|(g, u, _, _)| g.len() + u.len())
        .sum();

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
pub(crate) fn gemv_pass(l: &hipml_kernels::Launcher, model: &Model, _total: usize) -> Result<(), String> {
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
    println!("events: bytes {:.2} GB, span bandwidth {:.1} GB/s", total as f64 / 1e9, total as f64 / (sum / 1e3) / 1e9);
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
    let mut sets: Vec<(&str, Vec<(&ByteBuffer, usize, usize, &'static str, &'static str)>)> =
        Vec::new();
    for suffix in ["ffn_gate", "ffn_up", "ffn_down"] {
        let mut v = Vec::new();
        for i in 0..model.tensors().len() {
            let name = format!("blk.{i}.{suffix}.weight");
            let t = match model.tensor(&name) {
                Some(t) => t,
                None => continue,
            };
            let (a, b) = match t.ty() {
                TensorType::Q4K => ("gemv_q4_k", "gemv_q4_k_r1"),
                TensorType::Q6K => ("gemv_q6_k", "gemv_q6_k_r1"),
                _ => continue,
            };
            v.push((
                t.buffer(),
                t.shape()[0] as usize,
                t.shape()[1] as usize,
                a,
                b,
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

        // Exactness: the two entries must write the same bits.
        let mut worst = 0usize;
        for (w, m, k, a, b) in v.iter().take(4) {
            l.run_quant_resident(a, w, &x, &ya, *m, *k)?;
            l.run_quant_resident(b, w, &x, &yb, *m, *k)?;
            l.synchronize()?;
            let ha = ya.download_bytes(0, m * 4)?;
            let hb = yb.download_bytes(0, m * 4)?;
            if ha != hb {
                worst += 1;
            }
        }

        for pass in 0..2 {
            let entry_idx = pass % 2;
            l.synchronize()?;
            let t = Instant::now();
            for (w, m, k, a, b) in v.iter() {
                let e = if entry_idx == 0 { a } else { b };
                l.run_quant_resident(e, w, &x, if entry_idx == 0 { &ya } else { &yb }, *m, *k)?;
            }
            l.synchronize()?;
            let secs = t.elapsed().as_secs_f64();
            let which = if entry_idx == 0 { "2 rows/warp" } else { "1 row/warp " };
            println!(
                "{label:<9} {which} {:6.2} GB  {:7.3} ms  {:6.1} GB/s  ({} launches, {} blocks/launch avg)",
                bytes as f64 / 1e9,
                secs * 1e3,
                bytes as f64 / secs / 1e9,
                v.len(),
                v.iter().map(|t| t.1 / 16).sum::<usize>() / v.len()
            );
        }
        if worst > 0 {
            return Err(format!("{label}: {worst} of 4 outputs differ between entries"));
        }
        println!("{label:<9} entries write identical bytes (4 sampled tensors)");
    }
    Ok(())
}

/// Job kind for a quantized tensor, or `None` if the job kernel has no body.
fn job_kind(t: &hipml_model::DeviceTensor) -> Option<i32> {
    use hipml_kernels::{GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80};
    Some(match t.ty() {
        TensorType::Q4K => GEMV_JOB_Q4K,
        TensorType::Q6K => GEMV_JOB_Q6K,
        TensorType::Q5K => GEMV_JOB_Q5K,
        TensorType::Q8_0 => GEMV_JOB_Q80,
        _ => return None,
    })
}

/// The ffn_gate/ffn_up pair of every layer: same input, independent, shipped as
/// two launches per layer for 96 launches a token.
///
/// Three passes over the same 6.8 GB: sequential (what the graph does), one
/// kernel per pair (48 launches), and all 96 jobs in one launch (the ceiling a
/// token-long persistent queue would need). The last one is not schedulable —
/// the layers are ordered — so it bounds the idea rather than proposing it.
pub(crate) fn merge_pass(l: &hipml_kernels::Launcher, model: &Model) -> Result<(), String> {
    use hipml_kernels::{gemv_job_tiles, GemvJob};

    struct Pair<'a> {
        gate: (&'a ByteBuffer, usize, usize, i32),
        up: (&'a ByteBuffer, usize, usize, i32),
    }
    let mut pairs: Vec<Pair<'_>> = Vec::new();
    for i in 0..64 {
        let g = match model.tensor(&format!("blk.{i}.ffn_gate.weight")) {
            Some(t) => t,
            None => continue,
        };
        let u = match model.tensor(&format!("blk.{i}.ffn_up.weight")) {
            Some(t) => t,
            None => continue,
        };
        let (gk, uk) = match (job_kind(g), job_kind(u)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        pairs.push(Pair {
            gate: (g.buffer(), g.shape()[0] as usize, g.shape()[1] as usize, gk),
            up: (u.buffer(), u.shape()[0] as usize, u.shape()[1] as usize, uk),
        });
    }
    if pairs.is_empty() {
        return Err("no ffn pairs".to_string());
    }
    let k = pairs[0].gate.2;
    let m = pairs[0].gate.1;
    let bytes: usize = pairs
        .iter()
        .map(|p| p.gate.0.len() + p.up.0.len())
        .sum();
    let x = Buffer::alloc(k)?;
    let yg = Buffer::alloc(m)?;
    let yu = Buffer::alloc(m)?;
    let yb = Buffer::alloc(m.max(pairs[0].up.1))?;
    let entries = ["gemv_q4_k", "gemv_q6_k", "gemv_q5_k", "gemv_q8_0"];

    let pair_jobs: Vec<Buffer> = pairs
        .iter()
        .map(|p| {
            job_buffer(&[
                GemvJob {
                    w: p.gate.0.device_ptr(),
                    x: x.as_ptr() as usize,
                    y: yg.as_ptr() as usize,
                    m: p.gate.1 as i32,
                    k: p.gate.2 as i32,
                    kind: p.gate.3,
                    pad: 0,
                },
                GemvJob {
                    w: p.up.0.device_ptr(),
                    x: x.as_ptr() as usize,
                    y: yu.as_ptr() as usize,
                    m: p.up.1 as i32,
                    k: p.up.2 as i32,
                    kind: p.up.3,
                    pad: 0,
                },
            ])
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tiles = pairs
        .iter()
        .map(|p| gemv_job_tiles(p.gate.3, p.gate.1).max(gemv_job_tiles(p.up.3, p.up.1)))
        .max()
        .unwrap_or(1);

    // Exactness: the merged kernel must write the same bits as the two entries.
    let mut mismatched = 0usize;
    for p in pairs.iter().take(4) {
        l.run_quant_resident(entries[p.gate.3 as usize], p.gate.0, &x, &yg, p.gate.1, p.gate.2)?;
        l.run_quant_resident(entries[p.up.3 as usize], p.up.0, &x, &yu, p.up.1, p.up.2)?;
        l.synchronize()?;
        let a_gate = yg.download_bytes(0, p.gate.1 * 4)?;
        let a_up = yu.download_bytes(0, p.up.1 * 4)?;
        let jobs = job_buffer(&[
            GemvJob {
                w: p.gate.0.device_ptr(),
                x: x.as_ptr() as usize,
                y: yb.as_ptr() as usize,
                m: p.gate.1 as i32,
                k: p.gate.2 as i32,
                kind: p.gate.3,
                pad: 0,
            },
        ])?;
        l.gemv_jobs(&jobs, 1, gemv_job_tiles(p.gate.3, p.gate.1))?;
        l.synchronize()?;
        if yb.download_bytes(0, p.gate.1 * 4)? != a_gate {
            mismatched += 1;
        }
        let jobs = job_buffer(&[
            GemvJob {
                w: p.up.0.device_ptr(),
                x: x.as_ptr() as usize,
                y: yb.as_ptr() as usize,
                m: p.up.1 as i32,
                k: p.up.2 as i32,
                kind: p.up.3,
                pad: 0,
            },
        ])?;
        l.gemv_jobs(&jobs, 1, gemv_job_tiles(p.up.3, p.up.1))?;
        l.synchronize()?;
        if yb.download_bytes(0, p.up.1 * 4)? != a_up {
            mismatched += 1;
        }
    }
    if mismatched > 0 {
        return Err(format!("{mismatched} of 8 merged rows differ from the entries"));
    }
    println!("merge: merged rows identical to the standalone entries (8 sampled)");

    let pass = |label: &str, launches: usize, f: &dyn Fn() -> Result<(), String>| -> Result<(), String> {
        f()?;
        l.synchronize()?;
        let t = Instant::now();
        f()?;
        l.synchronize()?;
        report(label, bytes, launches, t.elapsed().as_secs_f64());
        Ok(())
    };

    pass("ffn pair sequential", pairs.len() * 2, &|| {
        for p in &pairs {
            l.run_quant_resident(entries[p.gate.3 as usize], p.gate.0, &x, &yg, p.gate.1, p.gate.2)?;
            l.run_quant_resident(entries[p.up.3 as usize], p.up.0, &x, &yu, p.up.1, p.up.2)?;
        }
        Ok(())
    })?;
    pass("ffn pair merged", pairs.len(), &|| {
        for jobs in &pair_jobs {
            l.gemv_jobs(jobs, 2, tiles)?;
        }
        Ok(())
    })?;

    // Ceiling: every pair's two jobs in one launch. Only the ordering between
    // layers stops the token from doing this.
    let mut all: Vec<GemvJob> = Vec::with_capacity(pairs.len() * 2);
    for p in &pairs {
        all.push(GemvJob {
            w: p.gate.0.device_ptr(),
            x: x.as_ptr() as usize,
            y: yg.as_ptr() as usize,
            m: p.gate.1 as i32,
            k: p.gate.2 as i32,
            kind: p.gate.3,
            pad: 0,
        });
        all.push(GemvJob {
            w: p.up.0.device_ptr(),
            x: x.as_ptr() as usize,
            y: yu.as_ptr() as usize,
            m: p.up.1 as i32,
            k: p.up.2 as i32,
            kind: p.up.3,
            pad: 0,
        });
    }
    let all_buf = job_buffer(&all)?;
    pass("ffn pair one launch", 1, &|| l.gemv_jobs(&all_buf, all.len(), tiles))
}

/// `GemvJob` is `repr(C)` plain data; this is the byte view the buffer needs.
fn job_buffer(jobs: &[hipml_kernels::GemvJob]) -> Result<Buffer, String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(jobs.as_ptr() as *const u8, std::mem::size_of_val(jobs))
    };
    let buf = Buffer::alloc(bytes.len().div_ceil(4))?;
    buf.upload_bytes(0, bytes)?;
    Ok(buf)
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
