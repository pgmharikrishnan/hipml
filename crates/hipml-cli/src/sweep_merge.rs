// SPDX-License-Identifier: Apache-2.0
//! The batch and split probes behind `gpu-sweep --merge` / `--split`.
//!
//! Both read the model's own resident weights: one kernel over a device job list
//! (independent projections merged, so one memory-pipeline fill instead of two),
//! and one kernel per k-slice of a row (so the grid is wider than the row count).
//! Split out from `sweep_ab` to keep both readable; `pair_pass` there is the same
//! question asked with two streams.

use std::time::Instant;

use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_model::Model;

use crate::sweep_ab::report;

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

/// Split-K A/B on the tensors the trace calls slow: 5120-row Q4_K projections
/// (ssm_out, attn_output) and the 17408-row ffn_down.
///
/// Same bytes, same kernel body, `splits` times more blocks and a reduce pass.
/// Reports effective GB/s per split count and whether the outputs still match
/// the unsplit kernel bit-for-bit (they are not expected to: the accumulation
/// order over k changes, which is why this is a probe and not a swap).
pub(crate) fn split_pass(l: &hipml_kernels::Launcher, model: &Model) -> Result<(), String> {
    for suffix in ["ssm_out", "attn_output", "attn_qkv", "attn_gate", "ffn_down"] {
        let mut v: Vec<(&ByteBuffer, usize, usize)> = Vec::new();
        for i in 0..64 {
            let t = match model.tensor(&format!("blk.{i}.{suffix}.weight")) {
                Some(t) => t,
                None => continue,
            };
            if t.ty() != TensorType::Q4K {
                continue;
            }
            v.push((t.buffer(), t.shape()[0] as usize, t.shape()[1] as usize));
        }
        if v.is_empty() {
            continue;
        }
        let (m_max, k) = (v.iter().map(|t| t.1).max().unwrap_or(1), v[0].2);
        let bytes: usize = v.iter().map(|t| t.0.len()).sum();
        let x = Buffer::alloc(k)?;
        // A real pattern: with an uninitialised (zero) x every output is zero
        // and any exactness comparison below would pass vacuously.
        let xf: Vec<f32> = (0..k).map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5).collect();
        x.upload_bytes(0, unsafe {
            std::slice::from_raw_parts(xf.as_ptr() as *const u8, k * 4)
        })?;
        let y = Buffer::alloc(m_max)?;
        let y2 = Buffer::alloc(m_max)?;
        let part = Buffer::alloc(m_max * 4)?;

        // Streaming-load variant over the same bytes.
        l.synchronize()?;
        let t = Instant::now();
        for (w, m, kk) in &v {
            l.run_quant_resident("gemv_q4_k_nt", w, &x, &y, *m, *kk)?;
        }
        l.synchronize()?;
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  sc1        {:6.2} GB  {:7.3} ms  {:6.1} GB/s  ({} launches)",
            bytes as f64 / 1e9,
            secs * 1e3,
            bytes as f64 / secs / 1e9,
            v.len()
        );

        for splits in [1usize, 2, 4, 8] {
            if splits > 1 {
                // Exactness against the unsplit entry for one tensor.
                let (w, m, _) = v[0];
                l.run_quant_resident("gemv_q4_k_r1", w, &x, &y, m, k)?;
                l.synchronize()?;
                let a = y.download_bytes(0, m * 4)?;
                l.gemv_q4_k_split(w, &x, &part, &y2, m, k, splits)?;
                l.synchronize()?;
                let b = y2.download_bytes(0, m * 4)?;
                let mut worst = 0.0f32;
                for (i, (p, q)) in a.chunks(4).zip(b.chunks(4)).enumerate() {
                    let (fa, fb) = (
                        f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
                        f32::from_le_bytes([q[0], q[1], q[2], q[3]]),
                    );
                    let _ = i;
                    worst = worst.max((fa - fb).abs() / fa.abs().max(1e-6));
                }
                print!("  splits {splits}: max rel |d| vs r1 = {worst:.3e}  ");
            } else {
                print!("  splits 1: ");
            }
            l.synchronize()?;
            let t = Instant::now();
            for (w, m, kk) in &v {
                if splits == 1 {
                    l.run_quant_resident("gemv_q4_k_r1", w, &x, &y, *m, *kk)?;
                } else {
                    l.gemv_q4_k_split(w, &x, &part, &y, *m, *kk, splits)?;
                }
            }
            l.synchronize()?;
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{:6.2} GB  {:7.3} ms  {:6.1} GB/s  ({} launches)",
                bytes as f64 / 1e9,
                secs * 1e3,
                bytes as f64 / secs / 1e9,
                v.len() * if splits == 1 { 1 } else { 2 }
            );
        }
    }
    Ok(())
}

