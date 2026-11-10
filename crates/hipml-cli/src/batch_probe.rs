// SPDX-License-Identifier: Apache-2.0
//! Per-tensor batched-GEMV probe behind `batch-bench --probe`.

use std::time::Instant;

use hipml_model::Model;

/// Per-tensor batched-GEMV probe: weight-only GB/s for one pass at B=1 and at
/// the requested B, so the kernel that stops scaling is named with numbers.
pub fn probe_kernels(model: &Model, slots: usize) {
    use hipml_hip::Buffer;
    use hipml_model::gpu_weights::{batch_kind_tensor, gemv_entry};
    let l = model.launcher();
    let names = [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.51.ffn_down.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_out.weight",
        "blk.3.attn_q.weight",
        "output.weight",
    ];
    let time = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            f()?;
            l.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        Ok(best)
    };
    println!("kernel probe (weight-only GB/s, one pass):");
    for name in names {
        let Some(t) = model.tensor(name) else {
            continue;
        };
        let shape = t.shape();
        if shape.len() != 2 {
            continue;
        }
        let (m, k) = (shape[0] as usize, shape[1] as usize);
        let Ok(entry) = gemv_entry(t.ty()) else {
            continue;
        };
        let Some(kind) = batch_kind_tensor(t.ty()) else {
            continue;
        };
        let bytes = t.nbytes() as f64;
        let x = match Buffer::alloc(slots * k) {
            Ok(b) => b,
            Err(e) => {
                println!("  {name}: alloc: {e}");
                continue;
            }
        };
        let y = match Buffer::alloc(slots * m) {
            Ok(b) => b,
            Err(e) => {
                println!("  {name}: alloc: {e}");
                continue;
            }
        };
        // The single-sequence entry asserts x is exactly one row.
        let (x1, y1) = match (Buffer::alloc(k), Buffer::alloc(m)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                println!("  {name}: alloc: failed");
                continue;
            }
        };
        let single = time(&|| l.run_quant_resident(entry, t.buffer(), &x1, &y1, m, k)).ok();
        let batched = time(&|| l.gemv_batched(kind, t.buffer(), &x, &y, m, k, slots)).ok();
        let gbs = |s: Option<f64>| match s {
            Some(sec) if sec > 0.0 => format!("{:6.0} GB/s", bytes / sec / 1e9),
            _ => "   n/a".to_string(),
        };
        println!(
            "  {name:<24} {:>10}  single {}  B={:<2} {}",
            format!("{}x{}", m, k),
            gbs(single),
            slots,
            gbs(batched)
        );
    }
}
