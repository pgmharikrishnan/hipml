// SPDX-License-Identifier: Apache-2.0
//! `hipml gemv-entries`: the whole-type GEMV rate for several entries over the
//! same resident tensors, with a bit-exactness check against the production one.
//!
//! One launch per tensor, one synchronize per pass, best of `--reps`: the same
//! accounting as `gpu-sweep --gemv`, so a candidate's number is directly
//! comparable with the production table. The first entry named is the
//! reference; every other entry must produce the same output bits on every
//! tensor of the kind or it is reported as differing.

use std::time::Instant;

use hipml_core::quant::TensorType;
use hipml_hip::{Buffer, Event};
use hipml_model::Model;

struct Args {
    gguf: String,
    kind: String,
    entries: Vec<String>,
    reps: usize,
    check: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        kind: String::new(),
        entries: Vec::new(),
        reps: 3,
        check: true,
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
            "--kind" => {
                a.kind = need(i)?.to_string();
                i += 2;
            }
            "--entries" => {
                a.entries = need(i)?.split(',').map(|s| s.to_string()).collect();
                i += 2;
            }
            "--reps" => {
                a.reps = need(i)?.parse().map_err(|e| format!("bad --reps: {e}"))?;
                i += 2;
            }
            "--no-check" => {
                a.check = false;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() || a.kind.is_empty() || a.entries.is_empty() {
        return Err("--gguf, --kind and --entries are required".to_string());
    }
    Ok(a)
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gemv-entries: {e}");
            return 2;
        }
    };
    let ty = match a.kind.as_str() {
        "Q4_K" => TensorType::Q4K,
        "Q5_K" => TensorType::Q5K,
        "Q6_K" => TensorType::Q6K,
        "Q8_0" => TensorType::Q8_0,
        "IQ4_XS" => TensorType::IQ4XS,
        "IQ4_NL" => TensorType::IQ4NL,
        "Q3_K" => TensorType::Q3K,
        "IQ3_S" => TensorType::IQ3S,
        other => {
            eprintln!("gemv-entries: unknown --kind {other}");
            return 2;
        }
    };
    let (model, _report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gemv-entries: load: {e}");
            return 1;
        }
    };
    let l = model.launcher();

    struct Item<'a> {
        buf: &'a hipml_hip::ByteBuffer,
        m: usize,
        k: usize,
        bytes: usize,
    }
    let mut items: Vec<Item<'_>> = Vec::new();
    for t in model.tensors() {
        if t.ty() != ty {
            continue;
        }
        let shape = t.shape();
        if shape.len() != 2 || t.name().ends_with("token_embd.weight") {
            continue;
        }
        let bytes = t.nbytes() as usize;
        if bytes < 1_000_000 {
            continue;
        }
        items.push(Item {
            buf: t.buffer(),
            m: shape[0] as usize,
            k: shape[1] as usize,
            bytes,
        });
    }
    if items.is_empty() {
        eprintln!("gemv-entries: no {} tensors matched", a.kind);
        return 1;
    }
    let total: usize = items.iter().map(|i| i.bytes).sum();
    let m_max = items.iter().map(|i| i.m).max().unwrap_or(1);
    let k_max = items.iter().map(|i| i.k).max().unwrap_or(1);
    let mut xs: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    for k in items.iter().map(|i| i.k) {
        if xs.contains_key(&k) {
            continue;
        }
        let xf: Vec<f32> = (0..k)
            .map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let b = match Buffer::from_host(&xf) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("gemv-entries: x: {e}");
                return 1;
            }
        };
        xs.insert(k, b);
    }
    // One output buffer per distinct row count: `run_quant_resident` asserts
    // y.len() == m.
    let mut ys: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    for m in items.iter().map(|i| i.m) {
        if ys.contains_key(&m) {
            continue;
        }
        match Buffer::alloc(m.max(1)) {
            Ok(b) => {
                ys.insert(m, b);
            }
            Err(e) => {
                eprintln!("gemv-entries: y: {e}");
                return 1;
            }
        }
    }
    let mut reference: Option<Vec<Vec<u8>>> = None;

    println!(
        "{}: {} tensors, {:.3} GB, m<= {}, k<= {}",
        a.kind,
        items.len(),
        total as f64 / 1e9,
        m_max,
        k_max
    );
    for (ei, entry) in a.entries.iter().enumerate() {
        // Exactness: every tensor's output bits against the first entry.
        let mut bad = 0usize;
        let mut idx = 0usize;
        let mut outs: Vec<Vec<u8>> = Vec::new();
        if a.check {
            for it in &items {
                if let Err(e) =
                    l.run_quant_resident(entry, it.buf, &xs[&it.k], &ys[&it.m], it.m, it.k)
                {
                    eprintln!("gemv-entries: {entry}: {e}");
                    return 1;
                }
                if let Err(e) = l.synchronize() {
                    eprintln!("gemv-entries: {e}");
                    return 1;
                }
                let got = match ys[&it.m].download_bytes(0, it.m * 4) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("gemv-entries: {e}");
                        return 1;
                    }
                };
                if ei == 0 {
                    outs.push(got);
                } else if let Some(r) = &reference {
                    if r[idx] != got {
                        bad += 1;
                    }
                }
                idx += 1;
            }
            if ei == 0 {
                reference = Some(outs);
            }
        }

        // Warm then best-of-reps, one event span per launch.
        for it in &items {
            if let Err(e) = l.run_quant_resident(entry, it.buf, &xs[&it.k], &ys[&it.m], it.m, it.k)
            {
                eprintln!("gemv-entries: {entry}: {e}");
                return 1;
            }
        }
        if let Err(e) = l.synchronize() {
            eprintln!("gemv-entries: {e}");
            return 1;
        }
        let mut marks = Vec::with_capacity(items.len());
        for _ in 0..items.len() {
            match (Event::new(), Event::new()) {
                (Ok(x), Ok(yy)) => marks.push((x, yy)),
                _ => {
                    eprintln!("gemv-entries: events");
                    return 1;
                }
            }
        }
        let mut best = f64::MAX;
        let mut best_rows: Vec<(usize, usize, f64)> = Vec::new();
        for _ in 0..a.reps {
            for (idx, it) in items.iter().enumerate() {
                if let Err(e) = marks[idx].0.record(l.stream()) {
                    eprintln!("gemv-entries: {e}");
                    return 1;
                }
                if let Err(e) =
                    l.run_quant_resident(entry, it.buf, &xs[&it.k], &ys[&it.m], it.m, it.k)
                {
                    eprintln!("gemv-entries: {entry}: {e}");
                    return 1;
                }
                if let Err(e) = marks[idx].1.record(l.stream()) {
                    eprintln!("gemv-entries: {e}");
                    return 1;
                }
            }
            if let Err(e) = l.synchronize() {
                eprintln!("gemv-entries: {e}");
                return 1;
            }
            let mut rows = Vec::with_capacity(items.len());
            let mut sum = 0.0f64;
            for (idx, it) in items.iter().enumerate() {
                let ms = match marks[idx].0.since(&marks[idx].1) {
                    Ok(v) => v as f64,
                    Err(e) => {
                        eprintln!("gemv-entries: {e}");
                        return 1;
                    }
                };
                sum += ms;
                rows.push((it.m, it.bytes, ms));
            }
            if sum < best {
                best = sum;
                best_rows = rows;
            }
        }
        println!(
            "\n{entry}: {:.3} ms, {:.1} GB/s{}",
            best,
            total as f64 / (best / 1e3) / 1e9,
            if a.check {
                format!(", {} of {} tensors differ", bad, items.len())
            } else {
                String::new()
            }
        );
        // Group by row count: the grid size depends on the entry's rows per
        // block, and the shapes differ by 3.4x, so a single aggregate hides
        // which shape moved.
        let rows = match hipml_kernels::spec(entry).map(|s| s.kind) {
            Some(hipml_kernels::KernelKind::RowTile { rows, .. }) => rows,
            Some(hipml_kernels::KernelKind::PerRow { .. }) => 1,
            _ => 16,
        };
        let mut agg: std::collections::BTreeMap<usize, (usize, f64, usize)> =
            std::collections::BTreeMap::new();
        for (m, bytes, ms) in best_rows {
            let e = agg.entry(m).or_insert((0, 0.0, 0));
            e.0 += bytes;
            e.1 += ms;
            e.2 += 1;
        }
        for (m, (bytes, ms, n)) in agg {
            println!(
                "  m={:<7} {:3} tensors {:6.3} GB {:8.3} ms {:6.1} GB/s  ({} blocks)",
                m,
                n,
                bytes as f64 / 1e9,
                ms,
                bytes as f64 / (ms / 1e3) / 1e9,
                m.div_ceil(rows as usize)
            );
        }
        let _ = Instant::now();
    }
    0
}
