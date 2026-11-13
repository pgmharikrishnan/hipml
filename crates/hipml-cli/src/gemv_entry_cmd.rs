// SPDX-License-Identifier: Apache-2.0
//! `hipml gemv-entries`: the whole-type GEMV rate for several entries over the
//! same resident tensors, with a bit-exactness check against the production one.
//!
//! One launch per tensor, one synchronize per pass, best of `--reps`: the same
//! accounting as `gpu-sweep --gemv`, so a candidate's number is directly
//! comparable with the production table. The first entry named is the
//! reference; every other entry must produce the same output bits on every
//! tensor of the kind or it is reported as differing.
//!
//! `--batch N` measures the *batched* path instead: the same whole-type stream
//! through `Launcher::gemv_batched` with `N` token rows, which is what the
//! verify's `R`-column pass launches, and checks every row against the
//! single-sequence entry bit for bit. The weight bytes do not change with `N`,
//! so the GB/s column read at `N` against the one-row number is the row-scaling
//! cost of the batched GEMV.

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
    /// 0 = the single-sequence entries named in `--entries`; `n > 0` = the
    /// batched path at `n` rows.
    batch: usize,
}

/// One resident 2-D quantized tensor of the measured kind.
struct Item<'a> {
    buf: &'a hipml_hip::ByteBuffer,
    m: usize,
    k: usize,
    bytes: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        kind: String::new(),
        entries: Vec::new(),
        reps: 3,
        check: true,
        batch: 0,
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
            "--batch" => {
                a.batch = need(i)?.parse().map_err(|e| format!("bad --batch: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() || a.kind.is_empty() || (a.entries.is_empty() && a.batch == 0) {
        return Err("--gguf, --kind and --entries (or --batch) are required".to_string());
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
    if a.batch > 0 {
        return run_batched(&a, &model, &items, total, ty);
    }
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
        // Wall time of the same launches: the event spans below exclude the
        // gap between one kernel finishing and the next starting, so the
        // difference is that gap and nothing else.
        let mut best_wall = f64::MAX;
        for _ in 0..a.reps {
            let wall0 = Instant::now();
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
            let wall = wall0.elapsed().as_secs_f64() * 1e3;
            if wall < best_wall {
                best_wall = wall;
            }
            if sum < best {
                best = sum;
                best_rows = rows;
            }
        }
        println!(
            "\n{entry}: {:.3} ms spans, {:.3} ms wall ({:+.3} gap), {:.1} GB/s spans, {:.1} GB/s wall{}",
            best,
            best_wall,
            best_wall - best,
            total as f64 / (best / 1e3) / 1e9,
            total as f64 / (best_wall / 1e3) / 1e9,
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

/// The batched path at `a.batch` token rows: one launch per tensor of the kind
/// with `B` x rows and one weight pass, exactly as the verify issues it.
fn run_batched(a: &Args, model: &Model, items: &[Item<'_>], total: usize, ty: TensorType) -> i32 {
    let l = model.launcher();
    let b = a.batch;
    let name = match hipml_model::gpu_weights::gemv_entry(ty) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("gemv-entries: {e}");
            return 2;
        }
    };
    let kind = match hipml_kernels::batch_kind_entry(name) {
        Some(k) => k,
        None => {
            eprintln!("gemv-entries: {name} has no batched GEMV");
            return 1;
        }
    };
    // What the `R = 1` verify launches, so the exactness contract is "the
    // batched row equals the single-sequence entry", not "equals its sibling".
    let single = hipml_kernels::stream_entry(name);

    let val = |i: usize, bi: usize| ((i * 37 + bi * 11) % 101) as f32 * 0.01 - 0.5;

    let mut xs: std::collections::HashMap<usize, Vec<Buffer>> = std::collections::HashMap::new();
    let mut xall: std::collections::HashMap<usize, Buffer> = std::collections::HashMap::new();
    for &k in items.iter().map(|i| i.k).collect::<Vec<_>>().iter() {
        if xall.contains_key(&k) {
            continue;
        }
        let mut all = Vec::with_capacity(b * k);
        for bi in 0..b {
            for i in 0..k {
                all.push(val(i, bi));
            }
        }
        let mut rows = Vec::with_capacity(b);
        for bi in 0..b {
            match Buffer::from_host(&all[bi * k..(bi + 1) * k]) {
                Ok(x) => rows.push(x),
                Err(e) => {
                    eprintln!("gemv-entries: x: {e}");
                    return 1;
                }
            }
        }
        match Buffer::from_host(&all) {
            Ok(x) => {
                xall.insert(k, x);
            }
            Err(e) => {
                eprintln!("gemv-entries: x: {e}");
                return 1;
            }
        }
        xs.insert(k, rows);
    }
    let mut ys: std::collections::HashMap<usize, (Buffer, Vec<Buffer>)> =
        std::collections::HashMap::new();
    for &m in items.iter().map(|i| i.m).collect::<Vec<_>>().iter() {
        if ys.contains_key(&m) {
            continue;
        }
        let mut rows = Vec::with_capacity(b);
        for _ in 0..b {
            match Buffer::alloc(m) {
                Ok(y) => rows.push(y),
                Err(e) => {
                    eprintln!("gemv-entries: y: {e}");
                    return 1;
                }
            }
        }
        match Buffer::alloc(b * m) {
            Ok(y) => {
                ys.insert(m, (y, rows));
            }
            Err(e) => {
                eprintln!("gemv-entries: y: {e}");
                return 1;
            }
        }
    }

    println!(
        "{} b={}: {} tensors, {:.3} GB, m<= {}, k<= {}",
        a.kind,
        b,
        items.len(),
        total as f64 / 1e9,
        items.iter().map(|i| i.m).max().unwrap_or(1),
        items.iter().map(|i| i.k).max().unwrap_or(1)
    );

    // Exactness: every batched row against the single-sequence entry on the
    // same x row.
    if a.check {
        let mut bad = 0usize;
        for it in items {
            let (yb, _) = &ys[&it.m];
            if let Err(e) = l.gemv_batched(kind, it.buf, &xall[&it.k], yb, it.m, it.k, b) {
                eprintln!("gemv-entries: b={b}: {e}");
                return 1;
            }
            if let Err(e) = l.synchronize() {
                eprintln!("gemv-entries: {e}");
                return 1;
            }
            let got = match yb.download_bytes(0, b * it.m * 4) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("gemv-entries: {e}");
                    return 1;
                }
            };
            for bi in 0..b {
                let (_, yr) = &ys[&it.m];
                if let Err(e) =
                    l.run_quant_resident(single, it.buf, &xs[&it.k][bi], &yr[bi], it.m, it.k)
                {
                    eprintln!("gemv-entries: {single}: {e}");
                    return 1;
                }
                if let Err(e) = l.synchronize() {
                    eprintln!("gemv-entries: {e}");
                    return 1;
                }
                let want = match yr[bi].download_bytes(0, it.m * 4) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("gemv-entries: {e}");
                        return 1;
                    }
                };
                if got[bi * it.m * 4..(bi + 1) * it.m * 4] != want[..] {
                    bad += 1;
                }
            }
        }
        println!("  rows: {}/{} differ from {single}", bad, b * items.len());
    }

    for it in items {
        if let Err(e) = l.gemv_batched(kind, it.buf, &xall[&it.k], &ys[&it.m].0, it.m, it.k, b) {
            eprintln!("gemv-entries: b={b}: {e}");
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
            (Ok(x), Ok(y)) => marks.push((x, y)),
            _ => {
                eprintln!("gemv-entries: events");
                return 1;
            }
        }
    }
    let mut best = f64::MAX;
    let mut best_wall = f64::MAX;
    let mut best_rows: Vec<(usize, usize, f64)> = Vec::new();
    for _ in 0..a.reps {
        let wall0 = Instant::now();
        for (idx, it) in items.iter().enumerate() {
            if let Err(e) = marks[idx].0.record(l.stream()) {
                eprintln!("gemv-entries: {e}");
                return 1;
            }
            if let Err(e) = l.gemv_batched(kind, it.buf, &xall[&it.k], &ys[&it.m].0, it.m, it.k, b)
            {
                eprintln!("gemv-entries: b={b}: {e}");
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
        let wall = wall0.elapsed().as_secs_f64() * 1e3;
        if wall < best_wall {
            best_wall = wall;
        }
        if sum < best {
            best = sum;
            best_rows = rows;
        }
    }
    println!(
        "\n{name}_b b={b}: {:.3} ms spans, {:.3} ms wall ({:+.3} gap, {} launches, {:.2} us each), {:.1} GB/s spans, {:.1} GB/s wall",
        best,
        best_wall,
        best_wall - best,
        items.len(),
        (best_wall - best) * 1e3 / items.len() as f64,
        total as f64 / (best / 1e3) / 1e9,
        total as f64 / (best_wall / 1e3) / 1e9
    );
    // Same 16-row block the `_b` kernels use, so the grid column is the real one.
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
            m.div_ceil(16)
        );
    }
    0
}
