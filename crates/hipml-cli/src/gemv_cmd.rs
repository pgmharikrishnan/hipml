// SPDX-License-Identifier: Apache-2.0
//! `hipml gemv-check` / `hipml gemv-bench`: the batched GEMV's exactness and
//! cost against `R` single launches, per resident weight format.

use hipml_model::Model;
/// `hipml gemv-check --gguf PATH --tensor NAME [--rows R]`: prove the batched
/// GEMV is bit-identical to `R` single launches.
pub fn gemv_check(args: &[String]) -> i32 {
    let mut gguf = String::new();
    let mut tensor = String::new();
    let mut rows = 2usize;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--gguf" => {
                gguf = need(i);
                i += 2;
            }
            "--tensor" => {
                tensor = need(i);
                i += 2;
            }
            "--rows" => {
                rows = need(i).parse().unwrap_or(2);
                i += 2;
            }
            other => {
                eprintln!("gemv-check: unknown argument {other}");
                return 2;
            }
        }
    }
    if gguf.is_empty() || tensor.is_empty() {
        eprintln!("gemv-check: --gguf and --tensor are required");
        return 2;
    }
    let (model, _) = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gemv-check: load: {e}");
            return 1;
        }
    };
    match hipml_model::gpu_weights::check_batched_gemv(&model, &tensor, rows, 0x9E37_79B9) {
        Ok((n, diff, worst)) => {
            println!(
                "{tensor}: {rows} columns, {n} values, {diff} differ, max |d| {worst:.3e}"
            );
            if diff == 0 {
                0
            } else {
                1
            }
        }
        Err(e) => {
            eprintln!("gemv-check: {e}");
            1
        }
    }
}

/// `hipml gemv-bench --gguf PATH --tensor NAME [--rows R] [--reps N]`: time the
/// batched GEMV against `rows` single launches, per format and row count.
pub fn gemv_bench(args: &[String]) -> i32 {
    let mut gguf = String::new();
    let mut tensors: Vec<String> = Vec::new();
    let mut max_rows = 4usize;
    let mut reps = 20usize;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--gguf" => {
                gguf = need(i);
                i += 2;
            }
            "--tensor" => {
                tensors.push(need(i));
                i += 2;
            }
            "--rows" => {
                max_rows = need(i).parse().unwrap_or(4);
                i += 2;
            }
            "--reps" => {
                reps = need(i).parse().unwrap_or(20).max(1);
                i += 2;
            }
            other => {
                eprintln!("gemv-bench: unknown argument {other}");
                return 2;
            }
        }
    }
    if gguf.is_empty() || tensors.is_empty() {
        eprintln!("gemv-bench: --gguf and at least one --tensor are required");
        return 2;
    }
    let (model, _) = match Model::load(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("gemv-bench: load: {e}");
            return 1;
        }
    };
    println!(
        "{:<24} {:>8} {:>4} {:>10} {:>10} {:>7} {:>8} {:>8}",
        "tensor", "kind", "rows", "single ms", "batch ms", "ratio", "GB/s", "tok/s"
    );
    let mut failed = false;
    for name in &tensors {
        match hipml_model::gpu_weights::bench_batched_gemv(&model, name, max_rows, reps) {
            Ok(ts) => {
                for t in ts {
                    println!(
                        "{:<24} {:>8} {:>4} {:>10.3} {:>10.3} {:>6.2}x {:>8.1} {:>8.1}",
                        t.name,
                        t.kind,
                        t.rows,
                        t.single_ms,
                        t.batched_ms,
                        t.ratio(),
                        t.batched_gbs(),
                        t.rows as f64 * 1e3 / t.batched_ms.max(1e-9),
                    );
                }
            }
            Err(e) => {
                eprintln!("gemv-bench: {e}");
                failed = true;
            }
        }
    }
    if failed {
        1
    } else {
        0
    }
}

