//! CLI commands added for the KV work: VRAM arithmetic and the depth bench.
//!
//! Split from `main.rs` by size. Both are thin: the launch path lives in
//! `hipml-kernels`, and these only read `hipMemGetInfo` and print a table.

/// `hipml vram`: current `hipMemGetInfo`, and with `--need BYTES` whether a
/// cache of that size fits right now. No kernel, so it is safe to run beside
/// another agent's job when only the arithmetic is wanted.
pub fn vram(args: &[String]) {
    let need = args
        .iter()
        .position(|a| a == "--need")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok());

    let devices = match hipml_hip::Device::probe() {
        Ok(d) => d,
        Err(e) => {
            println!("device: none ({e})");
            return;
        }
    };
    if devices.is_empty() {
        println!("device: none found");
        return;
    }
    println!(
        "{:<4} {:<34} {:>10} {:>10} {:>10}",
        "dev", "name", "total_GiB", "free_GiB", "used_GiB"
    );
    for d in &devices {
        #[cfg(feature = "hip")]
        let (free, total) = d.mem_info().unwrap_or((0, d.total_mem_bytes()));
        #[cfg(not(feature = "hip"))]
        let (free, total) = (0usize, d.total_mem_bytes());
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{:<4} {:<34} {:>10.2} {:>10.2} {:>10.2}",
            d.ordinal(),
            d.name(),
            gib(total),
            gib(free),
            gib(total.saturating_sub(free))
        );
        if let Some(n) = need {
            let fits = (n as usize) <= free;
            println!(
                "     need {} bytes ({:.2} GiB): {}",
                n,
                gib(n as usize),
                if fits { "fits" } else { "DOES NOT FIT" }
            );
        }
    }
}

/// `hipml bench-kv`: quantized decode attention over the bench depths.
pub fn bench_kv(args: &[String]) -> i32 {
    let depths: Vec<usize> = args
        .iter()
        .position(|a| a == "--depths")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1024, 8192, 32768, 100_000]);
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);

    #[cfg(feature = "hip")]
    {
        let report = match hipml_kernels::kvbench::run(&depths, reps) {
            Ok(r) => r,
            Err(e) => {
                println!("bench-kv failed: {e}");
                return 1;
            }
        };
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("device: {}", report.device);
        println!(
            "VRAM: total {:.2} GiB, free before {:.2} GiB, free after {:.2} GiB",
            gib(report.vram_total),
            gib(report.vram_free_before),
            gib(report.vram_free_after)
        );
        println!(
            "weights stream: {:.1} GB/s -> {:.2} ms for 17.19 GB",
            report.weights_gbs,
            report.weights_us / 1e3
        );
        println!("cache bytes are synthetic but format-valid; 100K correctness is `hipml parity --deep`\n");
        println!(
            "{:<5} {:>5} {:>6} {:>7} {:>6} {:>11} {:>9} {:>8} {:>8} {:>7}",
            "dtype", "heads", "layers", "depth", "splits", "us", "KV_GiB", "GB/s", "share", "tok/s"
        );
        for r in &report.rows {
            let step = report.weights_us + r.us;
            let (share, tps) = if r.layers > 1 {
                (
                    format!("{:.1}%", 100.0 * r.us / step),
                    format!("{:.2}", 1e6 / step),
                )
            } else {
                ("-".to_string(), "-".to_string())
            };
            println!(
                "{:<5} {:>5} {:>6} {:>7} {:>6} {:>11.1} {:>9.3} {:>8.1} {:>8} {:>7}",
                r.dtype.name(),
                r.heads,
                r.layers,
                r.seq,
                r.splits,
                r.us,
                r.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                r.gbs,
                share,
                tps
            );
        }
        0
    }
    #[cfg(not(feature = "hip"))]
    {
        let _ = (depths, reps);
        println!("built without the `hip` feature");
        1
    }
}
