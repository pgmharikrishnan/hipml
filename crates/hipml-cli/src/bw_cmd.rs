// SPDX-License-Identifier: Apache-2.0
//! `hipml bw-probe`: where the bandwidth wall is for a small read.
//!
//! The decode stream's rate rises with the size of the read: 540 MB runs at
//! 571 GB/s, one routed bank (4.7 MB) at 366, one shared-expert GEMV (0.59 MB)
//! at 125. A rate that rises monotonically with size is the signature of too
//! few bytes in flight, not of a bandwidth limit, so this command separates the
//! three levers that change in-flight bytes and nothing else:
//!
//!   read size        --sizes, in bytes, from one expert row tile to `lm_head`
//!   bytes per load   --vecs 4 / 8 / 16
//!   outstanding      --outs 1 / 2 / 4 / 8 independent loads per thread
//!   workgroups       --blocks, plus the GEMV's own row-tile geometry with
//!                    --warps/--rpw so 512 rows can be 512 blocks of one row or
//!                    16 blocks of 32 rows
//!
//! Each cell is `--launches` back-to-back launches of one kernel with a single
//! event pair around the burst, so the per-launch figure is a burst average and
//! not an event-timing artefact. The same grid is also timed with an empty
//! kernel; that node cost is subtracted to give the kernel's own rate. Both
//! numbers are printed: `raw` is what one launch of this shape delivers, `net`
//! is what the memory system does once the dispatch is paid for.
//!
//! Small reads are rotated over several distinct windows of one allocation, so
//! a repeated launch never re-reads bytes still resident in L2. A rate measured
//! out of cache would not be the rate the decode stream sees.

use std::io::Write;

use hipml_hip::{Buffer, ByteBuffer, Event};
use hipml_kernels::Launcher;

const SIZES: &[(&str, usize)] = &[
    ("gate_up_q4k", 589_824),
    ("down_q5k", 720_896),
    ("down_q6k", 860_160),
    ("router_f32", 2_097_152),
    ("one_bank", 4_718_592),
    ("layer_banks", 18_874_368),
    ("lm_head", 540_344_320),
];

const VECS: &[usize] = &[4, 8, 16];
const OUTS: &[usize] = &[1, 2, 4, 8];
const BLOCKS: &[usize] = &[32, 64, 128, 256, 512, 1024, 4096];

/// (rows, row_bytes, label) for the GEMV's own shape: one expert's gate/up and
/// one expert's down, the two row geometries the MoE stream actually issues.
const SHAPES: &[(usize, usize, &str)] = &[
    (512, 1152, "gate_up_512x2048_q4k"),
    (2048, 352, "down_2048x512_q5k"),
];

/// (warps, rows per warp) -> rows per block. The first four are "one workgroup
/// per row" and up; the last two are the 8-row and 16-row tiles the engine uses.
const GEOMS: &[(usize, usize)] = &[(1, 1), (2, 1), (4, 1), (8, 1), (8, 2), (8, 4)];

struct Args {
    reps: usize,
    launches: usize,
    threads: usize,
    csv: Option<String>,
    flat_only: bool,
    rows_only: bool,
    sizes: Vec<usize>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        reps: 3,
        launches: 128,
        threads: 256,
        csv: None,
        flat_only: false,
        rows_only: false,
        sizes: (0..SIZES.len()).collect(),
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--reps" => {
                a.reps = need(i)?.parse().map_err(|e| format!("bad --reps: {e}"))?;
                i += 2;
            }
            "--launches" => {
                a.launches = need(i)?
                    .parse()
                    .map_err(|e| format!("bad --launches: {e}"))?;
                i += 2;
            }
            "--threads" => {
                a.threads = need(i)?
                    .parse()
                    .map_err(|e| format!("bad --threads: {e}"))?;
                i += 2;
            }
            "--csv" => {
                a.csv = Some(need(i)?.to_string());
                i += 2;
            }
            "--sizes" => {
                let mut v = Vec::new();
                for s in need(i)?.split(',') {
                    let n: usize = s.parse().map_err(|e| format!("bad --sizes: {e}"))?;
                    let idx = SIZES
                        .iter()
                        .position(|(_, b)| *b == n)
                        .ok_or_else(|| format!("unknown size {n}"))?;
                    v.push(idx);
                }
                a.sizes = v;
                i += 2;
            }
            "--flat-only" => {
                a.flat_only = true;
                i += 1;
            }
            "--rows-only" => {
                a.rows_only = true;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.launches < 8 {
        return Err("--launches must be at least 8".to_string());
    }
    Ok(a)
}

/// A burst has a fixed cost (~80 us of pipeline restart on this runtime), so a
/// short burst prices that cost into every launch. Two things follow:
///
///   * a burst must be long enough that the fixed part is a few per cent;
///   * a burst must read more distinct bytes than the 64 MB Infinity Cache
///     holds, and the window cursor must keep advancing between reps, or the
///     best-of-three rep is the one served out of cache. A rate measured out of
///     cache is not the rate the decode stream sees.
const ALLOC: usize = 2 * 1024 * 1024 * 1024;
/// Distinct bytes one burst must touch.
const BURST_BYTES: usize = 256 * 1024 * 1024;

/// (launches, distinct windows) for one read size.
fn plan(bytes: usize, requested: usize) -> (usize, usize) {
    let copies = (ALLOC / bytes).clamp(1, 8192);
    let need = (BURST_BYTES / bytes).max(1);
    (requested.max(need).min(2048), copies)
}

/// The GPU clock's own per-launch cost: one event pair around `launches`
/// back-to-back launches, best of `reps`.
fn burst<F>(
    launches: usize,
    reps: usize,
    ev: &(Event, Event),
    l: &Launcher,
    mut f: F,
) -> Result<f64, String>
where
    F: FnMut(usize) -> Result<(), String>,
{
    let mut best = f64::MAX;
    for _ in 0..reps {
        ev.0.record(l.stream())?;
        for n in 0..launches {
            f(n)?;
        }
        ev.1.record(l.stream())?;
        l.synchronize()?;
        let us = ev.0.since(&ev.1)? as f64 * 1e3;
        if us < best {
            best = us;
        }
    }
    Ok(best / launches as f64)
}

fn gbps(bytes: usize, us: f64) -> f64 {
    if us <= 0.0 {
        return 0.0;
    }
    bytes as f64 / (us / 1e6) / 1e9
}

/// The rate above the dispatch floor, or `-` when the read is entirely hidden
/// under it: a span at or below the empty-kernel cost says nothing about the
/// memory system, and printing a rate there would be dividing by noise.
fn rate(bytes: usize, excess: f64) -> String {
    if excess < 0.05 {
        "-".to_string()
    } else {
        format!("{:.1}", gbps(bytes, excess))
    }
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bw-probe: {e}");
            return 2;
        }
    };
    let l = match Launcher::new() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bw-probe: {e}");
            return 1;
        }
    };
    println!(
        "kernel arch: {}\nprobe: up to {} launches x {} reps, {} threads, node cost subtracted",
        hipml_kernels::kernel_arch(),
        a.launches,
        a.reps,
        a.threads
    );

    // One allocation, several windows: a repeated launch must not re-read bytes
    // still resident in the 64 MB Infinity Cache.
    let alloc = ALLOC;
    let buf = match ByteBuffer::alloc(alloc) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bw-probe: alloc {alloc} B: {e}");
            return 1;
        }
    };
    let out = match Buffer::alloc(BLOCKS.iter().copied().max().unwrap_or(4096) * 4 + 64) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bw-probe: out alloc: {e}");
            return 1;
        }
    };
    let ev = match (Event::new(), Event::new()) {
        (Ok(x), Ok(y)) => (x, y),
        _ => {
            eprintln!("bw-probe: could not create events");
            return 1;
        }
    };

    // Bring the card to its running clock before anything is timed: a burst
    // measured from idle reads a lower clock than a decode token sees.
    let warm = plan(SIZES[0].1, a.launches).0;
    for _ in 0..3 {
        for n in 0..warm {
            if let Err(e) = l.bw_flat(
                "bw_flat_v16_o4",
                &buf,
                SIZES[5].1,
                (n % 8) * SIZES[5].1,
                16,
                256,
                a.threads,
                &out,
            ) {
                eprintln!("bw-probe: warm-up: {e}");
                return 1;
            }
        }
    }
    if let Err(e) = l.synchronize() {
        eprintln!("bw-probe: {e}");
        return 1;
    }

    let mut rows: Vec<String> = Vec::new();
    rows.push(
        "kind,label,bytes,vec,outstanding,blocks,threads,floor_us,span_us,excess_us,gbps_raw,gbps_net,lim"
            .to_string(),
    );

    // The node cost at every grid this run uses, so a 0.9 us kernel is not
    // priced with the dispatch it never pays alone.
    let node_launches = a.launches.max(1024);
    let mut node: std::collections::BTreeMap<(usize, usize), f64> =
        std::collections::BTreeMap::new();
    for &blocks in BLOCKS {
        if let Ok(us) = burst(node_launches, a.reps, &ev, &l, |_| {
            l.bw_empty(blocks, a.threads, &out)
        }) {
            node.insert((blocks, a.threads), us);
        }
    }
    // How fast the host can queue a launch at all, with no synchronize in the
    // loop. A span equal to this is the host, not the GPU, and a smaller read
    // cannot be timed by queuing it faster than the host issues.
    let host_us = {
        let mut best = f64::MAX;
        for _ in 0..a.reps {
            let t = std::time::Instant::now();
            for _ in 0..node_launches {
                if let Err(e) = l.bw_empty(32, a.threads, &out) {
                    eprintln!("bw-probe: {e}");
                    return 1;
                }
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / node_launches as f64;
            if us < best {
                best = us;
            }
        }
        if let Err(e) = l.synchronize() {
            eprintln!("bw-probe: {e}");
            return 1;
        }
        best
    };
    println!(
        "\nhost issue rate: {host_us:.3} us/launch (queue only, no sync); \
         empty-kernel GPU span at 32 blocks: {:.3} us",
        node.get(&(32, a.threads)).copied().unwrap_or(0.0)
    );
    for &(warps, rpw) in GEOMS {
        for &(nrows, _, _) in SHAPES {
            let threads = warps * 32;
            let blocks = nrows.div_ceil(warps * rpw);
            if !node.contains_key(&(blocks, threads)) {
                if let Ok(us) = burst(node_launches, a.reps, &ev, &l, |_| {
                    l.bw_empty(blocks, threads, &out)
                }) {
                    node.insert((blocks, threads), us);
                }
            }
        }
    }

    let floor_of = |blocks: usize,
                    threads: usize,
                    node: &std::collections::BTreeMap<(usize, usize), f64>|
     -> f64 {
        node.get(&(blocks, threads))
            .copied()
            .unwrap_or(0.0)
            .max(host_us)
    };

    if !a.rows_only {
        println!("\n=== flat: read size x vector width x outstanding loads x workgroups ===");
        println!(
            "{:<13} {:>10} {:>3} {:>4} {:>6} {:>8} {:>8} {:>8} {:>9} {:>9} {:>5}",
            "size",
            "bytes",
            "vec",
            "out",
            "blocks",
            "floor us",
            "span us",
            "excess",
            "raw GB/s",
            "net GB/s",
            "lim"
        );
        for &i in &a.sizes {
            let (label, bytes) = SIZES[i];
            let (launches, w) = plan(bytes, a.launches);
            let mut cursor = 0usize;
            for &vec in VECS {
                for &outstanding in OUTS {
                    let entry = format!("bw_flat_v{vec}_o{outstanding}");
                    for &blocks in BLOCKS {
                        let span = match burst(launches, a.reps, &ev, &l, |_| {
                            let base = (cursor % w) * bytes;
                            cursor += 1;
                            l.bw_flat(&entry, &buf, bytes, base, vec, blocks, a.threads, &out)
                        }) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("bw-probe: {entry}: {e}");
                                return 1;
                            }
                        };
                        let floor = floor_of(blocks, a.threads, &node);
                        let excess = span - floor;
                        let lim = if span < host_us * 1.10 { "host" } else { "gpu" };
                        println!(
                            "{:<13} {:>10} {:>3} {:>4} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>9.1} {:>9} {:>5}",
                            label,
                            bytes,
                            vec,
                            outstanding,
                            blocks,
                            floor,
                            span,
                            excess.max(0.0),
                            gbps(bytes, span),
                            rate(bytes, excess),
                            lim
                        );
                        rows.push(format!(
                            "flat,{label},{bytes},{vec},{outstanding},{blocks},{},{floor:.4},{span:.4},{:.4},{:.1},{},{lim}",
                            a.threads,
                            excess.max(0.0),
                            gbps(bytes, span),
                            rate(bytes, excess)
                        ));
                    }
                }
            }
        }
    }

    if !a.flat_only {
        println!("\n=== rows: the GEMV's own geometry, one workgroup per row and up ===");
        println!(
            "{:<23} {:>3} {:>5} {:>3} {:>6} {:>8} {:>8} {:>8} {:>9} {:>9} {:>5}",
            "shape",
            "vec",
            "warps",
            "rpw",
            "blocks",
            "floor us",
            "span us",
            "excess",
            "raw GB/s",
            "net GB/s",
            "lim"
        );
        for &(nrows, row_bytes, label) in SHAPES {
            let bytes = nrows * row_bytes;
            let (launches, w) = plan(bytes, a.launches);
            let mut cursor = 0usize;
            for &vec in VECS {
                let entry = format!("bw_rows_v{vec}");
                for &(warps, rpw) in GEOMS {
                    let threads = warps * 32;
                    let blocks = nrows.div_ceil(warps * rpw);
                    let floor = floor_of(blocks, threads, &node);
                    let span = match burst(launches, a.reps, &ev, &l, |_| {
                        let row0 = (cursor % w) * nrows;
                        cursor += 1;
                        l.bw_rows(
                            &entry, &buf, nrows, row_bytes, rpw, row0, blocks, threads, &out,
                        )
                    }) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("bw-probe: {entry}: {e}");
                            return 1;
                        }
                    };
                    let excess = span - floor;
                    let lim = if span < host_us * 1.10 { "host" } else { "gpu" };
                    println!(
                        "{:<23} {:>3} {:>5} {:>3} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>9.1} {:>9} {:>5}",
                        label,
                        vec,
                        warps,
                        rpw,
                        blocks,
                        floor,
                        span,
                        excess.max(0.0),
                        gbps(bytes, span),
                        rate(bytes, excess),
                        lim
                    );
                    rows.push(format!(
                        "rows,{label},{bytes},{vec},-,{blocks},{threads},{floor:.4},{span:.4},{:.4},{:.1},{},{lim}",
                        excess.max(0.0),
                        gbps(bytes, span),
                        rate(bytes, excess)
                    ));
                }
            }
        }
    }

    if let Some(path) = &a.csv {
        match std::fs::File::create(path) {
            Ok(mut f) => {
                let _ = f.write_all(rows.join("\n").as_bytes());
                let _ = f.write_all(b"\n");
                println!("\ncsv: {path} ({} rows)", rows.len() - 1);
            }
            Err(e) => eprintln!("bw-probe: csv {path}: {e}"),
        }
    }
    0
}
