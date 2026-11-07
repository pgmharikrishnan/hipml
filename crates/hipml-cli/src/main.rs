//! `hipml` — probe the device, list the kernel manifest, run the parity suite.
//!
//! The parity runner is the gate: a kernel that is not compared against the
//! reference here is not a kernel hipml can use.

use hipml_core::parity::{bench_cases, cases, compare, deep_cases, OpCase};
use hipml_core::tensor::Tensor;
use hipml_kernels::{kernel_arch, kernel_dir, KernelKind, KERNELS};

mod kv_cmds;

/// What happened to one case.
enum Outcome {
    Pass(String),
    Fail(String),
    Skip(String),
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");

    match cmd {
        "probe" => probe(),
        "kernels" => kernels(),
        "vram" => kv_cmds::vram(&args[1..]),
        "bench" => std::process::exit(bench()),
        "bench-kv" => std::process::exit(kv_cmds::bench_kv(&args[1..])),
        "parity" => {
            let filter = args
                .iter()
                .skip(1)
                .find(|a| !a.starts_with('-'))
                .cloned();
            let deep = args.iter().any(|a| a == "--deep");
            std::process::exit(parity(filter.as_deref(), deep));
        }
        "help" | "--help" | "-h" => help(),
        other => {
            eprintln!("unknown command: {other}\n");
            help();
            std::process::exit(2);
        }
    }
}

fn help() {
    println!(
        "hipml {version}\n\
         \n\
         usage: hipml <command>\n\
         \n\
         commands:\n\
         \x20 probe     show the HIP device and how this binary was built\n\
         \x20 kernels   list the kernel manifest\n\
         \x20 vram      hipMemGetInfo now; --need BYTES also reports whether a cache fits\n\
         \x20 parity    run every parity case (reference vs device); optional name filter\n\
         \x20           --deep adds the 100K-context cases (slow host reference)\n\
         \x20 bench     time each kernel launch and report effective bandwidth\n\
         \x20 bench-kv  quantized-KV decode attention: depth x q8/q4 x 1 head / 16 layers\n\
         \x20           --depths 1024,8192,32768,100000 --reps 3\n\
         \n\
         build with `--features hip` on a machine with the ROCm/HIP runtime.",
        version = env!("CARGO_PKG_VERSION")
    );
}

fn probe() {
    println!("hipml {}", env!("CARGO_PKG_VERSION"));
    println!("built with hip feature: {}", hipml_hip::HAS_HIP);
    println!("kernel arch:            {}", kernel_arch());
    match kernel_dir() {
        Some(dir) => println!("code object:            {}", dir.join("hipml.co").display()),
        None => println!("code object:            none (set HIP_PATH and rebuild)"),
    }
    match hipml_hip::Device::probe() {
        Ok(devices) if devices.is_empty() => println!("devices:                none found"),
        Ok(devices) => {
            for d in devices {
                let (major, minor) = d.compute_capability();
                println!(
                    "device {}: {} ({}.{}, {:.1} GiB)",
                    d.ordinal(),
                    d.name(),
                    major,
                    minor,
                    d.total_mem_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
                );
            }
        }
        Err(e) => println!("devices:                unavailable ({e})"),
    }
}

fn kernels() {
    println!("{:<20} {:<18} {}", "op", "entry", "geometry");
    for k in KERNELS {
        let geometry = match k.kind {
            KernelKind::PerRow { block } => format!("per-row, {block} threads"),
            KernelKind::PerHead { block } => format!("per-head, {block} threads"),
            KernelKind::PerHeadOnDim => "per-head, one thread per dim".to_string(),
            KernelKind::KvSplit => "per (kv head, split), 32*gqa-group threads".to_string(),
            KernelKind::Linear { block } => format!("linear, {block} threads"),
            KernelKind::HostOnly(why) => format!("host-only: {why}"),
        };
        println!("{:<20} {:<18} {}", k.name, k.entry, geometry);
    }
}

/// Run every case. Exit 0 only when nothing failed; skips are reported but do
/// not fail the run, so the suite stays meaningful on a machine without a GPU.
fn parity(filter: Option<&str>, deep: bool) -> i32 {
    let launcher = hipml_kernels::Launcher::new();
    match &launcher {
        Ok(l) => println!("device: {}", l.device_name),
        Err(e) => println!("device: none ({e})"),
    }

    let mut all = cases();
    if deep {
        all.extend(deep_cases());
    }

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    println!("\n{:<20} {:<7} {}", "case", "result", "detail");

    for case in all {
        if let Some(f) = filter {
            if !case.name.contains(f) {
                continue;
            }
        }
        let (inputs, expected) = case.reference();
        let outcome = run_case(
            &case,
            &inputs,
            &expected,
            launcher.as_ref().map_err(|e| e.as_str()),
        );

        let (verdict, detail) = match outcome {
            Outcome::Pass(d) => {
                pass += 1;
                ("PASS", d)
            }
            Outcome::Fail(d) => {
                fail += 1;
                ("FAIL", d)
            }
            Outcome::Skip(d) => {
                skip += 1;
                ("SKIP", d)
            }
        };
        println!("{:<20} {:<7} {}", case.name, verdict, detail);
    }

    println!("\n{pass} passed, {fail} failed, {skip} skipped");
    if fail > 0 {
        1
    } else {
        0
    }
}

fn run_case(
    case: &OpCase,
    inputs: &[Tensor],
    expected: &[f32],
    launcher: Result<&hipml_kernels::Launcher, &str>,
) -> Outcome {
    let Some(spec) = hipml_kernels::spec(case.name) else {
        return Outcome::Skip("no manifest entry".to_string());
    };
    if let KernelKind::HostOnly(why) = spec.kind {
        return Outcome::Skip(why.to_string());
    }
    let launcher = match launcher {
        Ok(l) => l,
        Err(e) => return Outcome::Skip(e.to_string()),
    };

    match launcher.run(case, inputs) {
        Err(e) => Outcome::Fail(format!("launch failed: {e}")),
        Ok(actual) => match compare(expected, &actual, case.tol) {
            Ok(stats) => Outcome::Pass(format!(
                "max_abs {:.2e}  max_rel {:.2e}",
                stats.max_abs, stats.max_rel
            )),
            Err(stats) => Outcome::Fail(format!(
                "max_abs {:.2e} > {:.2e}, max_rel {:.2e} > {:.2e}",
                stats.max_abs, case.tol.abs, stats.max_rel, case.tol.rel
            )),
        },
    }
}

/// Time each kernel launch on the device.
///
/// Copies are excluded: the same buffers are launched `REPS` times and the
/// launch cost is the difference between one repetition and many, so the number
/// is the kernel, not the transfer.
fn bench() -> i32 {
    const REPS: usize = 51;

    let launcher = match hipml_kernels::Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            println!("device: none ({e})");
            return 1;
        }
    };
    println!("device: {}\n", launcher.device_name);
    println!(
        "{:<20} {:>10} {:>10} {:>12}",
        "case", "launch_us", "GB/s", "bytes"
    );

    let all: Vec<OpCase> = cases().into_iter().chain(bench_cases()).collect();
    for case in all {
        // The quantized-KV cases are timed by `bench-kv`, with depths and layer
        // counts that a single parity-shaped launch cannot show.
        if case.name.starts_with("attn_decode_kv") {
            continue;
        }
        let Some(spec) = hipml_kernels::spec(case.name) else {
            continue;
        };
        if let KernelKind::HostOnly(_) = spec.kind {
            continue;
        }
        let (inputs, _) = case.reference();

        // Warm up: first launch pays module lookup and cache fills.
        if let Err(e) = launcher.run_reps(&case, &inputs, 1) {
            println!("{:<20} {}", case.name, format!("failed: {e}"));
            continue;
        }

        let t1 = match timed(|| launcher.run_reps(&case, &inputs, 1)) {
            Ok(t) => t,
            Err(e) => {
                println!("{:<20} {}", case.name, format!("failed: {e}"));
                continue;
            }
        };
        let tn = match timed(|| launcher.run_reps(&case, &inputs, REPS)) {
            Ok(t) => t,
            Err(e) => {
                println!("{:<20} {}", case.name, format!("failed: {e}"));
                continue;
            }
        };

        let label = if case.name == "gemm_f32" {
            format!("{} R={}", case.name, case.scalar("r") as usize)
        } else {
            case.name.to_string()
        };
        let launch_s = (tn - t1) / (REPS - 1) as f64;
        let bytes = bytes_moved(&case, &inputs);
        let gbs = bytes as f64 / launch_s / 1e9;
        println!(
            "{:<20} {:>10.2} {:>10.1} {:>12}",
            label,
            launch_s * 1e6,
            gbs,
            bytes
        );
    }
    0
}

fn timed(f: impl FnOnce() -> Result<Vec<f32>, String>) -> Result<f64, String> {
    let start = std::time::Instant::now();
    f()?;
    Ok(start.elapsed().as_secs_f64())
}

/// Bytes a single launch has to move: every input read once, output written once.
fn bytes_moved(case: &OpCase, inputs: &[Tensor]) -> usize {
    let read: usize = inputs.iter().map(|t| t.numel() * 4).sum();
    let written = match case.name {
        "gemv_f32" => case.scalar("m") as usize * 4,
        "gemm_f32" => case.scalar("r") as usize * case.scalar("m") as usize * 4,
        "attn_decode_f32" => case.scalar("n_heads") as usize * case.scalar("head_dim") as usize * 4,
        "rmsnorm_f32" | "softmax_f32" => case.scalar("n") as usize * 4,
        _ => inputs[0].numel() * 4,
    };
    read + written
}
