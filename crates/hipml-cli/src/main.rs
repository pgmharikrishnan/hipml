//! `hipml` — probe the device, list the kernel manifest, run the parity suite.
//!
//! The parity runner is the gate: a kernel that is not compared against the
//! reference here is not a kernel hipml can use.

use hipml_core::parity::{bench_cases, cases, compare, prefill_cases, quant_cases, OpCase};
use hipml_core::tensor::Tensor;
use hipml_kernels::{kernel_arch, kernel_dir, KernelKind, KERNELS};

mod gen_cmd;
mod load_cmd;
mod prefill_cmd;
mod quant_cmd;

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
        "bench" => std::process::exit(bench()),
        "gguf" => std::process::exit(quant_cmd::gguf(&args[1..])),
        "load" => std::process::exit(load_cmd::load(&args[1..])),
        "logits" => std::process::exit(load_cmd::logits(&args[1..])),
        "decode-bench" => std::process::exit(quant_cmd::decode_bench(&args[1..])),
        "generate" => std::process::exit(gen_cmd::generate(&args[1..])),
        "tokenize" => std::process::exit(gen_cmd::tokenize(&args[1..])),
        "matbench" => std::process::exit(gen_cmd::matbench(&args[1..])),
        "prefill-bench" => std::process::exit(prefill_cmd::bench(&args[1..])),
        "prefill-model" => std::process::exit(prefill_cmd::model(&args[1..])),
        "parity" => {
            let filter = args.iter().skip(1).find(|a| !a.starts_with('-')).cloned();
            std::process::exit(parity(filter.as_deref()));
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
         \x20 parity    run every parity case (reference vs device); optional name filter\n\
         \x20 bench     time each kernel launch and report effective bandwidth\n\
         \x20 gguf      list the tensors in a GGUF file; `gguf <file> read <tensor>`\n\
         \x20 decode-bench  stream every Q4_K/Q6_K/Q8_0 tensor through GEMV once\n\
         \x20 generate  --gguf PATH --text \"prompt\" --tokens 20   greedy decode\n\
         \x20 load      load every tensor into VRAM and verify residency\n\
         \x20 logits    run lm_head on a 5120-value f32 vector file\n\
         \x20 prefill-bench time the tiled prefill GEMM at Qwen3.8-27B shapes, M=512/2048\n\
         \x20 prefill-model prefill the whole model cold, then warm from a KV prefix\n\
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
            KernelKind::RowTile { rows, block } => {
                format!("row-tile {rows} rows, {block} threads")
            }
            KernelKind::Prefill {
                rows,
                tokens,
                block,
            } => format!("prefill {rows} rows x {tokens} tokens, {block} threads"),
            KernelKind::PerHead { block } => format!("per-head, {block} threads"),
            KernelKind::PerHeadOnDim => "per-head, one thread per dim".to_string(),
            KernelKind::Linear { block } => format!("linear, {block} threads"),
            KernelKind::HostOnly(why) => format!("host-only: {why}"),
        };
        println!("{:<20} {:<18} {}", k.name, k.entry, geometry);
    }
}

/// Run every case. Exit 0 only when nothing failed; skips are reported but do
/// not fail the run, so the suite stays meaningful on a machine without a GPU.
fn parity(filter: Option<&str>) -> i32 {
    let launcher = hipml_kernels::Launcher::new();
    match &launcher {
        Ok(l) => println!("device: {}", l.device_name),
        Err(e) => println!("device: none ({e})"),
    }

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    println!("\n{:<20} {:<7} {}", "case", "result", "detail");

    for case in cases() {
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

    // Quantized GEMV: build the reference only when a device is present, so a
    // GPU-less run does not spend seconds generating a 73 MB tensor just to skip.
    for case in quant_cases() {
        if let Some(f) = filter {
            if !case.name.contains(f) && !case.label.contains(f) {
                continue;
            }
        }
        let outcome = match launcher.as_ref() {
            Err(e) => Outcome::Skip(e.clone()),
            Ok(l) => {
                let (w, x, expected) = case.reference();
                match l.run_quant(case.name, &w, &x, case.m, case.k) {
                    Err(e) => Outcome::Fail(format!("launch failed: {e}")),
                    Ok(actual) => match compare(&expected, &actual, case.tol) {
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
        };
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
        println!("{:<28} {:<7} {}", case.label, verdict, detail);
    }

    // Tiled prefill GEMM: same contract, multi-token output.
    for case in prefill_cases() {
        if let Some(f) = filter {
            if !case.name.contains(f) && !case.label.contains(f) {
                continue;
            }
        }
        let outcome = match launcher.as_ref() {
            Err(e) => Outcome::Skip(e.clone()),
            Ok(l) => {
                let (w, x, expected) = case.reference();
                match l.run_prefill_reps(case.name, &w, &x, case.m, case.k, case.tokens, 1) {
                    Err(e) => Outcome::Fail(format!("launch failed: {e}")),
                    Ok((actual, _)) => match compare(&expected, &actual, case.tol) {
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
        };
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
        println!("{:<28} {:<7} {}", case.label, verdict, detail);
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

    // Quantized production shapes: ffn_up/ffn_down/ffn_gate at Qwen3.8-27B sizes.
    quant_cmd::bench_quant(&launcher, REPS);
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
