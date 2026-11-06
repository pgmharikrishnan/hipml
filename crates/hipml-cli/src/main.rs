//! `hipml` — probe the device, list the kernel manifest, run the parity suite.
//!
//! The parity runner is the gate: a kernel that is not compared against the
//! reference here is not a kernel hipml can use.

use hipml_core::parity::{cases, compare, OpCase};
use hipml_core::tensor::Tensor;
use hipml_kernels::{kernel_arch, kernel_dir, KernelKind, KERNELS};

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
