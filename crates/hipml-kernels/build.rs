//! Compile the HIP kernels when a HIP toolchain is present.
//!
//! `hipcc --genco` produces a code object per source file, which the host loads
//! with `hipModuleLoad`. Nothing is compiled without `HIP_PATH`, so the workspace
//! builds on a machine with no ROCm; the parity harness then reports every case
//! as skipped instead of pretending it passed.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=HIPML_ARCH");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let sources = hip_sources();

    let hip_path = match std::env::var("HIP_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            println!(
                "cargo:warning=HIP_PATH not set: {} HIP kernels were not compiled; \
                 parity cases will report as skipped",
                sources.len()
            );
            return;
        }
    };

    let hipcc = hip_path
        .join("bin")
        .join(if cfg!(windows) { "hipcc.bat" } else { "hipcc" });
    let hipcc = if hipcc.exists() {
        hipcc
    } else {
        println!(
            "cargo:warning=hipcc not found under {}; kernels not compiled",
            hip_path.display()
        );
        return;
    };

    // The R9700 is gfx1201. Override for other cards.
    let arch = std::env::var("HIPML_ARCH").unwrap_or_else(|_| "gfx1201".to_string());

    // One code object for the whole kernel set: one module to load, one manifest
    // to keep in step, and no per-file bookkeeping on the host.
    let obj = out_dir.join("hipml.co");
    let mut cmd = Command::new(&hipcc);
    cmd.arg("--genco")
        .arg("-O3")
        .arg(format!("--offload-arch={arch}"))
        .arg(format!("-I{}", hip_path.join("include").display()));
    for src in &sources {
        cmd.arg(src);
    }
    let status = cmd.arg("-o").arg(&obj).status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("hipcc failed with {s}"),
        Err(e) => panic!("could not run {}: {e}", hipcc.display()),
    }

    println!("cargo:rustc-env=HIPML_KERNEL_DIR={}", out_dir.display());
    println!("cargo:rustc-env=HIPML_KERNEL_ARCH={arch}");
}

/// Every `.hip` file under `kernels/`.
fn hip_sources() -> Vec<PathBuf> {
    let dir = PathBuf::from("kernels");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "hip").unwrap_or(false))
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no .hip sources under {}", dir.display());
    out
}
