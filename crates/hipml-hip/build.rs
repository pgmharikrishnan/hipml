//! Point the linker at the HIP runtime when the `hip` feature is on.
//!
//! `amdhip64.lib` lives in `$HIP_PATH/lib`; the matching DLL lives in
//! `$HIP_PATH/bin`, so that directory has to be on `PATH` at run time. Without
//! `HIP_PATH` the build still succeeds for the non-HIP half of the workspace,
//! which is the whole point of the feature gate.

fn main() {
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    if std::env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }

    let hip_path = match std::env::var("HIP_PATH") {
        Ok(p) => p,
        Err(_) => {
            println!(
                "cargo:warning=HIP_PATH is not set: linking amdhip64 will fail. \
                 Set HIP_PATH to the HIP SDK root (the directory containing bin/ and lib/)."
            );
            return;
        }
    };

    println!("cargo:rustc-link-search=native={hip_path}/lib");
    // Some layouts keep the import library beside the DLL.
    println!("cargo:rustc-link-search=native={hip_path}/bin");
}
