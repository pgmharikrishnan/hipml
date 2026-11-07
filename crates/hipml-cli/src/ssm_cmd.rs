//! `hipml ssm`: parity for the gated-delta-net kernels, then a per-layer
//! timing breakdown at the production shape.
//!
//! The timing separates the four projections from the recurrent work
//! (conv1d, L2 norms, delta-net recurrence, gated RMS norm) so the cost the
//! SSM block adds on top of streaming weights is visible.

use hipml_core::parity::{cases, ssm_real_layer_case, REAL_SSM};
use hipml_kernels::ssm::{SsmDims, SsmGpuLayer, SsmLayerRefs};
use hipml_kernels::Launcher;

use crate::{run_case, Outcome};

const SSM_CASES: [&str; 5] = [
    "ssm_conv1d_f32",
    "gdn_ar_f32",
    "ssm_l2norm_f32",
    "ssm_gate_norm_f32",
    "ssm_layer_f32",
];

/// `hipml ssm`
pub fn ssm(_args: &[String]) -> i32 {
    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            println!("device: none ({e})");
            return 1;
        }
    };
    println!("device: {}", launcher.device_name);

    let mut fail = 0usize;
    println!("\n{:<22} {:<7} {}", "case", "result", "detail");
    for case in cases().into_iter().filter(|c| SSM_CASES.contains(&c.name)) {
        let (inputs, expected) = case.reference();
        let outcome = run_case(&case, &inputs, &expected, Ok(&launcher));
        if report(case.name, outcome) {
            fail += 1;
        }
    }

    // The real layer shape: parity first, then reuse the generated weights for
    // the timing so the card sees the production dimensions.
    let real = ssm_real_layer_case();
    println!(
        "\nreal layer: n_embd {} conv {} d_conv {} · {} value heads x {} · {} key heads x {}",
        REAL_SSM.n_embd,
        REAL_SSM.conv_ch,
        REAL_SSM.d_conv,
        REAL_SSM.n_v,
        REAL_SSM.head_v,
        REAL_SSM.n_k,
        REAL_SSM.head_k
    );
    println!("building {} f32 weights on the host...", REAL_SSM.n_embd);
    let (inputs, expected) = real.reference();
    let outcome = run_case(&real, &inputs, &expected, Ok(&launcher));
    if report("ssm_layer_f32 real", outcome) {
        fail += 1;
    }

    if let Err(e) = bench(&launcher, &inputs) {
        println!("bench failed: {e}");
        return 1;
    }

    if fail > 0 {
        1
    } else {
        0
    }
}

/// Print one row. Returns true on failure.
fn report(name: &str, outcome: Outcome) -> bool {
    match outcome {
        Outcome::Pass(d) => {
            println!("{name:<22} {:<7} {d}", "PASS");
            false
        }
        Outcome::Fail(d) => {
            println!("{name:<22} {:<7} {d}", "FAIL");
            true
        }
        Outcome::Skip(d) => {
            println!("{name:<22} {:<7} {d}", "SKIP");
            false
        }
    }
}

fn bench(launcher: &Launcher, inputs: &[hipml_core::tensor::Tensor]) -> Result<(), String> {
    const REPS: usize = 50;

    let dims = SsmDims::from_parity(&REAL_SSM);
    let refs = SsmLayerRefs {
        wqkv: inputs[1].data(),
        wgate: inputs[2].data(),
        conv_w: inputs[3].data(),
        alpha: inputs[4].data(),
        beta: inputs[5].data(),
        dt_bias: inputs[6].data(),
        a: inputs[7].data(),
        norm: inputs[8].data(),
        wout: inputs[9].data(),
    };
    let layer = SsmGpuLayer::new(launcher.module(), launcher.stream(), dims, &refs)?;
    layer.write_x(inputs[0].data())?;
    layer.write_state(inputs[10].data(), inputs[11].data())?;

    // Warm up module/cache state; the timing below is launch + kernel only.
    layer.step()?;
    layer.synchronize()?;

    let gemv = layer.gemv_reps(REPS)?;
    let wout = layer.output_gemv_reps(REPS)?;
    let rec = layer.recurrent_reps(REPS)?;
    let full = layer.step_reps(REPS)?;

    let state_bytes = layer.state_bytes();
    let weight_bytes = layer.weight_bytes();
    println!(
        "\nresident f32 weights (one layer): {:.1} MB",
        weight_bytes as f64 / 1e6
    );
    println!(
        "state per sequence, per layer: {:.2} MB  (48 layers: {:.1} MB)",
        state_bytes as f64 / 1e6,
        48.0 * state_bytes as f64 / 1e6
    );
    println!(
        "\n{:<28} {:>12} {:>16}",
        "stage", "us/layer", "48 layers (ms)"
    );
    for (label, secs) in [
        ("gemv, input projections (4)", gemv),
        ("gemv, output projection", wout),
        ("recurrent (conv/l2/gdn/norm)", rec),
        ("full layer", full),
    ] {
        println!(
            "{:<28} {:>12.1} {:>16.3}",
            label,
            secs * 1e6,
            secs * 48.0 * 1e3
        );
    }
    println!(
        "\nweight-read reference: 17.19 GB at 632 GB/s = 27.19 ms/token\n\
         recurrent add: {:.2} ms/token ({:.1}% of the weight-read time)\n\
         full 48-layer projection: {:.2} ms/token (+{:.2} ms over the weight read)",
        48.0 * rec * 1e3,
        100.0 * (48.0 * rec * 1e3) / 27.19,
        48.0 * full * 1e3,
        48.0 * full * 1e3 - 27.19,
    );
    Ok(())
}
