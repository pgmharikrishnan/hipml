//! `hipml skinny-bench`: the R x K GEMM cost curve on the real shapes.
//!
//! The question this answers is not "how fast is one launch" but "what does the
//! second column cost". For each shape and each resident block format it times
//! `gemm_skinny_*` at R = 1, 2, 4, 8, 16 over one resident weight, and prints
//! the ratio against R = 1 next to the single-column GEMV that decode uses
//! today. Acceptance: R = 4 <= 1.1x, R = 8 <= 1.25x.
//!
//! `--geometry` sweeps the compile-time tile shapes (`gemm_skinny_q4_kg*`) at
//! R = 1 and R = 8 on the Q4_K shapes; `--gguf` measures the model's own
//! tensors instead of the built-in shape table.

use std::time::Instant;

use hipml_core::quant::{random_q4_k_bytes, random_q6_k_bytes, random_q8_0_bytes, TensorType};
use hipml_core::rng::Rng;
use hipml_hip::{Buffer, ByteBuffer};
use hipml_kernels::Launcher;

/// One measured shape: `[n, k]` weights of `ty`.
struct Shape {
    label: &'static str,
    ty: TensorType,
    n: usize,
    k: usize,
}

/// The ffn gate/up/down, o-proj and ssm projection shapes on Qwen3.8-27B. The
/// three named in the work package are exact; `q4_k 5120x5120` is the o-proj /
/// attention-output shape and `q6_k 17408x5120` the ffn_up sibling of the gate.
const SHAPES: &[Shape] = &[
    Shape {
        label: "q4_k 17408x5120 ffn_gate",
        ty: TensorType::Q4K,
        n: 17408,
        k: 5120,
    },
    Shape {
        label: "q6_k 17408x5120 ffn_up",
        ty: TensorType::Q6K,
        n: 17408,
        k: 5120,
    },
    Shape {
        label: "q6_k 5120x17408 ffn_down",
        ty: TensorType::Q6K,
        n: 5120,
        k: 17408,
    },
    Shape {
        label: "q4_k 5120x5120 o_proj",
        ty: TensorType::Q4K,
        n: 5120,
        k: 5120,
    },
    Shape {
        label: "q8_0 8192x5120 ssm",
        ty: TensorType::Q8_0,
        n: 8192,
        k: 5120,
    },
];

const ROWS: &[usize] = &[1, 2, 4, 8, 16];

fn weight_bytes(ty: TensorType, n: usize, k: usize) -> u64 {
    match ty {
        TensorType::Q4K => (n * k / 256 * 144) as u64,
        TensorType::Q6K => (n * k / 256 * 210) as u64,
        TensorType::Q8_0 => (n * k / 32 * 34) as u64,
        _ => 0,
    }
}

fn build_weights(ty: TensorType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    match ty {
        TensorType::Q4K => random_q4_k_bytes(&mut rng, n * k / 256),
        TensorType::Q6K => random_q6_k_bytes(&mut rng, n * k / 256),
        TensorType::Q8_0 => random_q8_0_bytes(&mut rng, n * k / 32),
        _ => Vec::new(),
    }
}

fn entry_for(ty: TensorType) -> &'static str {
    match ty {
        TensorType::Q4K => "gemm_skinny_q4_k",
        TensorType::Q6K => "gemm_skinny_q6_k",
        TensorType::Q8_0 => "gemm_skinny_q8_0",
        _ => unreachable!(),
    }
}

fn gemv_for(ty: TensorType) -> &'static str {
    match ty {
        TensorType::Q4K => "gemv_q4_k_r1",
        TensorType::Q6K => "gemv_q6_k_r1",
        TensorType::Q8_0 => "gemv_q8_0",
        _ => unreachable!(),
    }
}

/// Best of `tries` independent timed runs, each of `reps` launches.
fn best_ms(mut f: impl FnMut() -> Result<f64, String>, tries: usize) -> Result<f64, String> {
    let mut best = f64::MAX;
    for _ in 0..tries {
        let s = f()?;
        if s > 0.0 && s < best {
            best = s;
        }
    }
    Ok(best * 1e3)
}

pub fn run(args: &[String]) -> i32 {
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10)
        .max(1);
    let geometry = args.iter().any(|a| a == "--geometry");

    let launcher = match Launcher::new() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("device: none ({e})");
            return 1;
        }
    };
    println!(
        "device: {}  reps {}  weight bytes from the block format\n",
        launcher.device_name, reps
    );

    let mut worst4 = 0.0f64;
    let mut worst8 = 0.0f64;
    for (si, shape) in SHAPES.iter().enumerate() {
        let bytes = weight_bytes(shape.ty, shape.n, shape.k);
        let w = build_weights(shape.ty, shape.n, shape.k, 0x5C33_0001 + si as u64);
        let wb = match ByteBuffer::from_host(&w) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}: upload: {e}", shape.label);
                continue;
            }
        };
        // One 16-row activation tile; the kernel zero-pads R < 16.
        let x: Vec<f32> = (0..16 * shape.k)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let xb = match Buffer::from_host(&x) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}: x upload: {e}", shape.label);
                continue;
            }
        };
        let y = match Buffer::alloc(16 * shape.n) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{}: y alloc: {e}", shape.label);
                continue;
            }
        };

        // One-column GEMV: today's decode kernel on the same bytes.
        let x1 = Buffer::alloc(shape.k).ok();
        let y1 = Buffer::alloc(shape.n).ok();
        let gemv = match (x1.as_ref(), y1.as_ref()) {
            (Some(x1), Some(y1)) => best_ms(
                || {
                    let t = Instant::now();
                    for _ in 0..reps {
                        launcher
                            .run_quant_resident(gemv_for(shape.ty), &wb, x1, y1, shape.n, shape.k)?;
                    }
                    launcher.synchronize()?;
                    Ok(t.elapsed().as_secs_f64() / reps as f64)
                },
                3,
            )
            .ok(),
            _ => None,
        };

        let entry = entry_for(shape.ty);
        let mut base = 0.0f64;
        println!(
            "{}  ({:.1} MB)",
            shape.label,
            bytes as f64 / 1e6
        );
        if let Some(g) = gemv {
            println!(
                "  gemv r=1        {:8.3} ms  {:6.0} GB/s",
                g,
                bytes as f64 / g * 1e3 / 1e9
            );
        }
        for &r in ROWS {
            let ms = match best_ms(
                || {
                    launcher
                        .run_skinny_resident_reps(entry, &wb, &xb, &y, shape.n, shape.k, r, reps)
                },
                3,
            ) {
                Ok(ms) => ms,
                Err(e) => {
                    println!("  r={r:<2}  {e}");
                    continue;
                }
            };
            if r == 1 {
                base = ms;
            }
            let ratio = if base > 0.0 { ms / base } else { 0.0 };
            if r == 4 {
                worst4 = worst4.max(ratio);
            }
            if r == 8 {
                worst8 = worst8.max(ratio);
            }
            println!(
                "  skinny r={:<2}    {:8.3} ms  {:6.0} GB/s  {:.2}x",
                r,
                ms,
                bytes as f64 / ms * 1e3 / 1e9,
                ratio
            );
        }
        println!();
    }

    println!("worst case over the five shapes: r=4 {worst4:.2}x, r=8 {worst8:.2}x");

    if geometry {
        sweep_geometry(&launcher, reps);
    }
    0
}

/// Tile-shape sweep on the Q4_K entries at R = 1 and R = 8.
fn sweep_geometry(launcher: &Launcher, reps: usize) {
    const VARIANTS: &[(&str, &str)] = &[
        ("gemm_skinny_q4_kg1", "RT16 KW64 T128"),
        ("gemm_skinny_q4_kg2", "RT32 KW64 T128"),
        ("gemm_skinny_q4_kg3", "RT32 KW128 T128"),
        ("gemm_skinny_q4_kg4", "RT32 KW256 T128"),
        ("gemm_skinny_q4_kg5", "RT64 KW128 T256"),
        ("gemm_skinny_q4_kg6", "RT128 KW64 T256"),
    ];
    println!("\ngeometry sweep (Q4_K), ms at r=1 / r=8:");
    for shape in SHAPES.iter().filter(|s| s.ty == TensorType::Q4K) {
        let bytes = weight_bytes(shape.ty, shape.n, shape.k);
        let w = build_weights(shape.ty, shape.n, shape.k, 0x5C44_0001);
        let Ok(wb) = ByteBuffer::from_host(&w) else { continue };
        let x: Vec<f32> = (0..16 * shape.k).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let Ok(xb) = Buffer::from_host(&x) else { continue };
        let Ok(y) = Buffer::alloc(16 * shape.n) else { continue };
        println!("  {}", shape.label);
        for (entry, geom) in VARIANTS {
            let ms = |r: usize| {
                best_ms(
                    || {
                        launcher.run_skinny_resident_reps(
                            entry, &wb, &xb, &y, shape.n, shape.k, r, reps,
                        )
                    },
                    2,
                )
                .unwrap_or(f64::NAN)
            };
            let (a, b) = (ms(1), ms(8));
            println!(
                "    {geom:<16} {a:8.3} / {b:8.3}   ({:.0} GB/s at r=8)",
                bytes as f64 / b * 1e3 / 1e9
            );
        }
    }
}
