// SPDX-License-Identifier: Apache-2.0
//! `hipml decode-ablate`: the decode stage budget by removal, not by tracing.
//!
//! A stage span prices a HIP event, not a kernel
//! ([`../../docs/notes/decode-residual-gemv.md`](../../docs/notes/decode-residual-gemv.md)
//! §4: deleting two stage rows moved the traced token 0.94 ms and the untraced
//! step in the same process 0.05 ms). This command does the opposite: it
//! **removes** a stage's launches from the decode step and times the rest with
//! no trace, no per-kernel event pair and no span anywhere in the process. A
//! dropped stage costs nothing on the host, so it cannot be an instrument's
//! contribution.
//!
//! Every cell runs `--tokens` tokens with one synchronize each, `--reps` times,
//! and the cells are interleaved across reps so a clock ramp cannot favour one.
//! One model load, one lock hold, one process.
//!
//! ```text
//! hipml decode-ablate --gguf <dir>/Qwen3.8-27B-UD-Q4_K_M.gguf \
//!     --depth 0 --tokens 200 --reps 3 --kv q8
//! ```
//!
//! The cells and the arithmetic they support are documented in
//! `docs/notes/decode-ablation.md`.

use std::time::Instant;

use hipml_model::gpu::{Ablate, GpuEngine, KvMode, Stage};
use hipml_model::Model;

struct Args {
    gguf: String,
    depth: usize,
    /// Context the caches are sized for; defaults to `depth + tokens + 8`, and
    /// can be set independently to test the cache footprint at a shallow depth.
    ctx: Option<usize>,
    tokens: usize,
    reps: usize,
    kv: KvMode,
    graph: bool,
    ids_out: Option<String>,
    /// Run only these cells (default: the whole table).
    cells: Option<Vec<String>>,
}

impl Args {
    fn parse(args: &[String]) -> Result<Args, String> {
        let mut a = Args {
            gguf: String::new(),
            depth: 0,
            ctx: None,
            tokens: 200,
            reps: 2,
            kv: KvMode::Q8,
            graph: false,
            ids_out: None,
            cells: None,
        };
        let mut i = 0;
        while i < args.len() {
            let need = |i: usize| -> Result<&str, String> {
                args.get(i + 1)
                    .map(String::as_str)
                    .ok_or_else(|| format!("{} needs a value", args[i]))
            };
            match args[i].as_str() {
                "--gguf" | "--model" => {
                    a.gguf = need(i)?.to_string();
                    i += 2;
                }
                "--depth" => {
                    a.depth = need(i)?.parse().map_err(|e| format!("bad --depth: {e}"))?;
                    i += 2;
                }
                "--ctx" => {
                    a.ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                    i += 2;
                }
                "--tokens" => {
                    a.tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                    i += 2;
                }
                "--reps" => {
                    a.reps = need(i)?.parse().map_err(|e| format!("bad --reps: {e}"))?;
                    i += 2;
                }
                "--kv" => {
                    a.kv = KvMode::parse(need(i)?)?;
                    i += 2;
                }
                "--graph" => {
                    a.graph = true;
                    i += 1;
                }
                "--ids-out" => {
                    a.ids_out = Some(need(i)?.to_string());
                    i += 2;
                }
                "--cells" => {
                    a.cells = Some(
                        need(i)?
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                    i += 2;
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if a.gguf.is_empty() {
            return Err("--gguf PATH is required".to_string());
        }
        if a.kv == KvMode::F32 {
            return Err("--depth needs a quantized cache (--kv q8 or q4)".to_string());
        }
        Ok(a)
    }
}

/// One ablation cell: a label and the stages it drops.
struct Cell {
    label: &'static str,
    drop: Ablate,
}

fn cells() -> Vec<Cell> {
    let base = Ablate::gemv_only;
    // "gemv plus exactly one stage": the GEMV bytes are identical in every one
    // of these, so the delta over `gemv_only` is that stage's cost while the
    // weight stream is running around it.
    let plus = |s: Stage| -> Ablate {
        let mut a = base();
        a.keep(s);
        a
    };
    let mut plus_many = |stages: &[Stage]| -> Ablate {
        let mut a = base();
        for s in stages {
            a.keep(*s);
        }
        a
    };
    let only = |spec: &str| -> Ablate { Ablate::parse(spec).expect("literal") };

    vec![
        Cell {
            label: "full",
            drop: Ablate::none(),
        },
        // The isolated weight stream: every projection, nothing else.
        Cell {
            label: "gemv_only",
            drop: base(),
        },
        // Add one stage back at a time, in the order the block runs them.
        Cell {
            label: "gemv+attn_front",
            drop: plus(Stage::AttnFront),
        },
        Cell {
            label: "gemv+kv_append",
            drop: plus(Stage::KvAppend),
        },
        Cell {
            label: "gemv+attn_reduce",
            drop: plus(Stage::AttnReduce),
        },
        Cell {
            label: "gemv+q_gate",
            drop: plus(Stage::QGate),
        },
        Cell {
            label: "gemv+ssm_gates",
            drop: plus(Stage::SsmGates),
        },
        Cell {
            label: "gemv+ssm_conv",
            drop: plus(Stage::SsmConv),
        },
        Cell {
            label: "gemv+ssm_l2",
            drop: plus(Stage::SsmL2),
        },
        Cell {
            label: "gemv+ssm_rec",
            drop: plus(Stage::SsmRec),
        },
        Cell {
            label: "gemv+ssm_gn",
            drop: plus(Stage::SsmGateNorm),
        },
        Cell {
            label: "gemv+norm",
            drop: plus(Stage::Norm),
        },
        // Whole blocks, weight stream included.
        Cell {
            label: "gemv+attn",
            drop: plus_many(&[
                Stage::AttnFront,
                Stage::KvAppend,
                Stage::AttnReduce,
                Stage::QGate,
            ]),
        },
        Cell {
            label: "gemv+ssm",
            drop: plus_many(&[
                Stage::SsmGates,
                Stage::SsmConv,
                Stage::SsmL2,
                Stage::SsmRec,
                Stage::SsmGateNorm,
            ]),
        },
        Cell {
            label: "gemv+attn+ssm",
            drop: plus_many(&[
                Stage::AttnFront,
                Stage::KvAppend,
                Stage::AttnReduce,
                Stage::QGate,
                Stage::SsmGates,
                Stage::SsmConv,
                Stage::SsmL2,
                Stage::SsmRec,
                Stage::SsmGateNorm,
            ]),
        },
        // One stage removed from the full token: its marginal cost with every
        // other stage present.
        Cell {
            label: "-attn_front",
            drop: only("attn_front"),
        },
        Cell {
            label: "-kv_append",
            drop: only("kv_append"),
        },
        // The same launch a second time at the same site. The delta against
        // the cell above it is what one more launch costs where it runs; a
        // second copy that is cheap means the site is expensive, not the
        // kernel.
        Cell {
            label: "gemv+attn_reduce*2",
            drop: only("gemv_only,attn_reduce*2"),
        },
        Cell {
            label: "gemv+ssm_rec*2",
            drop: only("gemv_only,ssm_rec*2"),
        },
        Cell {
            label: "attn_reduce2_only",
            drop: only("all,attn_reduce*2"),
        },
        Cell {
            label: "ssm_rec2_only",
            drop: only("all,ssm_rec*2"),
        },
        Cell {
            label: "-attn_reduce",
            drop: only("attn_reduce"),
        },
        Cell {
            label: "-q_gate",
            drop: only("q_gate"),
        },
        Cell {
            label: "-ssm_gates",
            drop: only("ssm_gates"),
        },
        Cell {
            label: "-ssm_conv",
            drop: only("ssm_conv"),
        },
        Cell {
            label: "-ssm_l2",
            drop: only("ssm_l2"),
        },
        Cell {
            label: "-ssm_rec",
            drop: only("ssm_rec"),
        },
        Cell {
            label: "-ssm_gn",
            drop: only("ssm_gn"),
        },
        Cell {
            label: "-norm",
            drop: only("norm"),
        },
        Cell {
            label: "-ew",
            drop: only("ew"),
        },
        Cell {
            label: "-embed",
            drop: only("embed"),
        },
        Cell {
            label: "-lmhead",
            drop: only("lmhead"),
        },
        Cell {
            label: "-argmax",
            drop: only("argmax"),
        },
        Cell {
            label: "-attn",
            drop: only("attn"),
        },
        Cell {
            label: "-ssm",
            drop: only("ssm"),
        },
    ]
}

struct Sample {
    ms: f64,
}

pub fn run(args: &[String]) -> i32 {
    let a = match Args::parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("decode-ablate: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let (model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("decode-ablate: load: {e}");
            return 1;
        }
    };
    let n_ctx = a.ctx.unwrap_or(a.depth + a.tokens + 8);
    let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("decode-ablate: engine: {e}");
            return 1;
        }
    };
    let bytes = engine.weight_bytes();
    println!(
        "resident: {:.2} GiB in {} tensors on {} (upload {:.1}s)",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        report.upload_secs
    );
    println!(
        "graph: {} layers, {:.3} GiB streamed/token, {} KV {:.3} GiB at n_ctx {}, depth {}",
        engine.cfg.n_layer,
        bytes as f64 / (1u64 << 30) as f64,
        engine.kv_mode.name(),
        engine.kv_bytes(n_ctx) as f64 / (1u64 << 30) as f64,
        n_ctx,
        a.depth,
    );
    println!("build: {:.1}s", t0.elapsed().as_secs_f64());

    if let Err(e) = engine.fill_synthetic(a.depth) {
        eprintln!("decode-ablate: synthetic fill: {e}");
        return 1;
    }

    let all = cells();
    let picked: Vec<&Cell> = match &a.cells {
        Some(names) => {
            let mut v = Vec::new();
            for n in names {
                match all.iter().find(|c| c.label == n) {
                    Some(c) => v.push(c),
                    None => {
                        eprintln!(
                            "decode-ablate: unknown cell {n}; known: {}",
                            all.iter().map(|c| c.label).collect::<Vec<_>>().join(",")
                        );
                        return 2;
                    }
                }
            }
            v
        }
        None => all.iter().collect(),
    };

    println!(
        "\n{} tokens/cell, {} reps, cells interleaved across reps, untraced\n",
        a.tokens, a.reps
    );
    println!(
        "{:<4} {:<16} {:>9} {:>8}  {}",
        "rep", "cell", "ms/tok", "tok/s", "drops"
    );

    let mut results: Vec<(String, Vec<Sample>)> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();

    for rep in 0..a.reps {
        for cell in &picked {
            engine.set_ablate(cell.drop);
            // Every cell starts from the same zeros: a dropped producer leaves
            // its consumer reading the same buffer, and uninitialized memory can
            // hold subnormals this card executes an order of magnitude slower.
            if let Err(e) = engine.reset_scratch() {
                eprintln!("decode-ablate: reset: {e}");
                return 1;
            }
            if a.graph {
                if let Err(e) = engine.build_token_graph(a.depth) {
                    eprintln!("decode-ablate: capture: {e}");
                    return 1;
                }
            }
            let mut st = engine.state_at(a.depth);
            // Two warm tokens: clocks and the code objects settle before the
            // measured loop. Their ids are discarded.
            for _ in 0..2 {
                let r = if a.graph {
                    engine.graph_token(1, &mut st)
                } else {
                    engine.forward_token_id(1, &mut st)
                };
                if let Err(e) = r {
                    eprintln!("decode-ablate: warm: {e}");
                    return 1;
                }
            }
            // Each cell starts at the same depth, so every cell reads a cache
            // of the same shape.
            let mut st = engine.state_at(a.depth);
            let mut next = 1u32;
            let t = Instant::now();
            for _ in 0..a.tokens {
                let r = if a.graph {
                    engine.graph_token(next, &mut st)
                } else {
                    engine.forward_token_id(next, &mut st)
                };
                match r {
                    Ok(id) => {
                        next = id;
                        if rep == 0 && a.ids_out.is_some() && cell.label == "full" {
                            ids.push(id);
                        }
                    }
                    Err(e) => {
                        eprintln!("decode-ablate: {}: {e}", cell.label);
                        return 1;
                    }
                }
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / a.tokens as f64;
            println!(
                "{:<4} {:<16} {:>9.3} {:>8.2}  {}",
                rep,
                cell.label,
                ms,
                1e3 / ms,
                if cell.drop.is_none() {
                    "—".to_string()
                } else {
                    cell.drop.names().join(",")
                }
            );
            match results.iter_mut().find(|(l, _)| l == cell.label) {
                Some((_, v)) => v.push(Sample { ms }),
                None => results.push((cell.label.to_string(), vec![Sample { ms }])),
            }
        }
    }

    println!(
        "\n{:<16} {:>9} {:>9} {:>9} {:>8} {:>9}  {}",
        "cell", "mean", "min", "max", "GB/s", "tok/s", "drops"
    );
    for (label, samples) in &results {
        let cell = picked
            .iter()
            .find(|c| c.label == label)
            .expect("cell measured");
        let mut ms: Vec<f64> = samples.iter().map(|s| s.ms).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let (min, max) = (ms[0], ms[ms.len() - 1]);
        println!(
            "{:<16} {:>9.3} {:>9.3} {:>9.3} {:>8.1} {:>9.2}  {}",
            label,
            mean,
            min,
            max,
            bytes as f64 / (mean * 1e-3) / 1e9,
            1e3 / mean,
            if cell.drop.is_none() {
                "—".to_string()
            } else {
                cell.drop.names().join(",")
            }
        );
    }

    // The arithmetic the command exists for.
    let mean_of = |label: &str| -> Option<f64> {
        results
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, v)| v.iter().map(|s| s.ms).sum::<f64>() / v.len() as f64)
    };
    if let (Some(full), Some(iso)) = (mean_of("full"), mean_of("gemv_only")) {
        println!(
            "\nfull {full:.3} ms = isolated GEMV {iso:.3} ms + {:.3} ms of everything else",
            full - iso
        );
        println!(
            "isolated GEMV {:.1} GB/s over {:.3} GiB/token",
            bytes as f64 / (iso * 1e-3) / 1e9,
            bytes as f64 / (1u64 << 30) as f64
        );
        // Marginal costs, summed. If the stages were independent the sum would
        // equal `full - gemv_only`; the excess is the weight stream being slowed
        // by the stages that remain.
        // `lmhead` is a GEMV, so it is part of `gemv_only` and is not in this
        // list; its marginal cost is the weight stream, not a stage. The same
        // goes for `attn_gemv`, `ssm_gemv` and `ffn_gemv`.
        let stages = [
            "embed",
            "attn_front",
            "kv_append",
            "attn_reduce",
            "q_gate",
            "ssm_gates",
            "ssm_conv",
            "ssm_l2",
            "ssm_rec",
            "ssm_gn",
            "norm",
            "ew",
            "argmax",
        ];
        let mut sum = 0.0;
        let mut n = 0;
        for s in stages {
            if let Some(v) = mean_of(&format!("-{s}")) {
                let d = full - v;
                println!("  -{s:<10} {:>7.3} ms", d);
                sum += d;
                n += 1;
            }
        }
        if n == stages.len() {
            println!("  sum of marginal costs {sum:.3} ms");
            let implied = full - sum;
            println!(
                "  implied in-situ GEMV {implied:.3} ms ({:.1} GB/s), isolated {iso:.3} ms ({:.1} GB/s), \
                 ratio {:.1}%",
                bytes as f64 / (implied * 1e-3) / 1e9,
                bytes as f64 / (iso * 1e-3) / 1e9,
                implied / iso * 100.0
            );
            println!(
                "  full - gemv_only {:.3} ms  ->  interaction {:+.3} ms{}",
                full - iso,
                sum - (full - iso),
                if sum - (full - iso) > 0.05 {
                    " (stages cost more apart than together: the weight stream is being slowed)"
                } else {
                    ""
                }
            );
        }
    }

    if let Some(dest) = &a.ids_out {
        let text: Vec<String> = ids.iter().map(|v| v.to_string()).collect();
        if let Err(e) = std::fs::write(dest, text.join("\n") + "\n") {
            eprintln!("decode-ablate: ids out: {e}");
            return 1;
        }
        println!(
            "greedy ids: {} tokens from {} written to {dest} [last {}]",
            ids.len(),
            ids.first().copied().unwrap_or(0),
            ids.last().copied().unwrap_or(0)
        );
    }
    0
}
