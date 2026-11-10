// SPDX-License-Identifier: Apache-2.0
//! `hipml oracle-gate`: the CPU f32 oracle and the production GPU path, step
//! for step, on one prompt.
//!
//! Every earlier gate compared a kernel against a kernel, or the GPU against
//! itself (MTP exactness, batch `--check`). The engine-level gates ran at depth
//! 0 or on a synthetic prompt, and `--compare-cpu` refused `--chunk`, so the
//! chunked prefill — the path every benchmark number uses — was never compared
//! with the oracle at depth. This command does exactly that:
//!
//! 1. the CPU oracle pre-fills the prompt and greedily decodes `steps` ids;
//! 2. the GPU engine chunk-prefills the same ids and decodes the same count;
//! 3. the two id streams are compared, and at the first disagreement the gate
//!    prints both distributions and the margin that separated the two ids.
//!
//! It exits non-zero when the id streams disagree and the oracle's own margin
//! at that step is at least `--tol-nats` (default 0.05 nats). A smaller margin
//! is a near-tie: fp16 prefill operands and q8 KV can legitimately reorder two
//! ids that the oracle barely separates, and failing on that would make the
//! gate flaky. Near-ties are still printed as warnings.

use std::time::Instant;

use hipml_core::gen::top_logprobs;
use hipml_core::model::Model as CpuModel;
use hipml_model::gpu::{GpuEngine, KvMode};
use hipml_model::Model;

struct Args {
    gguf: String,
    ids_file: Option<String>,
    ids: Option<Vec<u32>>,
    steps: usize,
    chunk: usize,
    ctx: Option<usize>,
    kv: KvMode,
    tol_nats: f32,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        ids_file: None,
        ids: None,
        steps: 20,
        chunk: 2048,
        ctx: None,
        kv: KvMode::Q8,
        tol_nats: 0.05,
    };
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--gguf" => {
                a.gguf = need(i)?.to_string();
                i += 2;
            }
            "--ids-file" => {
                a.ids_file = Some(need(i)?.to_string());
                i += 2;
            }
            "--ids" => {
                let parsed: Result<Vec<u32>, _> =
                    need(i)?.split(',').map(|s| s.trim().parse::<u32>()).collect();
                a.ids = Some(parsed.map_err(|e| format!("bad --ids: {e}"))?);
                i += 2;
            }
            "--steps" => {
                a.steps = need(i)?.parse().map_err(|e| format!("bad --steps: {e}"))?;
                i += 2;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--ctx" => {
                a.ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                i += 2;
            }
            "--kv" => {
                a.kv = KvMode::parse(need(i)?)?;
                i += 2;
            }
            "--tol-nats" => {
                a.tol_nats = need(i)?.parse().map_err(|e| format!("bad --tol-nats: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.steps == 0 {
        return Err("--steps must be at least 1".to_string());
    }
    Ok(a)
}

/// One side's view of a sampled step: the id, the row it came from, and the
/// top few entries. The margin is the log-probability gap the winner keeps.
struct Step {
    id: u32,
    row: Vec<f32>,
    top: Vec<(usize, f32)>,
}

impl Step {
    fn new(logits: Vec<f32>, k: usize) -> Step {
        let top = top_logprobs(&logits, k);
        Step {
            id: top[0].0 as u32,
            row: logits,
            top,
        }
    }

    fn lp(&self, id: u32) -> f32 {
        hipml_core::gen::logprob_of(&self.row, id as usize)
    }

    fn margin(&self) -> f32 {
        match (self.top.first(), self.top.get(1)) {
            (Some((_, a)), Some((_, b))) => a - b,
            _ => f32::INFINITY,
        }
    }
}

fn fmt_top(top: &[(usize, f32)]) -> String {
    top.iter()
        .map(|(id, lp)| format!("{id} {lp:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("oracle-gate: {e}");
            return 2;
        }
    };
    let prompt = match (&a.ids_file, &a.ids) {
        (Some(p), _) => match crate::device_cmd::read_ids_file(p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("oracle-gate: {e}");
                return 2;
            }
        },
        (None, Some(v)) => v.clone(),
        _ => {
            eprintln!("oracle-gate: need --ids-file or --ids");
            return 2;
        }
    };

    // CPU oracle first: it is the arbiter and the slow half (about 2.3 s per
    // token on the reference machine).
    let t = Instant::now();
    let cpu = match CpuModel::load_file(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("oracle-gate: cpu load: {e}");
            return 1;
        }
    };
    let mut st = cpu.new_state(prompt.len() + a.steps + 1);
    let mut logits = Vec::new();
    for &id in &prompt {
        logits = cpu.forward_token(id, &mut st);
    }
    let mut cpu_steps = Vec::with_capacity(a.steps);
    for i in 0..a.steps {
        let s = Step::new(std::mem::take(&mut logits), 5);
        let next = s.id;
        cpu_steps.push(s);
        if i + 1 < a.steps {
            logits = cpu.forward_token(next, &mut st);
        }
    }
    println!(
        "cpu oracle: {} prompt tokens + {} steps in {:.1}s",
        prompt.len(),
        a.steps,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let (model, _report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("oracle-gate: resident load: {e}");
            return 1;
        }
    };
    let n_ctx = a.ctx.unwrap_or(prompt.len() + a.steps + 8);
    let mut engine = match GpuEngine::new(&model, n_ctx, a.kv) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("oracle-gate: graph: {e}");
            return 1;
        }
    };
    let mut gst = engine.new_state();
    let first = match engine.prefill_chunked(&prompt, &mut gst, a.chunk) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("oracle-gate: chunked prefill: {e}");
            return 1;
        }
    };
    let gpu_logits0 = match engine.logits().to_host() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("oracle-gate: logits: {e}");
            return 1;
        }
    };
    println!(
        "gpu engine: chunk {}-token tiles, {} KV, {} prompt tokens in {:.1}s",
        a.chunk.min(prompt.len()).max(1),
        a.kv.name(),
        prompt.len(),
        t.elapsed().as_secs_f64()
    );

    // Compare the prefill row itself before any decode: a wrong prefill state
    // shows up here even when the argmax happens to agree.
    let (mut worst, mut at) = (0.0f32, 0usize);
    for i in 0..gpu_logits0.len().min(cpu_steps[0].row.len()) {
        let d = (gpu_logits0[i] - cpu_steps[0].row[i]).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    println!(
        "prefill logits: max |gpu - cpu| = {worst:.4} at {at} \
         (gpu argmax {} / cpu argmax {})",
        first,
        cpu_steps[0].id
    );

    let mut gpu_steps = Vec::with_capacity(a.steps);
    let mut next = first;
    for i in 0..a.steps {
        let row = if i == 0 {
            gpu_logits0.clone()
        } else {
            match engine.logits().to_host() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("oracle-gate: logits: {e}");
                    return 1;
                }
            }
        };
        gpu_steps.push(Step::new(row, 5));
        if i + 1 < a.steps {
            next = match engine.forward_token_id(next, &mut gst) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("oracle-gate: decode: {e}");
                    return 1;
                }
            };
        }
    }

    let mut diverged: Option<(usize, f32)> = None;
    for i in 0..a.steps {
        let c = &cpu_steps[i];
        let g = &gpu_steps[i];
        let same = c.id == g.id;
        if !same && diverged.is_none() {
            diverged = Some((i, c.margin()));
        }
        // Report the gap each side puts between the two disputing ids. A gap
        // that changed sign is what a sampled id change means.
        let note = if same {
            ""
        } else if c.margin() >= a.tol_nats {
            "  <-- DIVERGES"
        } else {
            "  <-- near-tie"
        };
        println!(
            "  [{i:2}] cpu {:6} (m {:+.4})  gpu {:6} (m {:+.4}){note}",
            c.id,
            c.margin(),
            g.id,
            g.margin()
        );
        if !same {
            println!("       cpu top5: {}", fmt_top(&c.top));
            println!("       gpu top5: {}", fmt_top(&g.top));
            println!(
                "       cpu says {:6} {:+.4} vs {:6} {:+.4} (gap {:+.4})",
                g.id,
                c.lp(g.id),
                c.id,
                c.lp(c.id),
                c.lp(c.id) - c.lp(g.id)
            );
            println!(
                "       gpu says {:6} {:+.4} vs {:6} {:+.4} (gap {:+.4})",
                c.id,
                g.lp(c.id),
                g.id,
                g.lp(g.id),
                g.lp(g.id) - g.lp(c.id)
            );
        }
    }

    match diverged {
        None => {
            println!(
                "PASS: {} ids identical (chunk {}, {} KV, {} prompt tokens)",
                a.steps,
                a.chunk,
                a.kv.name(),
                prompt.len()
            );
            0
        }
        Some((step, margin)) if margin < a.tol_nats => {
            println!(
                "PASS (near-tie): first divergence at step {step}, oracle margin {margin:.4} \
                 nats < tol {}",
                a.tol_nats
            );
            0
        }
        Some((step, margin)) => {
            println!(
                "FAIL: first divergence at step {step}, oracle margin {margin:.4} nats \
                 >= tol {}; the GPU reordered two ids the oracle separates",
                a.tol_nats
            );
            1
        }
    }
}
