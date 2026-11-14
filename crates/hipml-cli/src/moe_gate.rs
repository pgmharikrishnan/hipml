// SPDX-License-Identifier: Apache-2.0
//! `hipml moe-gate`: the CPU MoE reference against llama.cpp's greedy ids.
//!
//! The four MoE conventions (softmax domain, top-k and its tie rule, the
//! renormalisation and its clamp, the weight-after-down order, the rank-order
//! fold, the shared-expert scalar) are only worth writing down if they produce
//! the reference engine's tokens. This command is that check: it reads a
//! fixture holding the prompt ids and the ids llama.cpp generated from them,
//! runs the same prompt through the CPU reference, and compares the streams.
//!
//! It deliberately does **not** compare logits. A logit tolerance would hide a
//! routing difference; an id stream does not. The fixture is
//! `bench/fixtures/moe_gate_qwen36.json`, and the command that produced it is
//! recorded inside it.
//!
//! The GPU is not involved: this is the CPU oracle only, so it runs in a build
//! without `--features hip` and needs no device lock.

use std::cmp::Ordering;
use std::time::Instant;

use hipml_core::gen::greedy_with_stop;
use hipml_core::gguf::Gguf;
use hipml_core::json;
use hipml_core::model::Model;
use hipml_core::tokenizer::Tokenizer;

struct Args {
    gguf: String,
    fixture: String,
    steps: Option<usize>,
    /// Run the resident GPU engine over the same prompt and fixture instead of
    /// the CPU reference.
    device: bool,
    /// Chunk size for the GPU prefill. The MoE path ignores it (one decode step
    /// per prompt token) but `generate` takes one, so the flag stays.
    chunk: usize,
    /// Also diff every layer's router decision against the CPU reference at the
    /// last prompt token.
    routes: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        fixture: String::new(),
        steps: None,
        device: false,
        chunk: 2048,
        routes: true,
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
            "--fixture" => {
                a.fixture = need(i)?.to_string();
                i += 2;
            }
            "--steps" => {
                a.steps = Some(need(i)?.parse().map_err(|e| format!("bad --steps: {e}"))?);
                i += 2;
            }
            "--device" => {
                a.device = true;
                i += 1;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--no-routes" => {
                a.routes = false;
                i += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.fixture.is_empty() {
        return Err("--fixture PATH is required".to_string());
    }
    Ok(a)
}

struct Fixture {
    prompt_ids: Vec<u32>,
    reference_ids: Vec<u32>,
    command: Option<String>,
    llama: Option<String>,
}

fn read_fixture(path: &str) -> Result<Fixture, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let v = json::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let ids = |key: &str| -> Result<Vec<u32>, String> {
        let arr = v
            .get(key)
            .and_then(|a| a.as_arr())
            .ok_or_else(|| format!("{path}: no {key} array"))?;
        arr.iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| format!("{path}: {key} holds a non-id"))
            })
            .collect()
    };
    let prompt_ids = ids("prompt_ids")?;
    let reference_ids = ids("reference_ids")?;
    if prompt_ids.is_empty() {
        return Err(format!("{path}: empty prompt_ids"));
    }
    if reference_ids.is_empty() {
        return Err(format!("{path}: empty reference_ids"));
    }
    Ok(Fixture {
        prompt_ids,
        reference_ids,
        command: v
            .get("command")
            .and_then(|s| s.as_str())
            .map(str::to_string),
        llama: v
            .get("llama_cpp")
            .and_then(|s| s.as_str())
            .map(str::to_string),
    })
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("moe-gate: {e}");
            return 2;
        }
    };
    let fx = match read_fixture(&a.fixture) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("moe-gate: {e}");
            return 2;
        }
    };
    if let Some(l) = &fx.llama {
        println!("reference: {l}");
    }
    if let Some(c) = &fx.command {
        println!("command:   {c}");
    }
    let steps = a.steps.unwrap_or(fx.reference_ids.len());
    let n_ctx = fx.prompt_ids.len() + steps + 8;
    println!(
        "prompt:    {} ids, {} reference ids",
        fx.prompt_ids.len(),
        fx.reference_ids.len()
    );

    // The device path must not also load the CPU model: it maps the same 22.85
    // GB file, and on Windows a second mapping while the resident weights are
    // already on the device fails with "insufficient system resources". The
    // routing gate loads its own CPU reference first, then drops it.
    if a.device {
        #[cfg(feature = "hip")]
        return run_device(&a, &fx, steps, n_ctx);
        #[cfg(not(feature = "hip"))]
        {
            eprintln!("moe-gate: --device needs a build with `--features hip`");
            return 2;
        }
    }

    let g = match Gguf::open(&a.gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("moe-gate: cannot open {}: {e}", a.gguf);
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("moe-gate: tokenizer: {e}");
            return 1;
        }
    };
    let t0 = Instant::now();
    let model = match Model::load_file(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("moe-gate: load: {e}");
            return 1;
        }
    };
    let cfg = &model.cfg;
    println!(
        "moe-gate: {} in {:.1}s ({} decoder layers, {} experts, {} used, expert ffn {}, \
         shared expert {}, ffn {})",
        cfg.arch,
        t0.elapsed().as_secs_f64(),
        cfg.n_layer,
        cfg.n_expert,
        cfg.n_expert_used,
        cfg.n_ff_exp,
        if cfg.has_shexp {
            cfg.n_ff_shexp.to_string()
        } else {
            "absent".to_string()
        },
        cfg.n_ff
    );
    if cfg.n_expert == 0 {
        eprintln!(
            "moe-gate: {} declares no experts; this gate is for MoE files",
            a.gguf
        );
        return 1;
    }

    let t1 = Instant::now();
    let mut logits_at: Vec<Vec<f32>> = Vec::with_capacity(steps);
    let out = greedy_with_stop(
        &model,
        &tok,
        &fx.prompt_ids,
        steps,
        n_ctx,
        &[],
        |_, _, logits| logits_at.push(logits.to_vec()),
    );
    let secs = t1.elapsed().as_secs_f64();
    println!(
        "cpu:       {} ids in {secs:.1}s ({:.2} s/token)",
        out.ids.len(),
        secs / steps.max(1) as f64
    );
    println!("ids:       {:?}", out.ids);

    let n = fx.reference_ids.len().min(out.ids.len());
    let first = (0..n).find(|&i| out.ids[i] != fx.reference_ids[i]);
    match first {
        None if out.ids.len() >= fx.reference_ids.len() => {
            println!(
                "PASS: {} / {} ids identical to llama.cpp",
                fx.reference_ids.len(),
                fx.reference_ids.len()
            );
            0
        }
        None => {
            println!(
                "FAIL: only {} of {} ids produced (stopped early)",
                out.ids.len(),
                fx.reference_ids.len()
            );
            1
        }
        Some(i) => {
            println!(
                "FAIL: first divergence at step {i}: llama.cpp {}, reference {}; \
                 llama.cpp tail {:?}, reference tail {:?}",
                fx.reference_ids[i],
                out.ids[i],
                &fx.reference_ids[i..(i + 4).min(fx.reference_ids.len())],
                &out.ids[i..(i + 4).min(out.ids.len())]
            );
            // The margin is the diagnosis: a large one is a convention error, a
            // near-tie is arithmetic precision and the id stream cannot settle
            // it.
            if let Some(logits) = logits_at.get(i) {
                let mut top: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(id, v)| (id, *v)).collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let at = |id: u32| logits.get(id as usize).copied().unwrap_or(f32::NAN);
                println!(
                    "  logits: llama id {} = {:.4}, reference id {} = {:.4}; \
                     top-2 {:?} {:.4} and {:?} {:.4}, margin {:.4} nats",
                    fx.reference_ids[i],
                    at(fx.reference_ids[i]),
                    out.ids[i],
                    at(out.ids[i]),
                    top[0].0,
                    top[0].1,
                    top[1].0,
                    top[1].1,
                    top[0].1 - top[1].1
                );
            }
            1
        }
    }
}

/// The GPU half of the gate: the same prompt and the same id stream, run on the
/// resident engine, plus a layer-by-layer diff of the router's decisions.
///
/// The id stream is the gate; the routing diff is the diagnosis. If a token
/// flips, "layer 17 selected a different expert at rank 3" and "layer 17's
/// arithmetic is off in the last bit" are different bugs, and only the second
/// one is a tolerance question. Comparing the ids alone cannot tell them apart.
#[cfg(feature = "hip")]
fn run_device(a: &Args, fx: &Fixture, steps: usize, n_ctx: usize) -> i32 {
    use hipml_model::gpu::{GpuEngine, KvMode};
    use hipml_model::Model as Resident;

    // The CPU reference maps the same 22.85 GB file. Load it and run the prompt
    // first, then drop it, so it never overlaps with the resident weights on the
    // device: on Windows a second mapping at that point fails with "insufficient
    // system resources" before it reads a byte.
    let cpu = if a.routes {
        Some(cpu_routes(&a.gguf, &fx.prompt_ids))
    } else {
        None
    };

    let t0 = Instant::now();
    let (model, report) = match Resident::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("moe-gate: resident load: {e}");
            return 1;
        }
    };
    println!(
        "resident:  {:.2} GiB in {} tensors on {} (upload {:.1}s)",
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        report.tensor_count,
        model.device_name(),
        report.upload_secs
    );
    let mut engine = match GpuEngine::new(&model, n_ctx, KvMode::Q8) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("moe-gate: graph: {e}");
            return 1;
        }
    };
    let engine_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "moe-gate: {} ({} decoder layers, {} experts, {} used, expert ffn {}, shared expert {}, ffn {})",
        engine.cfg.arch,
        engine.cfg.n_layer,
        engine.cfg.n_expert,
        engine.cfg.n_expert_used,
        engine.cfg.n_ff_exp,
        if engine.cfg.has_shexp {
            engine.cfg.n_ff_shexp.to_string()
        } else {
            "absent".to_string()
        },
        engine.cfg.n_ff
    );
    println!(
        "graph:     {} layers, {:.2} GiB streamed/token, {:.1} MiB KV at ctx {n_ctx} \
         (build {engine_ms:.0} ms)",
        engine.cfg.n_layer,
        engine.weight_bytes() as f64 / (1u64 << 30) as f64,
        engine.kv_bytes(n_ctx) as f64 / (1024.0 * 1024.0)
    );

    let mut st = engine.new_state();
    let t_prefill = Instant::now();
    let first = match engine.prefill_chunked(&fx.prompt_ids, &mut st, a.chunk) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("moe-gate: prefill: {e}");
            return 1;
        }
    };
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    println!(
        "prefill:   {} ids in {prefill_s:.2}s ({:.1} tok/s, one decode step per token)",
        fx.prompt_ids.len(),
        fx.prompt_ids.len() as f64 / prefill_s.max(1e-9)
    );

    // Routing at the last prompt token: the step whose logits pick `first`.
    let gpu_routes = match engine.moe_routes() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("moe-gate: route readback: {e}");
            return 1;
        }
    };

    let t_decode = Instant::now();
    let mut ids: Vec<u32> = Vec::with_capacity(steps);
    let mut next = first;
    for step in 0..steps {
        ids.push(next);
        if step + 1 < steps {
            match engine.forward_token_id(next, &mut st) {
                Ok(id) => next = id,
                Err(e) => {
                    eprintln!("moe-gate: decode step {}: {e}", step + 1);
                    return 1;
                }
            }
        }
    }
    let decode_s = t_decode.elapsed().as_secs_f64();
    let decode_n = steps.saturating_sub(1).max(1);
    println!(
        "decode:    {decode_n} steps in {decode_s:.2}s ({:.2} tok/s)",
        decode_n as f64 / decode_s.max(1e-9)
    );
    println!("ids:       {ids:?}");

    // VRAM: everything this process holds is the model plus the engine's
    // scratch, so total minus free is the resident footprint.
    let mut rc = 0;
    if let Ok((free, total)) = model.device_mem_info() {
        println!(
            "vram:      {:.2} GiB resident of {:.2} GiB",
            (total - free) as f64 / (1u64 << 30) as f64,
            total as f64 / (1u64 << 30) as f64
        );
    }

    if a.routes {
        match cpu {
            Some(Ok(cpu)) => {
                let n = cpu.len().min(gpu_routes.len());
                let (mut same, mut reordered, mut different) = (0usize, 0usize, 0usize);
                let mut worst = 0.0f32;
                for i in 0..n {
                    let cpu_ids: Vec<u32> = cpu[i].ids.iter().map(|&e| e as u32).collect();
                    let gpu_ids = &gpu_routes[i].0;
                    if &cpu_ids == gpu_ids {
                        same += 1;
                    } else {
                        let mut cs = cpu_ids.clone();
                        let mut gs = gpu_ids.clone();
                        cs.sort_unstable();
                        gs.sort_unstable();
                        if cs == gs {
                            // The same experts in a different rank: two
                            // probabilities within the router's arithmetic
                            // noise. The fold is rank-ordered, so the output
                            // differs in the last bits, not in which expert ran.
                            reordered += 1;
                        } else {
                            different += 1;
                            if different <= 5 {
                                println!("  layer {i}: cpu {cpu_ids:?} / gpu {gpu_ids:?}");
                            }
                        }
                    }
                    // Compare each expert's weight by id, not by rank: a rank
                    // swap must not look like a weight error on two experts.
                    for (rank, &e) in cpu[i].ids.iter().enumerate() {
                        let e = e as u32;
                        if let Some(gr) = gpu_ids.iter().position(|&x| x == e) {
                            worst = worst.max((cpu[i].weights[rank] - gpu_routes[i].1[gr]).abs());
                        }
                    }
                }
                if n == 0 {
                    println!("routing:   no MoE layers on either side");
                } else {
                    println!(
                        "routing:   {n} layers; {same} identical, {reordered} rank-order only, \
                         {different} selected a different expert set"
                    );
                    println!("routing:   worst |dweight| {worst:.3e} over shared expert ids");
                    if different > 0 {
                        println!(
                            "routing:   the 64-id stream is the gate; a set difference at the \
                             last rank is router arithmetic, not a convention error"
                        );
                    }
                }
            }
            Some(Err(e)) => println!("routing:   cpu reference unavailable: {e}"),
            None => {}
        }
    }

    let n = fx.reference_ids.len().min(ids.len());
    let first_bad = (0..n).find(|&i| ids[i] != fx.reference_ids[i]);
    match first_bad {
        None if ids.len() >= fx.reference_ids.len() => {
            println!(
                "PASS: {} / {} ids identical to llama.cpp",
                fx.reference_ids.len(),
                fx.reference_ids.len()
            );
        }
        None => {
            println!(
                "FAIL: only {} of {} ids produced",
                ids.len(),
                fx.reference_ids.len()
            );
            rc = 1;
        }
        Some(i) => {
            println!(
                "FAIL: first divergence at step {i}: llama.cpp {}, gpu {}; \
                 llama.cpp tail {:?}, gpu tail {:?}",
                fx.reference_ids[i],
                ids[i],
                &fx.reference_ids[i..(i + 4).min(fx.reference_ids.len())],
                &ids[i..(i + 4).min(ids.len())]
            );
            rc = 1;
        }
    }
    rc
}

/// The CPU reference's router decisions for the last prompt token.
///
/// The prompt is fed exactly as `greedy_with_stop` feeds it — every token
/// through the same state — so the activation the router sees at the last
/// position is the one the GPU's last prefill step saw.
#[cfg(feature = "hip")]
fn cpu_routes(gguf: &str, prompt_ids: &[u32]) -> Result<Vec<hipml_core::moe::Route>, String> {
    let model = hipml_core::model::Model::load_file(gguf).map_err(|e| format!("{e}"))?;
    let mut st = model.new_state(prompt_ids.len() + 8);
    let (last, head) = prompt_ids.split_last().ok_or("empty prompt")?;
    for &id in head {
        let _ = model.forward_token(id, &mut st);
    }
    let mut routes: Vec<hipml_core::moe::Route> = Vec::new();
    let _ = model.forward_hidden_routes(*last, &mut st, &mut routes);
    Ok(routes)
}
