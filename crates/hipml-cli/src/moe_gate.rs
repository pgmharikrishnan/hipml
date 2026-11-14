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
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        fixture: String::new(),
        steps: None,
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
    if let Some(l) = &fx.llama {
        println!("reference: {l}");
    }
    if let Some(c) = &fx.command {
        println!("command:   {c}");
    }
    if cfg.n_expert == 0 {
        eprintln!(
            "moe-gate: {} declares no experts; this gate is for MoE files",
            a.gguf
        );
        return 1;
    }

    let steps = a.steps.unwrap_or(fx.reference_ids.len());
    let n_ctx = fx.prompt_ids.len() + steps + 8;
    println!(
        "prompt:    {} ids, {} reference ids",
        fx.prompt_ids.len(),
        fx.reference_ids.len()
    );

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
