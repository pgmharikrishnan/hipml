// SPDX-License-Identifier: Apache-2.0
//! `hipml mtp-accept`: how often an MTP drafter agrees with greedy decode.
//!
//! Runs the CPU oracle greedily over a prompt and, at every generated position,
//! drafts `--depth` tokens ahead with the MTP block (recursively, feeding its
//! own output back). Both hidden-state conventions run side by side on the same
//! main-model pass. Reported per convention: P(draft k correct | drafts 1..k-1
//! correct) and the expected tokens per verification step for 1..depth drafts,
//! which is the whole speculative upside before the cost of the passes.
//!
//! ```text
//! hipml mtp-accept --gguf model.gguf --mtp mtp-sidecar.gguf --blk 64 \
//!     --ids-file prompt.ids --tokens 64 --depth 3
//! ```

use std::time::Instant;

use hipml_core::attention::KvState;
use hipml_core::gen::argmax_f32;
use hipml_core::model::Model;
use hipml_core::mtp::Mtp;

struct Args {
    gguf: String,
    mtp: String,
    blk: usize,
    ids: Vec<u32>,
    tokens: usize,
    depth: usize,
    stop_ids: Vec<u32>,
    stop_eos: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        mtp: String::new(),
        blk: 64,
        ids: Vec::new(),
        tokens: 64,
        depth: 3,
        stop_ids: Vec::new(),
        stop_eos: false,
    };
    let mut i = 0;
    while i < args.len() {
        let v = args
            .get(i + 1)
            .ok_or_else(|| format!("{} needs a value", args[i]))?;
        let num = |s: &str| s.parse::<usize>().map_err(|e| format!("{}: {e}", args[i]));
        match args[i].as_str() {
            "--gguf" => a.gguf = v.clone(),
            "--mtp" => a.mtp = v.clone(),
            "--blk" => a.blk = num(v)?,
            "--tokens" => a.tokens = num(v)?,
            "--depth" => a.depth = num(v)?.max(1),
            "--stop-eos" => {
                a.stop_eos = true;
                i += 1;
                continue;
            }
            "--stop-ids" => {
                a.stop_ids = v
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>().map_err(|e| format!("id {s}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--ids-file" => {
                let text = std::fs::read_to_string(v).map_err(|e| format!("{v}: {e}"))?;
                a.ids = text
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>().map_err(|e| format!("id {s}: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if a.gguf.is_empty() || a.mtp.is_empty() || a.ids.is_empty() {
        return Err("--gguf, --mtp and --ids-file are required".to_string());
    }
    Ok(a)
}

/// One convention's drafter and cache, and its drafts per generated index.
struct Lane {
    mtp: Mtp,
    kv: KvState,
    drafts: Vec<Vec<u32>>,
}

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("mtp-accept: {e}");
            return 2;
        }
    };
    let t0 = Instant::now();
    let model = match Model::load_file(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("mtp-accept: {}: {e}", a.gguf);
            return 1;
        }
    };
    let mut stop_ids = a.stop_ids.clone();
    if a.stop_eos {
        match hipml_core::gguf::Gguf::open(&a.gguf)
            .ok()
            .and_then(|g| g.meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32()))
        {
            Some(eos) => stop_ids.push(eos),
            None => eprintln!("mtp-accept: --stop-eos but the GGUF has no eos id"),
        }
    }
    let width = model.cfg.n_head_kv * model.cfg.head_dim;
    let n_ctx = a.ids.len() + a.tokens + a.depth + 2;
    let mut lanes = Vec::new();
    for post in [false, true] {
        match Mtp::load(&a.mtp, a.blk, post) {
            Ok(mtp) => lanes.push(Lane {
                mtp,
                kv: KvState::new(n_ctx, width),
                drafts: Vec::new(),
            }),
            Err(e) => {
                eprintln!("mtp-accept: {}: {e}", a.mtp);
                return 1;
            }
        }
    }
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());

    // Prompt: the drafter consumes (h_q, token_{q+1}) at position q.
    let t1 = Instant::now();
    let mut st = model.new_state(n_ctx);
    let mut h = Vec::new();
    for (p, &id) in a.ids.iter().enumerate() {
        if p > 0 {
            for l in &mut lanes {
                let hs = l.mtp.main_hidden(&model, &h);
                l.mtp.step(&model, &mut l.kv, &hs, id, p - 1, false);
            }
        }
        h = model.forward_hidden(id, &mut st);
    }
    let prompt_s = t1.elapsed().as_secs_f64();

    // Generate: after token P+i is known, draft P+i+1 .. P+i+depth.
    let t2 = Instant::now();
    let p0 = a.ids.len();
    let mut gen = Vec::with_capacity(a.tokens);
    let head = |x: &[f32]| {
        let n = hipml_core::ops::rmsnorm_f32(x, &model.w.output_norm, model.cfg.eps);
        argmax_f32(&model.w.output.gemv(&n)) as u32
    };
    let mut next = head(&h);
    let mut stopped = false;
    for i in 0..a.tokens {
        gen.push(next);
        if stop_ids.contains(&next) {
            stopped = true;
            break;
        }
        let q = p0 + i - 1;
        for l in &mut lanes {
            let mut hs = l.mtp.main_hidden(&model, &h);
            let mut tok = next;
            let mut chain = Vec::with_capacity(a.depth);
            for k in 0..a.depth {
                let (out, logits) = l.mtp.step(&model, &mut l.kv, &hs, tok, q + k, true);
                tok = argmax_f32(&logits) as u32;
                chain.push(tok);
                hs = l.mtp.self_hidden(&model, &out);
            }
            l.drafts.push(chain);
        }
        if i + 1 < a.tokens {
            h = model.forward_hidden(next, &mut st);
            next = head(&h);
        }
    }
    let gen_s = t2.elapsed().as_secs_f64();

    println!(
        "prompt {} tokens in {prompt_s:.1} s, {} generated in {gen_s:.1} s (stopped={stopped})",
        p0, a.tokens
    );
    let shown: Vec<String> = gen.iter().take(24).map(|t| t.to_string()).collect();
    println!("greedy ids: {}", shown.join(" "));
    for (li, l) in lanes.iter().enumerate() {
        report(
            if li == 0 {
                "pre-norm hidden "
            } else {
                "post-norm hidden"
            },
            l,
            &gen,
            a.depth,
        );
    }
    0
}

/// Conditional acceptance per draft index and tokens per step for 1..depth.
fn report(label: &str, l: &Lane, gen: &[u32], depth: usize) {
    // Leading matches at each index whose whole window is inside `gen`.
    let mut lead = Vec::new();
    for (i, d) in l.drafts.iter().enumerate() {
        if i + depth >= gen.len() {
            break;
        }
        let n = d
            .iter()
            .enumerate()
            .take_while(|(k, t)| gen[i + 1 + k] == **t)
            .count();
        lead.push(n);
    }
    let n = lead.len().max(1) as f64;
    let mut cond = Vec::new();
    for k in 0..depth {
        let reach = lead.iter().filter(|&&m| m >= k).count();
        let pass = lead.iter().filter(|&&m| m > k).count();
        cond.push(if reach == 0 {
            0.0
        } else {
            pass as f64 / reach as f64
        });
    }
    let per_step: Vec<String> = (1..=depth)
        .map(|g| {
            let e = lead.iter().map(|&m| m.min(g)).sum::<usize>() as f64 / n;
            format!("{g} draft(s) {:.3}", 1.0 + e)
        })
        .collect();
    let cs: Vec<String> = cond.iter().map(|c| format!("{c:.3}")).collect();
    println!(
        "{label}: n={} accept|prefix [{}]  tokens/step: {}",
        lead.len(),
        cs.join(", "),
        per_step.join(", ")
    );
}
