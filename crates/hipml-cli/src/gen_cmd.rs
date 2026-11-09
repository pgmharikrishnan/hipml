//! `hipml generate`: load a GGUF, run the forward pass, print greedy tokens.

use std::time::Instant;

use hipml_core::gen::{greedy_with, logprob_of, top_logprobs};
use std::io::Write;
use hipml_core::gguf::Gguf;
use hipml_core::model::Model;
use hipml_core::tokenizer::Tokenizer;

struct Args {
    gguf: String,
    text: Option<String>,
    ids: Option<Vec<u32>>,
    tokens: usize,
    ctx: Option<usize>,
    topk: usize,
    logprobs: bool,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut gguf = None;
    let mut text = None;
    let mut ids = None;
    let mut tokens = 20usize;
    let mut ctx = None;
    let mut topk = 0usize;
    let mut logprobs = false;
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&str, String> {
            args.get(i + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--gguf" => {
                gguf = Some(need(i)?.to_string());
                i += 2;
            }
            "--text" => {
                text = Some(need(i)?.to_string());
                i += 2;
            }
            "--ids" => {
                let v = need(i)?;
                let parsed: Result<Vec<u32>, _> =
                    v.split(',').map(|s| s.trim().parse::<u32>()).collect();
                ids = Some(parsed.map_err(|e| format!("bad --ids: {e}"))?);
                i += 2;
            }
            "--tokens" => {
                tokens = need(i)?.parse().map_err(|e| format!("bad --tokens: {e}"))?;
                i += 2;
            }
            "--ctx" => {
                ctx = Some(need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?);
                i += 2;
            }
            "--topk" => {
                topk = need(i)?.parse().map_err(|e| format!("bad --topk: {e}"))?;
                i += 2;
            }
            "--logprobs" => {
                logprobs = true;
                i += 1;
            }
            // `--cpu` is the default oracle path; accept the flag so a script
            // can say which path it means.
            "--cpu" => i += 1,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        gguf: gguf.ok_or("--gguf PATH is required")?,
        text,
        ids,
        tokens,
        ctx,
        topk,
        logprobs,
    })
}

pub fn generate(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "--device") && !args.iter().any(|a| a == "--cpu") {
        #[cfg(feature = "hip")]
        return crate::device_cmd::generate(args);
        #[cfg(not(feature = "hip"))]
        {
            eprintln!("generate: --device needs a build with `--features hip`");
            return 2;
        }
    }
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("generate: {e}");
            return 2;
        }
    };

    let t0 = Instant::now();
    let g = match Gguf::open(&a.gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("generate: cannot open {}: {e}", a.gguf);
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("generate: tokenizer: {e}");
            return 1;
        }
    };
    let model = match Model::load_file(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("generate: load: {e}");
            return 1;
        }
    };
    let cfg = &model.cfg;
    println!(
        "loaded {} ({} layers: {} full attention, {} gated delta net) in {:.1}s",
        cfg.arch,
        cfg.n_layer,
        cfg.n_full_attn(),
        cfg.n_layer - cfg.n_full_attn(),
        t0.elapsed().as_secs_f64()
    );

    // Prompt: either explicit ids or text through the BPE tokenizer.
    let prompt_ids = if let Some(ids) = &a.ids {
        ids.clone()
    } else if let Some(text) = &a.text {
        tok.encode(text)
    } else {
        eprintln!("generate: need --text or --ids");
        return 2;
    };
    if prompt_ids.is_empty() {
        eprintln!("generate: empty prompt");
        return 2;
    }

    let n_ctx = a.ctx.unwrap_or(prompt_ids.len() + a.tokens + 8);
    println!("prompt ids: {prompt_ids:?}");

    let t1 = Instant::now();
    let mut steps_done = 0usize;
    let out = greedy_with(&model, &tok, &prompt_ids, a.tokens, n_ctx, |i, id, logits| {
        steps_done = i + 1;
        if a.logprobs {
            println!("  [{i}] {id} logprob {:.6}", logprob_of(logits, id as usize));
        } else {
            println!("  [{i}] {id}");
        }
        if a.topk > 0 && i == 0 {
            for (idx, lp) in top_logprobs(logits, a.topk) {
                println!("        {idx:6} logprob {lp:.6}");
            }
        }
        let _ = std::io::stdout().flush();
    });
    let secs = t1.elapsed().as_secs_f64();
    let steps = (prompt_ids.len() + steps_done.saturating_sub(1)) as f64;
    println!("gen ids:    {:?}", out.ids);
    println!("steps:      {steps:.0} tokens in {secs:.1}s ({:.2}s/token)", secs / steps.max(1.0));
    println!("text:       {}", out.text);
    0
}

/// `hipml matbench --gguf PATH --tensor NAME --reps N`: time one GEMV.
pub fn matbench(args: &[String]) -> i32 {
    let mut gguf = None;
    let mut name = String::new();
    let mut reps = 3usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                gguf = args.get(i + 1).cloned();
                i += 2;
            }
            "--tensor" => {
                name = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--reps" => {
                reps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            other => {
                eprintln!("matbench: unknown argument {other}");
                return 2;
            }
        }
    }
    let Some(path) = gguf else {
        eprintln!("matbench: --gguf PATH is required");
        return 2;
    };
    let mut g = match Gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("matbench: {e}");
            return 1;
        }
    };
    let m = match hipml_core::mat::Mat::load(&mut g, &name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("matbench: {e}");
            return 1;
        }
    };
    println!(
        "{name}: {}x{} {:?}, threads={:?}",
        m.rows,
        m.cols,
        m.kind,
        std::thread::available_parallelism()
    );
    let x = vec![0.01f32; m.cols];
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..reps {
        let y = m.gemv(&x);
        sink += y[0];
    }
    let secs = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{:.1} ms/gemv  {:.1} GB/s  (sink {sink})",
        secs * 1e3,
        m.data.len() as f64 / secs / 1e9
    );
    0
}

/// Escape a string for the tiny JSON writer below. The bench only emits ASCII
/// prompts, but escaping the full set keeps the output valid.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// `hipml tokenize --gguf PATH (--text STR | --text-file PATH) [--json] [--out PATH]`:
/// print ids and the round-tripped text. Useful for diffing against
/// llama-server's `/tokenize`, and for building the bench fixture.
pub fn tokenize(args: &[String]) -> i32 {
    let mut gguf = None;
    let mut text = None;
    let mut json = false;
    let mut special = false;
    let mut list_special = false;
    let mut out_path = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                gguf = args.get(i + 1).cloned();
                i += 2;
            }
            "--text" => {
                text = args.get(i + 1).cloned();
                i += 2;
            }
            "--text-file" => {
                let path = match args.get(i + 1) {
                    Some(p) => p,
                    None => {
                        eprintln!("tokenize: --text-file needs a path");
                        return 2;
                    }
                };
                match std::fs::read_to_string(path) {
                    Ok(s) => text = Some(s),
                    Err(e) => {
                        eprintln!("tokenize: read {path}: {e}");
                        return 1;
                    }
                }
                i += 2;
            }
            "--json" => {
                json = true;
                i += 1;
            }
            "--special" => {
                special = true;
                i += 1;
            }
            "--list-special" => {
                list_special = true;
                i += 1;
            }
            "--out" => {
                out_path = args.get(i + 1).cloned();
                i += 2;
            }
            other => {
                eprintln!("tokenize: unknown argument {other}");
                return 2;
            }
        }
    }
    let Some(path) = gguf else {
        eprintln!("tokenize: --gguf PATH is required");
        return 2;
    };
    let g = match Gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("tokenize: cannot open {path}: {e}");
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenize: {e}");
            return 1;
        }
    };
    if list_special {
        for (id, s) in tok.tokens.iter().enumerate() {
            if tok.is_special(id as u32) {
                println!("{}\t{}\t{:?}", id, tok.token_type(id as u32), s);
            }
        }
        return 0;
    }
    let text = match text {
        Some(t) => t,
        None => {
            eprintln!("tokenize: need --text or --text-file");
            return 2;
        }
    };
    let ids = if special {
        tok.encode_special(&text)
    } else {
        tok.encode(&text)
    };
    let rendered = if json {
        format!(
            "{{\"ids\":{:?},\"text\":\"{}\",\"text_skip_special\":\"{}\",\"add_bos\":{},\"bos_id\":{},\"eos_id\":{}}}",
            ids,
            json_escape(&tok.decode(&ids)),
            json_escape(&tok.decode_text(&ids)),
            tok.add_bos,
            tok.bos_id,
            tok.eos_id
        )
    } else {
        format!(
            "ids:  {ids:?}\ntext: {}\ntext (no specials): {}",
            tok.decode(&ids),
            tok.decode_text(&ids)
        )
    };
    if let Some(path) = out_path {
        if let Err(e) = std::fs::write(&path, &rendered) {
            eprintln!("tokenize: write {path}: {e}");
            return 1;
        }
    } else {
        println!("{rendered}");
    }
    0
}
