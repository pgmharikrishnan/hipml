//! `hipml serve`: an HTTP server with continuous batching, MTP for the lone
//! request, and the SSM+KV prefix cache.
//!
//! ```text
//! hipml serve --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf \
//!             --mtp  C:\models\mtp-Qwen3.8-27B-Q4_0.gguf \
//!             --port 8093 --slots 16 --ctx 32768
//! ```
//!
//! Modules: `http` is the transport, `api` shapes the two endpoints, `sched`
//! owns the GPU, `metrics` is the Prometheus text, `chat` renders and
//! tokenizes, `json` is the tiny reader/writer.

mod api;
mod chat;
mod http;
mod json;
mod metrics;
mod sched;

use std::net::TcpListener;
use std::sync::mpsc::channel;
use std::sync::Arc;

use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;
use hipml_model::Model;

use http::Shared;
use metrics::Metrics;
use sched::SchedCfg;

struct Args {
    gguf: String,
    mtp: Option<String>,
    host: String,
    port: u16,
    ctx: usize,
    slots: usize,
    chunk: usize,
    drafts: usize,
    prefix_entries: usize,
    prefix_mb: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        mtp: None,
        host: "127.0.0.1".to_string(),
        port: 8093,
        ctx: 32768,
        slots: 16,
        chunk: 256,
        drafts: 1,
        prefix_entries: 16,
        prefix_mb: 2048,
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
            "--mtp" => {
                a.mtp = Some(need(i)?.to_string());
                i += 2;
            }
            "--host" => {
                a.host = need(i)?.to_string();
                i += 2;
            }
            "--port" => {
                a.port = need(i)?.parse().map_err(|e| format!("bad --port: {e}"))?;
                i += 2;
            }
            "--ctx" => {
                a.ctx = need(i)?.parse().map_err(|e| format!("bad --ctx: {e}"))?;
                i += 2;
            }
            "--slots" => {
                a.slots = need(i)?.parse().map_err(|e| format!("bad --slots: {e}"))?;
                i += 2;
            }
            "--chunk" => {
                a.chunk = need(i)?.parse().map_err(|e| format!("bad --chunk: {e}"))?;
                i += 2;
            }
            "--drafts" => {
                a.drafts = need(i)?.parse().map_err(|e| format!("bad --drafts: {e}"))?;
                i += 2;
            }
            "--prefix-entries" => {
                a.prefix_entries = need(i)?
                    .parse()
                    .map_err(|e| format!("bad --prefix-entries: {e}"))?;
                i += 2;
            }
            "--prefix-mb" => {
                a.prefix_mb = need(i)?
                    .parse()
                    .map_err(|e| format!("bad --prefix-mb: {e}"))?;
                i += 2;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() {
        return Err("--gguf PATH is required".to_string());
    }
    if a.slots == 0 || a.slots > 32 {
        return Err("--slots must be in 1..=32".to_string());
    }
    if a.chunk == 0 {
        return Err("--chunk must be positive".to_string());
    }
    if a.drafts > 3 {
        return Err("--drafts must be in 0..=3".to_string());
    }
    Ok(a)
}

fn help() {
    eprintln!(
        "hipml serve --gguf PATH [--mtp PATH] [--port 8093] [--host 127.0.0.1]\n\
         \x20           [--ctx 32768] [--slots 16] [--chunk 256] [--drafts 1]\n\
         \x20           [--prefix-entries 16] [--prefix-mb 2048]\n\
         \n\
         endpoints: GET /health /metrics /props, POST /tokenize,\n\
         \x20          POST /v1/chat/completions (SSE with \"stream\":true),\n\
         \x20          POST /completion (llama-style, SSE)\n\
         \n\
         one request alone is decoded with MTP speculation; two or more are\n\
         decoded by the batched engine, with new prompts prefilled in tiles."
    );
}

pub fn run(args: &[String]) -> i32 {
    // Batched decode at B>=4 uses the WMMA skinny GEMV by default: it is
    // measured ~1.5x faster than the exact `_b` kernels at the slot counts a
    // fleet uses. B<4 keeps the exact path (the tile pads R to 16 and loses
    // there), and so does the lone MTP request, so a single stream is
    // unchanged. `HIPML_BATCH_LEGACY=1` restores the exact batched path.
    if std::env::var_os("HIPML_BATCH_LEGACY").is_none() {
        std::env::set_var("HIPML_BATCH_SKINNY", "1");
    }
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            if e != "help" {
                eprintln!("serve: {e}");
            }
            help();
            return if e == "help" { 0 } else { 2 };
        }
    };

    let (mut model, report) = match Model::load(&a.gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("serve: load {}: {e}", a.gguf);
            return 1;
        }
    };
    let mut sidecar_bytes = 0u64;
    if let Some(path) = &a.mtp {
        match model.load_sidecar(path) {
            Ok(b) => sidecar_bytes = b,
            Err(e) => {
                eprintln!("serve: sidecar {path}: {e}");
                return 1;
            }
        }
    }
    let g = match Gguf::open(&a.gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("serve: reopen {}: {e}", a.gguf);
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("serve: tokenizer: {e}");
            return 1;
        }
    };
    let Some(template) = tok.chat_template.clone() else {
        eprintln!("serve: the GGUF has no tokenizer.chat_template");
        return 1;
    };

    let cfg = SchedCfg {
        ctx: a.ctx,
        slots: a.slots,
        chunk: a.chunk,
        drafts: a.drafts,
        prefix_entries: a.prefix_entries,
        prefix_bytes: a.prefix_mb.max(1) * 1024 * 1024,
        mtp: a.mtp.is_some() && a.drafts > 0,
    };
    let metrics = Arc::new(Metrics::default());
    let listener = match TcpListener::bind((a.host.as_str(), a.port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("serve: bind {}:{}: {e}", a.host, a.port);
            return 1;
        }
    };
    let (tx, rx) = channel();
    let shared = Arc::new(Shared {
        tok: Arc::new(tok),
        template: Arc::from(template),
        metrics: Arc::clone(&metrics),
        jobs: tx,
        next_id: std::sync::atomic::AtomicU64::new(0),
        model: a.gguf.clone(),
        cfg,
    });

    println!(
        "hipml serve on http://{}:{} | {} tensors, {:.2} GiB resident + {:.2} GiB drafter\n\
         slots {} x ctx {} | chunk {} | drafts {} | prefix cache {} entries / {} MiB",
        a.host,
        a.port,
        report.tensor_count,
        report.tensor_bytes as f64 / (1u64 << 30) as f64,
        sidecar_bytes as f64 / (1u64 << 30) as f64,
        cfg.slots,
        cfg.ctx,
        cfg.chunk,
        cfg.drafts,
        cfg.prefix_entries,
        a.prefix_mb
    );

    let l = Arc::clone(&shared);
    std::thread::spawn(move || http::listen(listener, l));
    match sched::run(&model, rx, cfg, metrics) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("serve: scheduler: {e}");
            1
        }
    }
}
