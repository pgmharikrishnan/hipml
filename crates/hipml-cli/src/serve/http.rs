// SPDX-License-Identifier: Apache-2.0
//! Hand-rolled HTTP/1.1: one request per connection, chunked SSE when asked.
//!
//! No external crates. The surface is what a benchmark client or `curl`
//! needs: `/health`, `/metrics`, `/props`, `/tokenize`, `/completion` and
//! `/v1/chat/completions`. The connection thread parses and tokenizes; the
//! scheduler owns the GPU.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hipml_core::tokenizer::Tokenizer;

use super::admit;
use super::api;
use super::chat;
use super::json::{self, Value};
use super::metrics::Metrics;
use super::sched::{Job, SchedCfg};
use super::wire::{write_body, write_json};

const MAX_BODY: usize = 32 * 1024 * 1024;
/// A client that opens a socket and never finishes its request would
/// otherwise hold a thread forever.
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HEADERS: usize = 100;
const MAX_LINE: u64 = 16 * 1024;

pub struct Shared {
    pub tok: Arc<Tokenizer>,
    pub template: Arc<str>,
    pub metrics: Arc<Metrics>,
    pub jobs: std::sync::mpsc::Sender<Job>,
    pub next_id: std::sync::atomic::AtomicU64,
    pub model: String,
    pub cfg: SchedCfg,
    /// WP51: admission bounds. `inflight` is `slots + queue_max`; `conns` is
    /// the thread bound. A request past either is answered 503, not queued.
    pub inflight: Arc<admit::Bound>,
    pub conns: Arc<admit::Bound>,
    /// WP51: false when batched decode at B>=4 uses the WMMA tile, which is
    /// faster but not bit-identical to the single-column launch.
    pub exact: bool,
}

/// Generation endpoints are the only ones that can hold a slot, so they are
/// the only ones that count against the admission bound.
fn is_generation(path: &str) -> bool {
    path == "/completion"
        || path.starts_with("/v1/chat/completions")
        || path.starts_with("/chat/completions")
}

pub fn listen(listener: TcpListener, shared: Arc<Shared>) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut s) => {
                let Some(conn) = shared.conns.try_enter() else {
                    let msg = format!("too many connections ({} open)", shared.conns.used());
                    let _ = admit::busy(&mut s, &msg, 1, &shared.metrics, true);
                    continue;
                };
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let _conn = conn;
                    if let Err(e) = handle(shared, s) {
                        eprintln!("serve: connection: {e}");
                    }
                });
            }
            Err(e) => eprintln!("serve: accept: {e}"),
        }
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn handle(shared: Arc<Shared>, mut stream: TcpStream) -> Result<(), String> {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(READ_TIMEOUT));
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            // Say why instead of closing silently; the client may be gone.
            let body = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
            let _ = write_json(&mut stream, "400 Bad Request", &body);
            return Err(e);
        }
    };
    // Generation can legitimately wait minutes for its first token; the
    // timeout only guards the request read.
    let _ = stream.set_read_timeout(None);
    let path = req.path.clone();
    // Backpressure: a generation request past the bound is told to come back
    // rather than queued behind every other request. `_slot` lives to the end
    // of this function, i.e. until the whole reply (or stream) is written.
    let _slot = if is_generation(&path) {
        match shared.inflight.try_enter() {
            Some(t) => Some(t),
            None => {
                let msg = format!(
                    "busy: {} slots and {} queued requests; retry shortly",
                    shared.cfg.slots,
                    shared.inflight.max() - shared.cfg.slots
                );
                return admit::busy(&mut stream, &msg, 1, &shared.metrics, false);
            }
        }
    } else {
        None
    };
    match (req.method.as_str(), path.as_str()) {
        ("GET", "/health") | ("GET", "/v1/health") => {
            write_json(&mut stream, "200 OK", "{\"status\":\"ok\"}")
        }
        ("GET", "/metrics") => {
            let vram = vram();
            let body = shared.metrics.render(vram);
            write_body(
                &mut stream,
                "200 OK",
                "text/plain; version=0.0.4",
                body.as_bytes(),
            )
        }
        ("GET", "/props") => {
            let body = format!(
                "{{\"model\":\"{}\",\"n_ctx\":{},\"n_slots\":{},\"drafts\":{},\"n_ctx_per_slot\":{}}}",
                json::escape(&shared.model),
                shared.cfg.ctx,
                shared.cfg.slots,
                shared.cfg.drafts,
                shared.cfg.ctx
            );
            write_json(&mut stream, "200 OK", &body)
        }
        ("POST", "/tokenize") => tokenize(&shared, &mut stream, &req),
        ("POST", p)
            if p.starts_with("/v1/chat/completions") || p.starts_with("/chat/completions") =>
        {
            api::chat_completion(&shared, &mut stream, &req.body)
        }
        ("POST", "/completion") => api::completion(&shared, &mut stream, &req.body),
        ("GET", "/") => write_json(&mut stream, "200 OK", "{\"status\":\"ok\",\"hipml\":true}"),
        _ => write_json(&mut stream, "404 Not Found", "{\"error\":\"not found\"}"),
    }
}

fn vram() -> Option<(usize, usize)> {
    #[cfg(feature = "hip")]
    {
        hipml_hip::Device::first()
            .ok()
            .and_then(|d| d.mem_info().ok())
    }
    #[cfg(not(feature = "hip"))]
    {
        None
    }
}

fn tokenize(shared: &Shared, stream: &mut TcpStream, req: &Request) -> Result<(), String> {
    let body = json::parse(std::str::from_utf8(&req.body).map_err(|e| e.to_string())?)
        .map_err(|e| format!("json: {e}"))?;
    let text = body
        .get("content")
        .or_else(|| body.get("text"))
        .and_then(Value::as_str)
        .ok_or("tokenize needs content")?;
    let ids = shared.tok.encode_special(text);
    let out = format!(
        "{{\"tokens\":{:?},\"n_tokens\":{},\"text\":\"{}\"}}",
        ids,
        ids.len(),
        json::escape(text)
    );
    write_json(stream, "200 OK", &out)
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    (&mut reader)
        .take(MAX_LINE)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if line.is_empty() {
        return Err("empty request".to_string());
    }
    if !line.ends_with('\n') {
        return Err("request line too long".to_string());
    }
    let mut parts = line.trim_end().split_whitespace();
    let method = parts.next().ok_or("no method")?.to_string();
    let target = parts.next().ok_or("no path")?.to_string();
    let path = target
        .split_once('?')
        .map(|(p, _)| p)
        .unwrap_or(&target)
        .to_string();

    let mut length = 0usize;
    let mut expect_continue = false;
    let mut chunked = false;
    for count in 0.. {
        if count == MAX_HEADERS {
            return Err("too many headers".to_string());
        }
        let mut h = String::new();
        let n = (&mut reader)
            .take(MAX_LINE)
            .read_line(&mut h)
            .map_err(|e| e.to_string())?;
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if !h.ends_with('\n') {
            return Err("header line too long".to_string());
        }
        if let Some((k, v)) = h.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            let v = v.trim();
            if k == "content-length" {
                length = v.parse().map_err(|_| "bad content-length")?;
            } else if k == "expect" && v.eq_ignore_ascii_case("100-continue") {
                expect_continue = true;
            } else if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            }
        }
    }
    if chunked {
        // Not implemented; say so instead of parsing an empty body.
        return Err("chunked request bodies are not supported; send Content-Length".to_string());
    }
    if length > MAX_BODY {
        return Err(format!("body too large: {length}"));
    }
    if expect_continue {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|e| e.to_string())?;
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body })
}

#[allow(clippy::too_many_arguments)]
pub fn new_job(
    shared: &Shared,
    prompt: chat::Prompt,
    max_tokens: usize,
    ignore_eos: bool,
    stop_strings: &[String],
    stop_ids: Vec<u32>,
    drafts: usize,
    cache: bool,
) -> Result<
    (
        u64,
        std::sync::mpsc::Receiver<super::sched::Event>,
        Instant,
        Arc<AtomicU8>,
    ),
    String,
> {
    let (tx, rx) = channel();
    let id = shared.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    let mut stops = stop_ids;
    if !ignore_eos && !stops.contains(&shared.tok.eos_id) {
        stops.push(shared.tok.eos_id);
    }
    for s in stop_strings {
        let ids = shared.tok.encode(s);
        if ids.len() == 1 {
            stops.push(ids[0]);
        }
    }
    stops.sort_unstable();
    stops.dedup();
    let cancel = Arc::new(AtomicU8::new(0));
    let job = Job {
        prompt: prompt.ids,
        cuts: prompt.cuts,
        max_tokens,
        stop_ids: stops,
        drafts,
        cache,
        tx,
        cancel: cancel.clone(),
    };
    let at = Instant::now();
    shared
        .jobs
        .send(job)
        .map_err(|_| "scheduler is not running".to_string())?;
    Ok((id, rx, at, cancel))
}
