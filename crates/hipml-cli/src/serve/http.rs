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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hipml_core::tokenizer::Tokenizer;

use super::api;
use super::chat;
use super::json::{self, Value};
use super::metrics::Metrics;
use super::sched::{Job, SchedCfg};

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
}

pub fn listen(listener: TcpListener, shared: Arc<Shared>) {
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let shared = shared.clone();
                std::thread::spawn(move || {
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
    let req = read_request(&mut stream)?;
    // Generation can legitimately wait minutes for its first token; the
    // timeout only guards the request read.
    let _ = stream.set_read_timeout(None);
    let path = req.path.clone();
    match (req.method.as_str(), path.as_str()) {
        ("GET", "/health") | ("GET", "/v1/health") => {
            write_json(&mut stream, "200 OK", "{\"status\":\"ok\"}")
        }
        ("GET", "/metrics") => {
            let vram = vram();
            let body = shared.metrics.render(vram);
            write_body(&mut stream, "200 OK", "text/plain; version=0.0.4", body.as_bytes())
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
        ("POST", p) if p.starts_with("/v1/chat/completions") || p.starts_with("/chat/completions") => {
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
        hipml_hip::Device::first().ok().and_then(|d| d.mem_info().ok())
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
    (&mut reader).take(MAX_LINE).read_line(&mut line).map_err(|e| e.to_string())?;
    if line.is_empty() {
        return Err("empty request".to_string());
    }
    let mut parts = line.trim_end().split_whitespace();
    let method = parts.next().ok_or("no method")?.to_string();
    let target = parts.next().ok_or("no path")?.to_string();
    let path = target.split_once('?').map(|(p, _)| p).unwrap_or(&target).to_string();

    let mut length = 0usize;
    let mut expect_continue = false;
    let mut chunked = false;
    for count in 0.. {
        if count == MAX_HEADERS {
            return Err("too many headers".to_string());
        }
        let mut h = String::new();
        let n = (&mut reader).take(MAX_LINE).read_line(&mut h).map_err(|e| e.to_string())?;
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
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
    Ok(Request {
        method,
        path,
        body,
    })
}

pub fn write_json(stream: &mut TcpStream, status: &str, body: &str) -> Result<(), String> {
    write_body(stream, status, "application/json", body.as_bytes())
}

pub fn write_body(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|_| stream.write_all(body))
        .and_then(|_| stream.flush())
        .map_err(|e| e.to_string())
}

/// A chunked `text/event-stream` writer. `event` frames one `data:` line.
pub struct Sse {
    stream: TcpStream,
}

impl Sse {
    pub fn start(stream: &mut TcpStream) -> Result<Sse, String> {
        stream
            .try_clone()
            .map_err(|e| e.to_string())
            .and_then(|mut s| {
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                            Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\
                            Connection: close\r\nX-Accel-Buffering: no\r\n\r\n";
                s.write_all(head.as_bytes())
                    .and_then(|_| s.flush())
                    .map(|_| Sse { stream: s })
                    .map_err(|e| e.to_string())
            })
    }

    pub fn event(&mut self, data: &str) -> Result<(), String> {
        let payload = format!("data: {data}\n\n");
        let head = format!("{:x}\r\n", payload.len());
        self.stream
            .write_all(head.as_bytes())
            .and_then(|_| self.stream.write_all(payload.as_bytes()))
            .and_then(|_| self.stream.write_all(b"\r\n"))
            .and_then(|_| self.stream.flush())
            .map_err(|e| e.to_string())
    }

    pub fn end(&mut self) -> Result<(), String> {
        self.stream
            .write_all(b"0\r\n\r\n")
            .and_then(|_| self.stream.flush())
            .map_err(|e| e.to_string())
    }
}

/// Incremental token -> text, holding back an incomplete UTF-8 sequence.
pub struct Text {
    bytes: Vec<u8>,
    taken: usize,
}

impl Text {
    pub fn new() -> Text {
        Text {
            bytes: Vec::new(),
            taken: 0,
        }
    }

    pub fn push(&mut self, tok: &Tokenizer, ids: &[u32]) -> String {
        for &id in ids {
            if tok.is_special(id) {
                continue;
            }
            self.bytes.extend_from_slice(&tok.token_bytes(id));
        }
        let rest = &self.bytes[self.taken..];
        match std::str::from_utf8(rest) {
            Ok(s) => {
                let out = s.to_string();
                self.taken = self.bytes.len();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid == 0 {
                    if e.error_len().is_some() {
                        // Invalid byte: drop it so the stream cannot stall.
                        self.taken += 1;
                    }
                    return String::new();
                }
                let out = String::from_utf8_lossy(&rest[..valid]).into_owned();
                self.taken += valid;
                out
            }
        }
    }

    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.bytes[self.taken..]).into_owned();
        self.taken = self.bytes.len();
        out
    }
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
pub fn new_job(shared: &Shared, prompt: chat::Prompt, max_tokens: usize, ignore_eos: bool,
               stop_strings: &[String], stop_ids: Vec<u32>, drafts: usize, cache: bool)
    -> Result<(u64, std::sync::mpsc::Receiver<super::sched::Event>, Instant, Arc<AtomicU8>), String>
{
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
