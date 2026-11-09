//! The HTTP wire layer: response writing, the chunked SSE writer and the
//! incremental token-to-text decoder.
//!
//! Split from `http.rs` so the request parser and the routing stay readable;
//! nothing here knows about the scheduler.

use std::io::Write;
use std::net::TcpStream;
use std::time::{SystemTime, UNIX_EPOCH};

use hipml_core::tokenizer::Tokenizer;

pub fn write_json(stream: &mut TcpStream, status: &str, body: &str) -> Result<(), String> {
    write_body(stream, status, "application/json", body.as_bytes())
}

/// `write_json` with one extra header line (`""` for none). Used to advertise
/// whether this server's batched decode is bit-exact.
pub fn write_json_h(
    stream: &mut TcpStream,
    status: &str,
    extra: &str,
    body: &str,
) -> Result<(), String> {
    write_body_h(stream, status, "application/json", extra, body.as_bytes())
}

pub fn write_body(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    write_body_h(stream, status, content_type, "", body)
}

pub fn write_body_h(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    extra: &str,
    body: &[u8],
) -> Result<(), String> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
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
    /// `extra` is one extra header line (`""` for none).
    pub fn start_h(stream: &mut TcpStream, extra: &str) -> Result<Sse, String> {
        stream
            .try_clone()
            .map_err(|e| e.to_string())
            .and_then(|mut s| {
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                     Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\
                     {extra}Connection: close\r\nX-Accel-Buffering: no\r\n\r\n"
                );
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
