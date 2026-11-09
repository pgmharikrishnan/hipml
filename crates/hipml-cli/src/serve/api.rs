//! The two generation endpoints, in the shapes llama-server clients expect.
//!
//! `/v1/chat/completions` speaks OpenAI chunks; `/completion` speaks llama's
//! own `{"content","tokens","stop","timings"}` events. Both reduce to the same
//! scheduler job: prompt ids in, token ids out.

use std::net::TcpStream;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::chat;
use super::http::{new_job, now_unix, write_body, write_json, Shared, Sse, Text};
use super::json::{self, Value};
use super::sched::{Done, Event};

fn parse(body: &[u8]) -> Result<Value, String> {
    let s = std::str::from_utf8(body).map_err(|e| format!("body is not utf-8: {e}"))?;
    json::parse(s).map_err(|e| format!("json: {e}"))
}

fn bad(stream: &mut TcpStream, msg: &str) -> Result<(), String> {
    let body = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(msg));
    write_json(stream, "400 Bad Request", &body)
}

fn max_tokens(v: &Value, prompt: usize, ctx: usize) -> usize {
    let n = v
        .get("max_tokens")
        .or_else(|| v.get("max_completion_tokens"))
        .or_else(|| v.get("n_predict"))
        .and_then(Value::as_i64)
        .unwrap_or(128);
    let room = ctx.saturating_sub(prompt).max(1);
    if n <= 0 {
        room
    } else {
        (n as usize).min(room)
    }
}

fn strings(v: Option<&Value>) -> Vec<String> {
    v.and_then(Value::as_arr)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

fn token_ids(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_arr)
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
        .unwrap_or_default()
}

fn bool_field(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn finish_reason(done: &Done) -> &str {
    match done.finish.as_str() {
        "stop" => "stop",
        _ => "length",
    }
}

fn timings(done: &Done, ttft_ms: f64, decode_ms: f64) -> String {
    let prompt_s = done.prefill_ms / 1e3;
    let pred_s = decode_ms / 1e3;
    let pp = if prompt_s > 0.0 {
        done.prompt_tokens as f64 / prompt_s
    } else {
        0.0
    };
    let dp = if pred_s > 0.0 {
        done.completion_tokens as f64 / pred_s
    } else {
        0.0
    };
    format!(
        "\"prompt_n\":{},\"prompt_ms\":{:.3},\"prompt_per_second\":{:.3},\
         \"predicted_n\":{},\"predicted_ms\":{:.3},\"predicted_per_second\":{:.3},\
         \"ttft_ms\":{:.3},\"prefix_hit_tokens\":{}",
        done.prompt_tokens,
        done.prefill_ms,
        pp,
        done.completion_tokens,
        decode_ms,
        dp,
        ttft_ms,
        done.hit_tokens
    )
}

fn drain(rx: &Receiver<Event>, started: Instant) -> Result<(Vec<u32>, Done, f64, f64), String> {
    let mut ids = Vec::new();
    let mut first_at: Option<Instant> = None;
    loop {
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(Event::Tokens(t)) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                ids.extend_from_slice(&t);
            }
            Ok(Event::Done(d)) => {
                let first = first_at.unwrap_or_else(Instant::now);
                let now = Instant::now();
                return Ok((
                    ids,
                    d,
                    first.duration_since(started).as_secs_f64() * 1e3,
                    now.duration_since(first).as_secs_f64() * 1e3,
                ));
            }
            Ok(Event::Error(e)) => return Err(e),
            Err(e) => return Err(format!("scheduler stopped: {e}")),
        }
    }
}

pub fn chat_completion(
    shared: &Arc<Shared>,
    stream: &mut TcpStream,
    body: &[u8],
) -> Result<(), String> {
    let v = match parse(body) {
        Ok(v) => v,
        Err(e) => return bad(stream, &e),
    };
    let messages = match chat::messages(&v) {
        Ok(m) => m,
        Err(e) => return bad(stream, &e),
    };
    let prompt = match chat::chat(&shared.tok, &shared.template, &messages) {
        Ok(p) => p,
        Err(e) => return bad(stream, &e),
    };
    let max = max_tokens(&v, prompt.ids.len(), shared.cfg.ctx);
    let ignore_eos = bool_field(&v, "ignore_eos", false);
    let (id, rx, at) = match new_job(
        shared,
        prompt,
        max,
        ignore_eos,
        &strings(v.get("stop")),
        token_ids(v.get("stop_token_ids")),
        shared.cfg.drafts,
    ) {
        Ok(x) => x,
        Err(e) => return bad(stream, &e),
    };
    let model = json::escape(&shared.model);
    if bool_field(&v, "stream", false) {
        stream_chat(shared, stream, rx, at, id, &model)
    } else {
        let (ids, done, ttft, decode_ms) = match drain(&rx, at) {
            Ok(x) => x,
            Err(e) => {
                let b = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                return write_json(stream, "500 Internal Server Error", &b);
            }
        };
        let text = shared.tok.decode_text(&ids);
        let body = format!(
            "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion\",\"created\":{},\
             \"model\":\"{model}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\
             \"content\":\"{}\"}},\"finish_reason\":\"{}\"}}],\
             \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\
             \"timings\":{{{}}}}}",
            now_unix(),
            json::escape(&text),
            finish_reason(&done),
            done.prompt_tokens,
            done.completion_tokens,
            done.prompt_tokens + done.completion_tokens,
            timings(&done, ttft, decode_ms)
        );
        write_json(stream, "200 OK", &body)
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_chat(
    shared: &Arc<Shared>,
    stream: &mut TcpStream,
    rx: Receiver<Event>,
    at: Instant,
    id: u64,
    model: &str,
) -> Result<(), String> {
    let mut sse = Sse::start(stream)?;
    let mut text = Text::new();
    let mut first_at: Option<Instant> = None;
    let mut last_at = at;
    let role = format!(
        "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\
         \"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\
         \"finish_reason\":null}}]}}",
        now_unix()
    );
    sse.event(&role)?;
    loop {
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(Event::Tokens(t)) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                last_at = Instant::now();
                let delta = text.push(&shared.tok, &t);
                if !delta.is_empty() {
                    let event = format!(
                        "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\
                         \"created\":{},\"model\":\"{model}\",\"choices\":[{{\"index\":0,\
                         \"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
                        now_unix(),
                        json::escape(&delta)
                    );
                    sse.event(&event)?;
                }
            }
            Ok(Event::Done(done)) => {
                let tail = text.finish();
                if !tail.is_empty() {
                    let event = format!(
                        "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\
                         \"created\":{},\"model\":\"{model}\",\"choices\":[{{\"index\":0,\
                         \"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
                        now_unix(),
                        json::escape(&tail)
                    );
                    sse.event(&event)?;
                }
                let first = first_at.unwrap_or(at);
                let ttft = first.duration_since(at).as_secs_f64() * 1e3;
                let decode_ms = last_at.duration_since(first).as_secs_f64() * 1e3;
                let final_event = format!(
                    "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\
                     \"created\":{},\"model\":\"{model}\",\"choices\":[{{\"index\":0,\
                     \"delta\":{{}},\"finish_reason\":\"{}\"}}],\
                     \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\
                     \"total_tokens\":{}}},\"timings\":{{{}}}}}",
                    now_unix(),
                    finish_reason(&done),
                    done.prompt_tokens,
                    done.completion_tokens,
                    done.prompt_tokens + done.completion_tokens,
                    timings(&done, ttft, decode_ms)
                );
                sse.event(&final_event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
            Ok(Event::Error(e)) => {
                let event = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                sse.event(&event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
            Err(e) => {
                let event = format!(
                    "{{\"error\":{{\"message\":\"scheduler stopped: {}\"}}}}",
                    json::escape(&e.to_string())
                );
                sse.event(&event)?;
                return sse.end();
            }
        }
    }
}

pub fn completion(
    shared: &Arc<Shared>,
    stream: &mut TcpStream,
    body: &[u8],
) -> Result<(), String> {
    let v = match parse(body) {
        Ok(v) => v,
        Err(e) => return bad(stream, &e),
    };
    let prompt = match chat::completion(&shared.tok, &v) {
        Ok(p) => p,
        Err(e) => return bad(stream, &e),
    };
    let max = max_tokens(&v, prompt.ids.len(), shared.cfg.ctx);
    let ignore_eos = bool_field(&v, "ignore_eos", false);
    let (id, rx, at) = match new_job(
        shared,
        prompt,
        max,
        ignore_eos,
        &strings(v.get("stop")),
        token_ids(v.get("stop_token_ids")),
        shared.cfg.drafts,
    ) {
        Ok(x) => x,
        Err(e) => return bad(stream, &e),
    };
    let streaming = bool_field(&v, "stream", false);
    if !streaming {
        let (ids, done, ttft, decode_ms) = match drain(&rx, at) {
            Ok(x) => x,
            Err(e) => {
                let b = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                return write_json(stream, "500 Internal Server Error", &b);
            }
        };
        let text = shared.tok.decode_text(&ids);
        let body = format!(
            "{{\"content\":\"{}\",\"stop\":{},\"tokens\":{:?},\"timings\":{{{}}}}}",
            json::escape(&text),
            done.finish == "stop",
            ids,
            timings(&done, ttft, decode_ms)
        );
        write_body(stream, "200 OK", "application/json", body.as_bytes())
    } else {
        stream_completion(shared, stream, rx, at, id)
    }
}

fn stream_completion(
    shared: &Arc<Shared>,
    stream: &mut TcpStream,
    rx: Receiver<Event>,
    at: Instant,
    _id: u64,
) -> Result<(), String> {
    let mut sse = Sse::start(stream)?;
    let mut text = Text::new();
    let mut first_at: Option<Instant> = None;
    let mut last_at = at;
    loop {
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(Event::Tokens(t)) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                last_at = Instant::now();
                let delta = text.push(&shared.tok, &t);
                let event = format!(
                    "{{\"content\":\"{}\",\"stop\":false,\"tokens\":{:?}}}",
                    json::escape(&delta),
                    t
                );
                sse.event(&event)?;
            }
            Ok(Event::Done(done)) => {
                let first = first_at.unwrap_or(at);
                let ttft = first.duration_since(at).as_secs_f64() * 1e3;
                let decode_ms = last_at.duration_since(first).as_secs_f64() * 1e3;
                let event = format!(
                    "{{\"content\":\"\",\"stop\":{},\"tokens\":[],\"timings\":{{{}}}}}",
                    done.finish == "stop",
                    timings(&done, ttft, decode_ms)
                );
                sse.event(&event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
            Ok(Event::Error(e)) => {
                let event = format!("{{\"error\":\"{}\"}}", json::escape(&e));
                sse.event(&event)?;
                return sse.end();
            }
            Err(e) => {
                let event = format!("{{\"error\":\"{}\"}}", json::escape(&e.to_string()));
                sse.event(&event)?;
                return sse.end();
            }
        }
    }
}
