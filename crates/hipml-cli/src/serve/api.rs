// SPDX-License-Identifier: Apache-2.0
//! The two generation endpoints, in the shapes llama-server clients expect.
//!
//! `/v1/chat/completions` speaks OpenAI chunks; `/completion` speaks llama's
//! own `{"content","tokens","stop","timings"}` events. Both reduce to the same
//! scheduler job: prompt ids in, token ids out.

use std::io::ErrorKind;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::chat;
use super::http::{new_job, Shared};
use super::json::{self, Value};
use super::reply::{self, Delta, Shaper};
use super::sched::{Done, Event, Lp, LpReq, CANCEL_GONE, CANCEL_STOP};
use super::wire::{now_unix, write_body_h, write_json, write_json_h, Sse, Text};
use hipml_core::tokenizer::Tokenizer;
use hipml_core::tool_json;

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

/// `stop` is a string or an array of strings in both APIs.
fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Str(s)) => vec![s.clone()],
        Some(Value::Arr(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

/// Stop strings, matched on the decoded text. A stop string that is one token
/// also stops the scheduler directly (`new_job`); every other one is caught
/// here, the text is cut before it, and the request is told to end. Text that
/// could still grow into a stop string is held back until it cannot.
struct Stops {
    pats: Vec<String>,
    held: String,
    hit: bool,
}

impl Stops {
    fn new(mut pats: Vec<String>) -> Stops {
        pats.retain(|p| !p.is_empty());
        Stops {
            pats,
            held: String::new(),
            hit: false,
        }
    }

    /// Feed decoded text; returns the part that may be sent now.
    fn feed(&mut self, s: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.held.push_str(s);
        if let Some(at) = self
            .pats
            .iter()
            .filter_map(|p| self.held.find(p.as_str()))
            .min()
        {
            self.hit = true;
            let out = self.held[..at].to_string();
            self.held.clear();
            return out;
        }
        // Keep the longest suffix that is a proper prefix of some pattern.
        let mut keep = 0;
        for p in &self.pats {
            for (i, _) in self.held.char_indices() {
                let suffix = &self.held[i..];
                if p.starts_with(suffix) {
                    keep = keep.max(suffix.len());
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        self.held.drain(..cut).collect()
    }

    fn finish(&mut self) -> String {
        if self.hit {
            String::new()
        } else {
            std::mem::take(&mut self.held)
        }
    }
}

/// True when the peer has closed (or reset) its side. Used while a request
/// waits for tokens, so a client that hangs up during a queue wait or a long
/// prefill is noticed without having to write to it. A client that half-closes
/// its sending side after the request also reads as gone; HTTP clients do not.
fn peer_closed(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut b = [0u8; 1];
    let closed = match stream.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != ErrorKind::WouldBlock,
    };
    let _ = stream.set_nonblocking(false);
    closed
}

/// The next scheduler event, watching the socket meanwhile. On a hang-up the
/// request is cancelled so the scheduler frees its slot (or its prefill).
///
/// The socket is checked every 250 ms of wall time, not only when the
/// channel is quiet: a decoding slot sends a token every few tens of ms, so a
/// non-streaming client that hangs up would otherwise go unnoticed until the
/// whole reply was generated. `peek` carries the last check across calls.
fn next_event(
    rx: &Receiver<Event>,
    stream: &TcpStream,
    cancel: &AtomicU8,
    peek: &mut Instant,
) -> Result<Event, String> {
    let every = Duration::from_millis(250);
    let deadline = Instant::now() + Duration::from_secs(3600);
    loop {
        if peek.elapsed() >= every {
            *peek = Instant::now();
            if peer_closed(stream) {
                cancel.store(CANCEL_GONE, Ordering::Relaxed);
                return Err("client disconnected".to_string());
            }
        }
        match rx.recv_timeout(every) {
            Ok(ev) => return Ok(ev),
            Err(RecvTimeoutError::Timeout) => {
                if Instant::now() > deadline {
                    cancel.store(CANCEL_GONE, Ordering::Relaxed);
                    return Err("scheduler timed out".to_string());
                }
            }
            Err(RecvTimeoutError::Disconnected) => return Err("scheduler stopped".to_string()),
        }
    }
}

/// Decoded text for a run of ids, after stop strings. Sets the cancel flag
/// the first time a stop string matches.
fn deliver(
    shared: &Shared,
    text: &mut Text,
    stops: &mut Stops,
    cancel: &AtomicU8,
    t: &[u32],
) -> String {
    if stops.hit {
        return String::new();
    }
    let out = stops.feed(&text.push(&shared.tok, t));
    if stops.hit {
        cancel.store(CANCEL_STOP, Ordering::Relaxed);
    }
    out
}

/// The undelivered tail at `Done`, and the finish reason it implies.
fn tail(text: &mut Text, stops: &mut Stops, done: &mut Done) -> String {
    let mut out = stops.feed(&text.finish());
    out.push_str(&stops.finish());
    if stops.hit {
        done.finish = "stop".to_string();
    }
    out
}

fn token_ids(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_arr)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

fn bool_field(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn finish_reason(done: &Done, calls: usize) -> &'static str {
    if calls > 0 {
        return "tool_calls";
    }
    match done.finish.as_str() {
        "stop" => "stop",
        _ => "length",
    }
}

/// `choices[0].message` in the OpenAI shape. `reasoning_content` is present
/// only when the prompt opened a think block, and `tool_calls` only when the
/// reply is one — the same fields llama-server's default reasoning format adds.
fn message(reply: &reply::Reply, id: u64) -> String {
    let mut m = format!(
        "{{\"role\":\"assistant\",\"content\":\"{}\"",
        json::escape(&reply.content)
    );
    if let Some(reasoning) = &reply.reasoning {
        m.push_str(",\"reasoning_content\":\"");
        m.push_str(&json::escape(reasoning));
        m.push('"');
    }
    if !reply.calls.is_empty() {
        m.push_str(",\"tool_calls\":[");
        for (i, call) in reply.calls.iter().enumerate() {
            if i > 0 {
                m.push(',');
            }
            m.push_str(&format!(
                "{{\"id\":\"call_{id}_{i}\",\"type\":\"function\",\"function\":\
                 {{\"name\":\"{}\",\"arguments\":\"{}\"}}}}",
                json::escape(&call.name),
                json::escape(&tool_json::arguments_json(&call.arguments))
            ));
        }
        m.push(']');
    }
    m.push('}');
    m
}

/// This server decodes greedily: `argmax` over the logits, one id a step. A
/// request that asks for anything else would be silently served the greedy
/// answer, which is worse than a refusal — a client that set
/// `temperature: 0.8` would believe it got a sample. So `temperature > 0` and
/// `n > 1` are a 400 that says so.
///
/// `top_p` and `seed` are accepted: under greedy decode `top_p` cannot change
/// the argmax (the highest-probability token is inside every nucleus), and a
/// seed has nothing to seed. Both are no-ops by arithmetic, not by neglect.
fn greedy_only(v: &Value) -> Result<(), String> {
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        if t > 0.0 {
            return Err(format!(
                "temperature {t} is not supported: this server is greedy-only \
                 (temperature must be 0 or absent). Sampling is refused rather \
                 than silently ignored."
            ));
        }
    }
    if let Some(n) = v.get("n").and_then(Value::as_i64) {
        if n > 1 {
            return Err(format!(
                "n {n} is not supported: this server is greedy-only and returns \
                 one choice per request."
            ));
        }
    }
    Ok(())
}

/// `logprobs` and `top_logprobs`. The rules are llama-server's where llama has
/// one, so a client cannot get a different answer from the same body: a
/// non-boolean `logprobs` is a `400`, `top_logprobs` without `logprobs` is a
/// `400`, and an omitted `top_logprobs` means 20, which is llama's default.
///
/// One deliberate difference: the depth is bounded at 20 (llama accepts more)
/// because the depth is what the top-k selection costs, and a reply carrying
/// 200 alternatives is not a thing a client meant to ask for.
fn logprob_params(v: &Value) -> Result<LpReq, String> {
    let on = match v.get("logprobs") {
        None | Some(Value::Bool(false)) => false,
        Some(Value::Bool(true)) => true,
        Some(_) => return Err("logprobs must be a boolean".to_string()),
    };
    if !on && v.get("top_logprobs").is_some() {
        return Err("top_logprobs requires logprobs to be true".to_string());
    }
    let top = match v.get("top_logprobs") {
        None => 20,
        Some(x) => match x.as_i64() {
            Some(n) if (0..=20).contains(&n) => n as usize,
            _ => return Err("top_logprobs must be an integer between 0 and 20".to_string()),
        },
    };
    Ok(LpReq { on, top })
}

/// One `logprobs.content[]` element, in the shape llama-server emits: the id,
/// the token's text, its raw bytes, its log-softmax, and the top-k
/// alternatives in the same shape. `bytes` is the token's bytes rather than the
/// text's, so a multi-byte character split across two tokens stays recoverable.
fn logprob_entry(tok: &Tokenizer, id: u32, lp: &Lp) -> String {
    let bytes = tok.token_bytes(id);
    let mut out = format!(
        "{{\"id\":{id},\"token\":\"{}\",\"bytes\":{:?},\"logprob\":{:?},\"top_logprobs\":[",
        json::escape(&String::from_utf8_lossy(&bytes)),
        bytes,
        lp.logprob
    );
    for (i, &(id, v)) in lp.top.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let b = tok.token_bytes(id);
        out.push_str(&format!(
            "{{\"id\":{id},\"token\":\"{}\",\"bytes\":{:?},\"logprob\":{:?}}}",
            json::escape(&String::from_utf8_lossy(&b)),
            b,
            v
        ));
    }
    out.push_str("]}");
    out
}

/// `choices[].logprobs` for a run of ids, comma-prefixed so it drops into the
/// choice object, or `""` when the request asked for none.
fn logprobs_json(tok: &Tokenizer, ids: &[u32], lps: &[Lp]) -> String {
    if lps.is_empty() {
        return String::new();
    }
    let content: Vec<String> = ids
        .iter()
        .zip(lps)
        .map(|(&id, lp)| logprob_entry(tok, id, lp))
        .collect();
    format!(",\"logprobs\":{{\"content\":[{}]}}", content.join(","))
}

/// The header that says whether batched decode can change an answer.
fn exact_header(shared: &Shared) -> &'static str {
    if shared.exact {
        "X-Hipml-Batch-Exact: 1\r\n"
    } else {
        "X-Hipml-Batch-Exact: 0\r\nX-Hipml-Batch-Tile-Min: 4\r\n"
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

/// A whole non-streamed reply: delivered text, the ids behind it, the logprob
/// entries when the request asked for them, the summary, TTFT and decode time.
type Reply = (String, Vec<u32>, Vec<Lp>, Done, f64, f64);

fn drain(
    shared: &Shared,
    stream: &TcpStream,
    rx: &Receiver<Event>,
    cancel: &AtomicU8,
    mut stops: Stops,
    started: Instant,
    mut text: Text,
) -> Result<Reply, String> {
    let mut ids = Vec::new();
    let mut lps: Vec<Lp> = Vec::new();
    let mut out = String::new();
    let mut first_at: Option<Instant> = None;
    let mut peek = Instant::now();
    loop {
        match next_event(rx, stream, cancel, &mut peek)? {
            Event::Tokens(t, lp) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                if !stops.hit {
                    ids.extend_from_slice(&t);
                    if let Some(lp) = lp {
                        lps.extend(lp);
                    }
                }
                out.push_str(&deliver(shared, &mut text, &mut stops, cancel, &t));
            }
            Event::Done(mut d) => {
                out.push_str(&tail(&mut text, &mut stops, &mut d));
                let first = first_at.unwrap_or_else(Instant::now);
                let now = Instant::now();
                return Ok((
                    out,
                    ids,
                    lps,
                    d,
                    first.duration_since(started).as_secs_f64() * 1e3,
                    now.duration_since(first).as_secs_f64() * 1e3,
                ));
            }
            Event::Error(e) => return Err(e),
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
    if let Err(e) = greedy_only(&v) {
        return bad(stream, &e);
    }
    let lp = match logprob_params(&v) {
        Ok(p) => p,
        Err(e) => return bad(stream, &e),
    };
    let messages = match chat::messages(&v) {
        Ok(m) => m,
        Err(e) => return bad(stream, &e),
    };
    let tools = chat::tools(&v);
    let prompt = match chat::chat(&shared.tok, &shared.template, &messages, &tools) {
        Ok(p) => p,
        Err(e) => return bad(stream, &e),
    };
    let think = prompt.think;
    let max = max_tokens(&v, prompt.ids.len(), shared.cfg.ctx);
    let ignore_eos = bool_field(&v, "ignore_eos", false);
    let stop = strings(v.get("stop"));
    let (id, rx, at, cancel) = match new_job(
        shared,
        prompt,
        max,
        ignore_eos,
        &stop,
        token_ids(v.get("stop_token_ids")),
        shared.cfg.drafts,
        bool_field(&v, "cache_prompt", true),
        lp,
    ) {
        Ok(x) => x,
        Err(e) => return bad(stream, &e),
    };
    let model = json::escape(&shared.model);
    let stops = Stops::new(stop);
    if bool_field(&v, "stream", false) {
        stream_chat(shared, stream, rx, at, id, &model, &cancel, stops, think)
    } else {
        let (text, ids, lps, done, ttft, decode_ms) =
            match drain(shared, stream, &rx, &cancel, stops, at, Text::chat()) {
                Ok(x) => x,
                Err(e) => {
                    let b = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                    return write_json(stream, "500 Internal Server Error", &b);
                }
            };
        let reply = reply::shape(&text, think);
        let body = format!(
            "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion\",\"created\":{},\
             \"model\":\"{model}\",\"choices\":[{{\"index\":0,\"message\":{},\
             \"finish_reason\":\"{}\"{}}}],\
             \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\
             \"timings\":{{{}}}}}",
            now_unix(),
            message(&reply, id),
            finish_reason(&done, reply.calls.len()),
            logprobs_json(&shared.tok, &ids, &lps),
            done.prompt_tokens,
            done.completion_tokens,
            done.prompt_tokens + done.completion_tokens,
            timings(&done, ttft, decode_ms)
        );
        write_json_h(stream, "200 OK", exact_header(shared), &body)
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
    cancel: &AtomicU8,
    mut stops: Stops,
    think: bool,
) -> Result<(), String> {
    let mut sse = Sse::start_h(stream, exact_header(shared))?;
    let mut text = Text::chat();
    let mut shaper = Shaper::new(think);
    let mut calls = 0usize;
    // Logprob entries whose text the shaper has not emitted yet, so an entry
    // travels with the chunk that carries its token instead of arriving early.
    let mut held_ids: Vec<u32> = Vec::new();
    let mut held: Vec<Lp> = Vec::new();
    let mut first_at: Option<Instant> = None;
    let mut last_at = at;
    let role = format!(
        "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\"created\":{},\
         \"model\":\"{model}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\
         \"finish_reason\":null}}]}}",
        now_unix()
    );
    sse.event(&role)?;
    let mut peek = Instant::now();
    loop {
        let ev = match next_event(&rx, stream, cancel, &mut peek) {
            Ok(ev) => ev,
            Err(e) => {
                let event = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                sse.event(&event)?;
                return sse.end();
            }
        };
        match ev {
            Event::Tokens(t, lp) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                last_at = Instant::now();
                let delta = deliver(shared, &mut text, &mut stops, cancel, &t);
                if let Some(lp) = lp {
                    held_ids.extend_from_slice(&t);
                    held.extend(lp);
                }
                if !delta.is_empty() {
                    for piece in shaper.feed(&delta) {
                        emit(
                            &mut sse,
                            &shared.tok,
                            &piece,
                            id,
                            model,
                            &mut calls,
                            &mut held_ids,
                            &mut held,
                        )?;
                    }
                }
            }
            Event::Done(mut done) => {
                let tail = tail(&mut text, &mut stops, &mut done);
                for piece in shaper.feed(&tail).into_iter().chain(shaper.finish()) {
                    emit(
                        &mut sse,
                        &shared.tok,
                        &piece,
                        id,
                        model,
                        &mut calls,
                        &mut held_ids,
                        &mut held,
                    )?;
                }
                // An entry whose token produced no text (the turn's end token)
                // still has to be delivered, so one empty chunk carries it.
                if !held.is_empty() {
                    emit(
                        &mut sse,
                        &shared.tok,
                        &Delta::Content(String::new()),
                        id,
                        model,
                        &mut calls,
                        &mut held_ids,
                        &mut held,
                    )?;
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
                    finish_reason(&done, calls),
                    done.prompt_tokens,
                    done.completion_tokens,
                    done.prompt_tokens + done.completion_tokens,
                    timings(&done, ttft, decode_ms)
                );
                sse.event(&final_event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
            Event::Error(e) => {
                let event = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                sse.event(&event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
        }
    }
}

/// Emit one shaped piece, carrying whatever logprob entries were held back
/// with it. The first piece after a token flushes them; later pieces have none.
#[allow(clippy::too_many_arguments)]
fn emit(
    sse: &mut Sse,
    tok: &Tokenizer,
    piece: &Delta,
    id: u64,
    model: &str,
    calls: &mut usize,
    held_ids: &mut Vec<u32>,
    held: &mut Vec<Lp>,
) -> Result<(), String> {
    let lp = logprobs_json(tok, held_ids, held);
    held_ids.clear();
    held.clear();
    chunk(sse, piece, id, model, calls, &lp)
}

/// One streamed piece as an OpenAI chunk delta: reasoning and answer text carry
/// their own field, and a tool call carries `index`, `id`, `type` and the whole
/// `function` (llama streams the arguments in fragments; one piece is the same
/// call, assembled). `lp` is the choice's `logprobs` object, comma-prefixed, or
/// `""` when the request asked for none.
fn chunk(
    sse: &mut Sse,
    piece: &Delta,
    id: u64,
    model: &str,
    calls: &mut usize,
    lp: &str,
) -> Result<(), String> {
    let delta = match piece {
        Delta::Reasoning(t) => format!("\"reasoning_content\":\"{}\"", json::escape(t)),
        Delta::Content(t) => format!("\"content\":\"{}\"", json::escape(t)),
        Delta::Call(call) => {
            let index = *calls;
            *calls += 1;
            format!(
                "\"tool_calls\":[{{\"index\":{index},\"id\":\"call_{id}_{index}\",\
                 \"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}]",
                json::escape(&call.name),
                json::escape(&tool_json::arguments_json(&call.arguments))
            )
        }
    };
    let event = format!(
        "{{\"id\":\"chatcmpl-{id}\",\"object\":\"chat.completion.chunk\",\
         \"created\":{},\"model\":\"{model}\",\"choices\":[{{\"index\":0,\
         \"delta\":{{{delta}}}{lp},\"finish_reason\":null}}]}}",
        now_unix()
    );
    sse.event(&event)
}

pub fn completion(shared: &Arc<Shared>, stream: &mut TcpStream, body: &[u8]) -> Result<(), String> {
    let v = match parse(body) {
        Ok(v) => v,
        Err(e) => return bad(stream, &e),
    };
    if let Err(e) = greedy_only(&v) {
        return bad(stream, &e);
    }
    let prompt = match chat::completion(&shared.tok, &v) {
        Ok(p) => p,
        Err(e) => return bad(stream, &e),
    };
    let max = max_tokens(&v, prompt.ids.len(), shared.cfg.ctx);
    let ignore_eos = bool_field(&v, "ignore_eos", false);
    let stop = strings(v.get("stop"));
    let (id, rx, at, cancel) = match new_job(
        shared,
        prompt,
        max,
        ignore_eos,
        &stop,
        token_ids(v.get("stop_token_ids")),
        shared.cfg.drafts,
        bool_field(&v, "cache_prompt", true),
        // `logprobs` is an OpenAI field; llama's own `/completion` asks for
        // probabilities with `n_probs`, which this server does not implement.
        LpReq { on: false, top: 0 },
    ) {
        Ok(x) => x,
        Err(e) => return bad(stream, &e),
    };
    let streaming = bool_field(&v, "stream", false);
    let stops = Stops::new(stop);
    if !streaming {
        let (text, ids, _lps, done, ttft, decode_ms) =
            match drain(shared, stream, &rx, &cancel, stops, at, Text::new()) {
                Ok(x) => x,
                Err(e) => {
                    let b = format!("{{\"error\":{{\"message\":\"{}\"}}}}", json::escape(&e));
                    return write_json(stream, "500 Internal Server Error", &b);
                }
            };
        let body = format!(
            "{{\"content\":\"{}\",\"stop\":{},\"tokens\":{:?},\"timings\":{{{}}}}}",
            json::escape(&text),
            done.finish == "stop",
            ids,
            timings(&done, ttft, decode_ms)
        );
        write_body_h(
            stream,
            "200 OK",
            "application/json",
            exact_header(shared),
            body.as_bytes(),
        )
    } else {
        stream_completion(shared, stream, rx, at, id, &cancel, stops)
    }
}

fn stream_completion(
    shared: &Arc<Shared>,
    stream: &mut TcpStream,
    rx: Receiver<Event>,
    at: Instant,
    _id: u64,
    cancel: &AtomicU8,
    mut stops: Stops,
) -> Result<(), String> {
    let mut sse = Sse::start_h(stream, exact_header(shared))?;
    let mut text = Text::new();
    let mut first_at: Option<Instant> = None;
    let mut last_at = at;
    let mut peek = Instant::now();
    loop {
        let ev = match next_event(&rx, stream, cancel, &mut peek) {
            Ok(ev) => ev,
            Err(e) => {
                let event = format!("{{\"error\":\"{}\"}}", json::escape(&e));
                sse.event(&event)?;
                return sse.end();
            }
        };
        match ev {
            Event::Tokens(t, _lp) => {
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                if stops.hit {
                    continue;
                }
                last_at = Instant::now();
                let delta = deliver(shared, &mut text, &mut stops, cancel, &t);
                let event = format!(
                    "{{\"content\":\"{}\",\"stop\":false,\"tokens\":{:?}}}",
                    json::escape(&delta),
                    t
                );
                sse.event(&event)?;
            }
            Event::Done(mut done) => {
                let tail = tail(&mut text, &mut stops, &mut done);
                let first = first_at.unwrap_or(at);
                let ttft = first.duration_since(at).as_secs_f64() * 1e3;
                let decode_ms = last_at.duration_since(first).as_secs_f64() * 1e3;
                let event = format!(
                    "{{\"content\":\"{}\",\"stop\":{},\"tokens\":[],\"timings\":{{{}}}}}",
                    json::escape(&tail),
                    done.finish == "stop",
                    timings(&done, ttft, decode_ms)
                );
                sse.event(&event)?;
                sse.event("[DONE]")?;
                return sse.end();
            }
            Event::Error(e) => {
                let event = format!("{{\"error\":\"{}\"}}", json::escape(&e));
                sse.event(&event)?;
                return sse.end();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{greedy_only, Stops};
    use crate::serve::json::parse;

    fn check(body: &str) -> Result<(), String> {
        greedy_only(&parse(body).expect("valid json"))
    }

    #[test]
    fn greedy_is_accepted_and_sampling_is_refused() {
        assert!(check(r#"{"prompt": [1, 2]}"#).is_ok());
        assert!(check(r#"{"prompt": [1], "temperature": 0}"#).is_ok());
        assert!(check(r#"{"prompt": [1], "temperature": 0.0, "n": 1}"#).is_ok());
        // `top_p` and `seed` are no-ops under greedy decode, not lies.
        assert!(check(r#"{"prompt": [1], "temperature": 0, "top_p": 0.9, "seed": 7}"#).is_ok());
        let e = check(r#"{"prompt": [1], "temperature": 0.8}"#).unwrap_err();
        assert!(e.contains("greedy-only"), "{e}");
        let e = check(r#"{"prompt": [1], "n": 2}"#).unwrap_err();
        assert!(e.contains("greedy-only"), "{e}");
    }

    fn run(pats: &[&str], parts: &[&str]) -> (String, bool) {
        let mut s = Stops::new(pats.iter().map(|p| p.to_string()).collect());
        let mut out = String::new();
        for p in parts {
            out.push_str(&s.feed(p));
        }
        out.push_str(&s.finish());
        (out, s.hit)
    }

    #[test]
    fn cuts_before_a_stop_split_across_deltas() {
        assert_eq!(
            run(&["\nUser:"], &["Hi", " there\nU", "ser: next"]),
            ("Hi there".to_string(), true)
        );
    }

    #[test]
    fn releases_held_text_that_never_becomes_a_stop() {
        assert_eq!(
            run(&["</a>"], &["x</", "b> y"]),
            ("x</b> y".to_string(), false)
        );
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(run(&["END", "B"], &["aBcEND"]), ("a".to_string(), true));
    }

    #[test]
    fn no_stops_passes_everything() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".to_string(), false));
    }
}
