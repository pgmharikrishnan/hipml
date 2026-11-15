// SPDX-License-Identifier: Apache-2.0
//! The reply split: `reasoning_content`, `content` and `tool_calls`.
//!
//! llama-server's default reasoning format reads the template's own markers as
//! structure, and the benchmark compares against llama, so the same three
//! rules apply here:
//!
//! * a prompt that ends inside a `<think>` block means the reply starts in
//!   reasoning: everything before `</think>` is `reasoning_content`, everything
//!   after it is `content`;
//! * a block that never closes is all `reasoning_content` with an empty
//!   `content` — a truncated reply is not an answer;
//! * a `<tool_call>` block in the answer is a tool call, not text, and the
//!   answer around it stays in `content`.
//!
//! One state machine serves both reply paths, so a streamed reply and a whole
//! one split the same way. It holds back only what it cannot decide yet: a
//! partial marker, or a tool call whose close tag has not arrived.

use hipml_core::tool_json::{self, ToolCall};

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// One piece of a reply, in the order the model produced it.
pub enum Delta {
    Reasoning(String),
    Content(String),
    Call(ToolCall),
}

/// The splitter. `feed` may emit any number of deltas for one chunk.
pub struct Shaper {
    /// The close marker has been seen (or there was none to wait for).
    answering: bool,
    /// Text not emitted yet: a partial marker, or a partial tool call.
    buf: String,
    /// The answer's leading whitespace has been dropped.
    started: bool,
    /// Nothing has been emitted, so a re-emitted open tag is still possible.
    fresh: bool,
}

impl Shaper {
    pub fn new(think: bool) -> Shaper {
        Shaper {
            answering: !think,
            buf: String::new(),
            started: false,
            fresh: think,
        }
    }

    pub fn feed(&mut self, s: &str) -> Vec<Delta> {
        self.buf.push_str(s);
        let mut out = Vec::new();
        loop {
            if !self.answering {
                match self.buf.find(THINK_CLOSE) {
                    Some(p) => {
                        let head = self.buf[..p].to_string();
                        if let Some(text) = self.reasoning(&head) {
                            out.push(Delta::Reasoning(text));
                        }
                        self.buf.drain(..p + THINK_CLOSE.len());
                        self.answering = true;
                        continue;
                    }
                    None => {
                        let cut = self.buf.len() - self.hold(THINK_CLOSE);
                        if cut > 0 {
                            let head = self.buf[..cut].to_string();
                            if let Some(text) = self.reasoning(&head) {
                                out.push(Delta::Reasoning(text));
                            }
                            self.buf.drain(..cut);
                        }
                        break;
                    }
                }
            }
            match self.buf.find(CALL_OPEN) {
                Some(p) => {
                    // The text before the block is the answer; the block itself
                    // is held until its close tag arrives.
                    let text = self.buf[..p].trim_end().to_string();
                    if let Some(text) = self.content(&text) {
                        out.push(Delta::Content(text));
                    }
                    self.buf.drain(..p);
                    let Some(c) = self.buf.find(CALL_CLOSE) else {
                        break;
                    };
                    let end = c + CALL_CLOSE.len();
                    let block = self.buf[..end].to_string();
                    self.buf.drain(..end);
                    // A block the template does not define is text the model
                    // wrote: `parse_calls` hands it back untouched.
                    let (rest, calls) = tool_json::parse_calls(&block);
                    if !calls.is_empty() {
                        out.extend(calls.into_iter().map(Delta::Call));
                    }
                    if let Some(text) = self.content(&rest) {
                        out.push(Delta::Content(text));
                    }
                    continue;
                }
                None => {
                    let cut = self.buf.len() - self.hold(CALL_OPEN);
                    if cut > 0 {
                        let text = self.buf[..cut].to_string();
                        if let Some(text) = self.content(&text) {
                            out.push(Delta::Content(text));
                        }
                        self.buf.drain(..cut);
                    }
                    break;
                }
            }
        }
        out
    }

    /// Whatever is still held back: a partial marker is text the model wrote.
    pub fn finish(&mut self) -> Vec<Delta> {
        let mut out = Vec::new();
        if self.buf.is_empty() {
            return out;
        }
        let rest = std::mem::take(&mut self.buf);
        if self.answering {
            if let Some(text) = self.content(&rest) {
                out.push(Delta::Content(text));
            }
        } else if let Some(text) = self.reasoning(&rest) {
            out.push(Delta::Reasoning(text));
        }
        out
    }

    /// A reasoning piece, with a re-emitted open tag and the newline that
    /// follows it dropped.
    fn reasoning(&mut self, text: &str) -> Option<String> {
        let mut text = text;
        if self.fresh {
            self.fresh = false;
            if let Some(rest) = text.strip_prefix(THINK_OPEN) {
                text = rest.strip_prefix('\n').unwrap_or(rest);
            }
        }
        if text.is_empty() {
            return None;
        }
        Some(text.to_string())
    }

    /// An answer piece, with the answer's leading whitespace dropped.
    fn content(&mut self, text: &str) -> Option<String> {
        let text = if self.started {
            text
        } else {
            text.trim_start()
        };
        if text.is_empty() {
            return None;
        }
        self.started = true;
        Some(text.to_string())
    }

    /// Longest suffix that could still become `marker` (or, while the reply is
    /// still fresh, the open think tag the model might re-emit).
    fn hold(&self, marker: &str) -> usize {
        let keep = held(&self.buf, marker);
        if self.fresh {
            keep.max(held(&self.buf, THINK_OPEN))
        } else {
            keep
        }
    }
}

/// Longest suffix of `text` that is a proper prefix of `marker`.
fn held(text: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(text.len());
    (1..=max)
        .rev()
        .find(|n| text.ends_with(&marker[..*n]))
        .unwrap_or(0)
}

/// A whole reply, split the way the streaming path splits it.
pub struct Reply {
    pub reasoning: Option<String>,
    pub content: String,
    pub calls: Vec<ToolCall>,
}

pub fn shape(text: &str, think: bool) -> Reply {
    let mut shaper = Shaper::new(think);
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls = Vec::new();
    for delta in shaper.feed(text).into_iter().chain(shaper.finish()) {
        match delta {
            Delta::Reasoning(t) => reasoning.push_str(&t),
            Delta::Content(t) => content.push_str(&t),
            Delta::Call(c) => calls.push(c),
        }
    }
    Reply {
        reasoning: if think { Some(reasoning) } else { None },
        content,
        calls,
    }
}
