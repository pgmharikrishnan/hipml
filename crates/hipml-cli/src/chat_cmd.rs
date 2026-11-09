//! `hipml chat`: render the model's GGUF chat template and tokenize it with
//! special tokens as single ids.
//!
//! Conversation file format (one message per `--- role` block):
//!
//! ```text
//! --- system
//! You are a terse assistant.
//! --- user
//! Write a Rust function that adds two numbers.
//! --- assistant
//! fn add(a: i32, b: i32) -> i32 { a + b }
//! ```
//!
//! `--tools FILE` is one compact JSON tool definition per line; the template's
//! `tool | tojson` is emitted verbatim so the bytes match llama.cpp's renderer.
//! Assistant reasoning can be supplied as `--- assistant reasoning` (the block
//! after the role line is the reasoning; there is no separate content form) —
//! see `parse_messages`.

use hipml_core::chat::{render, ChatOptions, Message};
use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;

struct Args {
    gguf: String,
    messages: String,
    /// Full request body JSON (messages + tools + template kwargs), the same
    /// shape llama-server's `/apply-template` takes. Overrides `--messages`.
    json_file: Option<String>,
    tools: Option<String>,
    enable_thinking: bool,
    reasoning_effort: String,
    preserve_thinking: bool,
    generation_prompt: bool,
    json: bool,
    out: Option<String>,
    ids_out: Option<String>,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        gguf: String::new(),
        messages: String::new(),
        json_file: None,
        tools: None,
        enable_thinking: true,
        reasoning_effort: "xhigh".to_string(),
        preserve_thinking: true,
        generation_prompt: true,
        json: false,
        out: None,
        ids_out: None,
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
            "--messages" => {
                a.messages = need(i)?.to_string();
                i += 2;
            }
            "--json-file" => {
                a.json_file = Some(need(i)?.to_string());
                i += 2;
            }
            "--tools" => {
                a.tools = Some(need(i)?.to_string());
                i += 2;
            }
            "--enable-thinking" => {
                a.enable_thinking = match need(i)? {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    other => return Err(format!("bad --enable-thinking {other}")),
                };
                i += 2;
            }
            "--no-thinking" => {
                a.enable_thinking = false;
                i += 1;
            }
            "--reasoning-effort" => {
                a.reasoning_effort = need(i)?.to_string();
                i += 2;
            }
            "--no-preserve-thinking" => {
                a.preserve_thinking = false;
                i += 1;
            }
            "--no-generation-prompt" => {
                a.generation_prompt = false;
                i += 1;
            }
            "--json" => {
                a.json = true;
                i += 1;
            }
            "--out" => {
                a.out = Some(need(i)?.to_string());
                i += 2;
            }
            "--ids-out" => {
                a.ids_out = Some(need(i)?.to_string());
                i += 2;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.gguf.is_empty() || (a.messages.is_empty() && a.json_file.is_none()) {
        return Err("--gguf and (--messages or --json-file) are required".to_string());
    }
    Ok(a)
}

/// Parse the `--- role` block format. A role line is `--- system`, `--- user`,
/// `--- assistant` or `--- tool`. `--- assistant reasoning` attaches reasoning
/// to the assistant message that follows? To keep the format unambiguous the
/// reasoning is written as a `::reasoning` block inside an assistant message:
///
/// ```text
/// --- assistant
/// ::reasoning
/// the private thought
/// ::content
/// the visible answer
/// ```
fn parse_messages(text: &str) -> Result<Vec<Message>, String> {
    let mut messages: Vec<Message> = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;
    let finish = |cur: &mut Option<(String, Vec<String>)>, out: &mut Vec<Message>| {
        if let Some((role, lines)) = cur.take() {
            out.push(build_message(&role, &lines));
        }
    };
    for line in text.lines() {
        if let Some(role) = line.strip_prefix("--- ") {
            finish(&mut current, &mut messages);
            let role = role.trim().to_string();
            if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
                return Err(format!("unknown role {role:?}"));
            }
            current = Some((role, Vec::new()));
        } else {
            match current.as_mut() {
                Some((_, lines)) => lines.push(line.to_string()),
                None if line.trim().is_empty() => {}
                None => return Err("content before the first --- role line".to_string()),
            }
        }
    }
    finish(&mut current, &mut messages);
    if messages.is_empty() {
        return Err("no messages".to_string());
    }
    Ok(messages)
}

fn build_message(role: &str, lines: &[String]) -> Message {
    // Drop one trailing blank line left by the block format.
    let mut lines = lines.to_vec();
    if lines.last().map(|s| s.is_empty()).unwrap_or(false) {
        lines.pop();
    }
    let joined = lines.join("\n");
    if role == "assistant" {
        if let Some(rest) = joined.strip_prefix("::reasoning\n") {
            if let Some((reasoning, content)) = rest.split_once("\n::content\n") {
                return Message::assistant_with_reasoning(content, reasoning);
            }
        }
        if let Some(content) = joined.strip_prefix("::content\n") {
            return Message::assistant_with_reasoning(content, "");
        }
    }
    Message::new(role, joined)
}

/// Escape a string for the small JSON writer (no serde in this crate).
pub(crate) fn json_escape(s: &str) -> String {
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

pub fn run(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("chat: {e}");
            return 2;
        }
    };
    let g = match Gguf::open(&a.gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("chat: cannot open {}: {e}", a.gguf);
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("chat: tokenizer: {e}");
            return 1;
        }
    };
    let Some(template) = tok.chat_template.as_deref() else {
        eprintln!("chat: {} has no tokenizer.chat_template", a.gguf);
        return 1;
    };
    let (messages, opts) = if let Some(path) = &a.json_file {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("chat: read {path}: {e}");
                return 1;
            }
        };
        match hipml_core::chat_request::load(&text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("chat: {path}: {e}");
                return 1;
            }
        }
    } else {
        let text = match std::fs::read_to_string(&a.messages) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("chat: read {}: {e}", a.messages);
                return 1;
            }
        };
        let messages = match parse_messages(&text) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("chat: {}: {e}", a.messages);
                return 1;
            }
        };
        let tools = match &a.tools {
            None => Vec::new(),
            Some(path) => match std::fs::read_to_string(path) {
                Ok(s) => s
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(str::to_string)
                    .collect(),
                Err(e) => {
                    eprintln!("chat: read {path}: {e}");
                    return 1;
                }
            },
        };
        let opts = ChatOptions {
            enable_thinking: a.enable_thinking,
            reasoning_effort: a.reasoning_effort.clone(),
            preserve_thinking: a.preserve_thinking,
            add_generation_prompt: a.generation_prompt,
            tools,
        };
        (messages, opts)
    };
    let rendered = match render(template, &messages, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("chat: {e}");
            return 1;
        }
    };
    let ids = tok.encode_special(&rendered);

    if let Some(path) = &a.ids_out {
        let json = format!("{{\"ids\":{ids:?}}}");
        if let Err(e) = std::fs::write(path, json) {
            eprintln!("chat: write {path}: {e}");
            return 1;
        }
    }
    if let Some(path) = &a.out {
        if let Err(e) = std::fs::write(path, &rendered) {
            eprintln!("chat: write {path}: {e}");
            return 1;
        }
    }
    if a.json {
        println!(
            "{{\"text\":\"{}\",\"ids\":{:?},\"eos_id\":{},\"bos_id\":{},\
             \"template_bytes\":{},\"enable_thinking\":{}}}",
            json_escape(&rendered),
            ids,
            tok.eos_id,
            tok.bos_id,
            template.len(),
            a.enable_thinking
        );
    } else {
        println!("{rendered}");
        if !a.json {
            println!("\nids: {ids:?}");
        }
    }
    0
}
