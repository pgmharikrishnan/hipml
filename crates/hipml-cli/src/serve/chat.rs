//! Chat rendering and tokenization for the HTTP surface.
//!
//! The server reuses the WP43 path exactly: the GGUF's own template, special
//! tokens as single ids, and the same `ChatOptions` the CLI defaults to. A
//! request is reduced to prompt ids here (on the connection thread); the
//! scheduler only ever sees ids.

use hipml_core::chat::{render, ChatOptions, Message};
use hipml_core::tokenizer::Tokenizer;

use super::json::Value;

/// A prompt plus the token boundaries worth snapshotting in the prefix cache.
///
/// `cuts` are message boundaries that are also exact token prefixes of `ids`
/// (usually the end of the system turn). A tile may finish on a cut and the
/// scheduler captures that state, so the next request that shares the prefix
/// restores instead of recomputing it.
pub struct Prompt {
    pub ids: Vec<u32>,
    pub cuts: Vec<usize>,
}

pub fn messages(v: &Value) -> Result<Vec<Message>, String> {
    let arr = v
        .get("messages")
        .and_then(Value::as_arr)
        .ok_or("messages must be an array")?;
    if arr.is_empty() {
        return Err("messages is empty".to_string());
    }
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("message {i}: role is required"))?;
        let role = match role {
            "developer" => "system",
            r => r,
        };
        let content = m
            .get("content")
            .map(content_text)
            .transpose()?
            .unwrap_or_default();
        let reasoning = m
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if role == "assistant" && !reasoning.is_empty() {
            out.push(Message::assistant_with_reasoning(content, reasoning));
        } else {
            out.push(Message::new(role, content));
        }
    }
    Ok(out)
}

/// OpenAI content may be a string or an array of typed parts. Text parts are
/// concatenated; an image part is rejected rather than silently dropped.
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        Value::Null => Ok(String::new()),
        Value::Arr(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(Value::as_str) {
                            out.push_str(t);
                        }
                    }
                    Some("image_url") => {
                        return Err("image parts are not supported by this model".to_string())
                    }
                    Some(other) => return Err(format!("unsupported content part {other:?}")),
                    None => return Err("content part without a type".to_string()),
                }
            }
            Ok(out)
        }
        _ => Err("content must be a string or an array of parts".to_string()),
    }
}

fn opts(add_generation_prompt: bool) -> ChatOptions {
    ChatOptions {
        enable_thinking: true,
        reasoning_effort: "xhigh".to_string(),
        preserve_thinking: true,
        add_generation_prompt,
        tools: Vec::new(),
    }
}

/// Render and tokenize a conversation, keeping message boundaries that are
/// exact token prefixes of the whole prompt.
pub fn chat(tok: &Tokenizer, template: &str, messages: &[Message]) -> Result<Prompt, String> {
    let base = opts(true);
    let text = render(template, messages, &base).map_err(|e| format!("chat template: {e}"))?;
    let ids = tok.encode_special(&text);
    if ids.is_empty() {
        return Err("rendered conversation is empty".to_string());
    }

    let mut cuts = Vec::new();
    let plain = opts(false);
    for k in 1..messages.len() {
        let Ok(text) = render(template, &messages[..k], &plain) else {
            continue;
        };
        let prefix = tok.encode_special(&text);
        if !prefix.is_empty() && prefix.len() < ids.len() && ids.starts_with(&prefix) {
            if cuts.last() != Some(&prefix.len()) {
                cuts.push(prefix.len());
            }
        }
    }
    // This template refuses to render a conversation with no user turn, so the
    // system-only prefix above is absent exactly when the shared system prompt
    // is what we want to cache. Recover the end of each system turn from the
    // full render: it closes with a special token, which is a stable token
    // boundary, plus the newline that starts the next turn.
    for m in messages.iter().filter(|m| m.role == "system") {
        if let Some(cut) = end_of_turn(&text, &m.content) {
            let prefix = tok.encode_special(&text[..cut]);
            if !prefix.is_empty() && prefix.len() < ids.len() && ids.starts_with(&prefix) {
                cuts.push(prefix.len());
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    Ok(Prompt { ids, cuts })
}

/// Byte offset just past the special token that closes the turn carrying
/// `content` (and past one trailing newline).
fn end_of_turn(text: &str, content: &str) -> Option<usize> {
    if content.is_empty() {
        return None;
    }
    let start = text.find(content)? + content.len();
    let rest = &text[start..];
    let lt = rest.find('<')?;
    let gt = rest[lt..].find('>')?;
    let mut cut = start + lt + gt + 1;
    if text[cut..].starts_with('\n') {
        cut += 1;
    }
    Some(cut)
}

/// `/completion`: a raw prompt string or an array of token ids.
pub fn completion(tok: &Tokenizer, body: &Value) -> Result<Prompt, String> {
    let p = body
        .get("prompt")
        .ok_or("prompt is required")?;
    let ids = match p {
        Value::Str(s) => tok.encode_special(s),
        Value::Arr(items) => {
            if items.iter().any(|v| v.as_arr().is_some()) {
                return Err("batched prompts are not supported; send one array of ids".to_string());
            }
            // An id past the vocabulary would index past the embedding table
            // on the device; reject it here, before the scheduler sees it.
            let vocab = tok.tokens.len() as u64;
            let mut ids = Vec::with_capacity(items.len());
            for it in items {
                let n = it.as_u64().ok_or("prompt ids must be non-negative integers")?;
                if n >= vocab {
                    return Err(format!("prompt id {n} is outside the {vocab}-token vocabulary"));
                }
                ids.push(n as u32);
            }
            ids
        }
        _ => return Err("prompt must be a string or an array of token ids".to_string()),
    };
    if ids.is_empty() {
        return Err("prompt is empty".to_string());
    }
    // Test knob: `cache_cut` makes a raw-id prompt capture a prefix state at
    // that boundary, so a second identical request exercises the prefix-hit
    // tail path (the default cut is the whole prompt, which never hits).
    let n = ids.len();
    let cut = body
        .get("cache_cut")
        .and_then(Value::as_u64)
        .map(|c| c as usize)
        .filter(|c| *c > 0 && *c < n);
    Ok(Prompt {
        ids,
        cuts: vec![cut.unwrap_or(n)],
    })
}

#[cfg(test)]
mod tests {
    use super::end_of_turn;

    #[test]
    fn system_turn_ends_after_the_close_marker() {
        let sys = "be brief";
        let text = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n";
        let cut = end_of_turn(text, sys).unwrap();
        assert_eq!(&text[..cut], "<|im_start|>system\nbe brief<|im_end|>\n");
    }
}
