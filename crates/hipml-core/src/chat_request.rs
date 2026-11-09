//! Build `Message`s and `ChatOptions` from a JSON request body.
//!
//! The body is the same shape llama-server's `/apply-template` accepts
//! (`messages`, `tools`, `chat_template_kwargs`, `add_generation_prompt`), so
//! the template verifier can feed identical bytes to both engines. Numbers are
//! parsed by `tool_json` to keep their source lexeme for `tojson`.

use crate::chat::{ChatOptions, ContentPart, Message};
use crate::json::Value;
use crate::tool_json::{self, Arg, ToolCall};

/// Convert the server's parsed JSON into the lexeme-preserving argument type.
/// Integers survive as integers because that is how llama's `tojson` prints
/// them; a value that was already an `f64` in the server body is best effort.
pub fn value_to_arg(v: &Value) -> Arg {
    match v {
        Value::Null => Arg::Null,
        Value::Bool(b) => Arg::Bool(*b),
        Value::Num(n) => {
            if n.fract() == 0.0 && *n >= i64::MIN as f64 && *n <= i64::MAX as f64 {
                Arg::Int(*n as i64)
            } else {
                Arg::Float(*n)
            }
        }
        Value::Str(s) => Arg::Str(s.clone()),
        Value::Arr(a) => Arg::Arr(a.iter().map(value_to_arg).collect()),
        Value::Obj(fields) => Arg::Obj(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_arg(v)))
                .collect(),
        ),
    }
}

fn content(v: Option<&Arg>) -> Result<(String, Vec<ContentPart>), String> {
    let Some(v) = v else {
        return Ok((String::new(), Vec::new()));
    };
    match v {
        Arg::Null => Ok((String::new(), Vec::new())),
        Arg::Str(s) => Ok((s.clone(), Vec::new())),
        Arg::Arr(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let kind = item.get("type").and_then(Arg::as_str);
                match kind {
                    Some("text") => {
                        let t = item
                            .get("text")
                            .and_then(Arg::as_str)
                            .ok_or_else(|| format!("content part {i}: text is missing"))?;
                        parts.push(ContentPart::Text(t.to_string()));
                    }
                    Some("image") | Some("image_url") => parts.push(ContentPart::Image),
                    Some("video") | Some("video_url") => parts.push(ContentPart::Video),
                    Some(other) => {
                        return Err(format!("content part {i}: unsupported type {other:?}"))
                    }
                    None => {
                        if item.get("image").is_some() || item.get("image_url").is_some() {
                            parts.push(ContentPart::Image);
                        } else if item.get("video").is_some() || item.get("video_url").is_some() {
                            parts.push(ContentPart::Video);
                        } else {
                            return Err(format!("content part {i} has no type"));
                        }
                    }
                }
            }
            Ok((String::new(), parts))
        }
        _ => Err("content must be a string, an array of parts, or null".into()),
    }
}

fn tool_call(v: &Arg, i: usize) -> Result<ToolCall, String> {
    // The template accepts either `{function: {...}}` or the function fields
    // at the top level.
    let f = v.get("function").unwrap_or(v);
    let name = f
        .get("name")
        .and_then(Arg::as_str)
        .ok_or_else(|| format!("tool_call {i}: name is missing"))?
        .to_string();
    let arguments = match f.get("arguments") {
        None | Some(Arg::Null) => Vec::new(),
        Some(Arg::Str(s)) => {
            tool_json::parse_arguments(s).map_err(|e| format!("tool_call {i}: arguments: {e}"))?
        }
        Some(Arg::Obj(fields)) => fields.clone(),
        Some(other) => {
            return Err(format!(
                "tool_call {i}: arguments must be a JSON string or object, got {other:?}"
            ))
        }
    };
    Ok(ToolCall { name, arguments })
}

pub fn message_from_arg(v: &Arg, i: usize) -> Result<Message, String> {
    let role = v
        .get("role")
        .and_then(Arg::as_str)
        .ok_or_else(|| format!("message {i}: role is required"))?;
    let role = if role == "developer" { "system" } else { role };
    let (text, parts) = content(v.get("content"))?;
    let reasoning = v.get("reasoning_content").and_then(Arg::as_str);
    let calls = match v.get("tool_calls") {
        None | Some(Arg::Null) => Vec::new(),
        Some(Arg::Arr(items)) => items
            .iter()
            .enumerate()
            .map(|(k, item)| tool_call(item, k))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(format!("message {i}: tool_calls must be an array")),
    };
    let mut m = match reasoning {
        Some(r) if !r.is_empty() => Message::assistant_with_reasoning(text, r),
        _ => Message::new(role, text),
    };
    m.tool_calls = calls;
    m.parts = parts;
    Ok(m)
}

/// Parse a request body into the renderer's inputs.
pub fn load(text: &str) -> Result<(Vec<Message>, ChatOptions), String> {
    let root = tool_json::parse(text).map_err(|e| format!("json: {e}"))?;
    let arr = root
        .get("messages")
        .and_then(Arg::as_arr)
        .ok_or("messages must be an array")?;
    if arr.is_empty() {
        return Err("messages is empty".into());
    }
    let messages = arr
        .iter()
        .enumerate()
        .map(|(i, m)| message_from_arg(m, i))
        .collect::<Result<Vec<_>, _>>()?;

    let mut opts = ChatOptions::default();
    if let Some(k) = root.get("chat_template_kwargs") {
        if let Some(b) = k.get("enable_thinking").and_then(Arg::as_bool) {
            opts.enable_thinking = b;
        }
        if let Some(s) = k.get("reasoning_effort").and_then(Arg::as_str) {
            opts.reasoning_effort = s.to_string();
        }
        if let Some(b) = k.get("preserve_thinking").and_then(Arg::as_bool) {
            opts.preserve_thinking = b;
        }
    }
    if let Some(b) = root.get("add_generation_prompt").and_then(Arg::as_bool) {
        opts.add_generation_prompt = b;
    }
    if let Some(t) = root.get("tools").and_then(Arg::as_arr) {
        opts.tools = t.iter().map(tool_json::tojson).collect();
    }
    Ok((messages, opts))
}
