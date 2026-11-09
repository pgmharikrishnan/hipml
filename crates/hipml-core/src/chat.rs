//! The model's chat template, rendered from `tokenizer.chat_template`.
//!
//! The GGUF stores a Jinja template; hipml has no Jinja engine (and no external
//! crates), so this module implements the *text-only* branches of the exact
//! template Qwen3.8-27B ships. It reads the template string from the file and
//! validates its signature before rendering. An unfamiliar template is rejected
//! (`Unsupported`), never silently replaced by a guessed ChatML format, so a
//! different model cannot be rendered with the wrong format.
//!
//! The template's text path is short and fully specified:
//!
//! ```text
//! system:      <|im_start|>system\n[reasoning\n\n]content<|im_end|>\n
//! user:        <|im_start|>user\ncontent<|im_end|>\n
//! assistant:   <|im_start|>assistant\n<think>\n[reasoning]\n</think>\n\ncontent<|im_end|>\n
//! tool(s):     <|im_start|>user\n<tool_response>\ncontent\n</tool_response><|im_end|>\n
//! generation:  <|im_start|>assistant\n<think>\n            (thinking on)
//!              <|im_start|>assistant\n<think>\n\n</think>\n\n (thinking off)
//! ```
//!
//! With thinking on (the default) the template injects a system turn carrying
//! the default `xhigh` reasoning instruction even when the caller supplies no
//! system message; that is part of the template, not a default added here.

/// One chat message. `content` is the text the template's `render_content`
/// produces for a string (the only form the text path accepts).
#[derive(Clone, Debug, Default)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Optional assistant reasoning that the template emits inside `<think>`.
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Message {
        Message {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
        }
    }

    pub fn assistant_with_reasoning(content: impl Into<String>, reasoning: impl Into<String>) -> Message {
        Message {
            role: "assistant".into(),
            content: content.into(),
            reasoning_content: Some(reasoning.into()),
        }
    }
}

/// Template knobs exposed by llama.cpp's `chat_template_kwargs`.
#[derive(Clone, Debug)]
pub struct ChatOptions {
    /// `enable_thinking` (undefined in the template means true).
    pub enable_thinking: bool,
    /// `reasoning_effort`, default `xhigh`.
    pub reasoning_effort: String,
    /// `preserve_thinking`, undefined means true.
    pub preserve_thinking: bool,
    pub add_generation_prompt: bool,
    /// Tool definitions as compact JSON objects, in the order to emit them.
    /// The template's `tool | tojson` is passed through verbatim.
    pub tools: Vec<String>,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions {
            enable_thinking: true,
            reasoning_effort: "xhigh".to_string(),
            preserve_thinking: true,
            add_generation_prompt: true,
            tools: Vec::new(),
        }
    }
}

/// Why rendering failed.
#[derive(Debug)]
pub enum ChatError {
    Unsupported(String),
    BadConversation(String),
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatError::Unsupported(m) => write!(f, "unsupported chat template: {m}"),
            ChatError::BadConversation(m) => write!(f, "bad conversation: {m}"),
        }
    }
}

impl std::error::Error for ChatError {}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, \
validate key assumptions, consider plausible alternatives, and prioritize correctness, \
consistency, and clarity in the final answer.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
directly to the conclusion without unnecessary elaboration.";

const TOOL_RULES: &str = "\n\nIf you choose to call a function ONLY reply in the following format \
with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\
\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must \
be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You \
may provide optional reasoning for your function call in natural language BEFORE the function \
call, but NOT after\n- If there is no function call available, answer the question like normal \
with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Render `messages` with the template stored in the GGUF.
///
/// `template` must be the file's `tokenizer.chat_template` string. The text-only
/// Qwen3.5/Qwen3.8-27B template is recognised by the markers below; anything else is
/// an error rather than a best-effort guess.
pub fn render(
    template: &str,
    messages: &[Message],
    opts: &ChatOptions,
) -> Result<String, ChatError> {
    validate_template(template)?;
    if messages.is_empty() {
        return Err(ChatError::BadConversation("no messages provided".into()));
    }

    let reasoning = reasoning_instructions(opts)?;
    let mut out = String::new();

    if !opts.tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if !reasoning.is_empty() {
            out.push_str(&reasoning);
            out.push_str("\n\n");
        }
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &opts.tools {
            out.push('\n');
            out.push_str(&tojson(tool));
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_RULES);
        if messages[0].role == "system" {
            let content = messages[0].content.trim();
            if !content.is_empty() {
                out.push_str("\n\n");
                out.push_str(content);
            }
        }
        out.push_str("<|im_end|>\n");
    } else if messages[0].role == "system" {
        let content = messages[0].content.trim();
        if !content.is_empty() {
            out.push_str("<|im_start|>system\n");
            if !reasoning.is_empty() {
                out.push_str(&reasoning);
                out.push_str("\n\n");
            }
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        } else if !reasoning.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(&reasoning);
            out.push_str("<|im_end|>\n");
        }
    } else if !reasoning.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&reasoning);
        out.push_str("<|im_end|>\n");
    }

    // `multi_step_tool`: walking back, the last user message that is not a
    // `<tool_response>` wrapper. Assistant turns at or before it may drop their
    // thinking block when `preserve_thinking` is false.
    let last_query = last_query_index(messages)
        .ok_or_else(|| ChatError::BadConversation("no user query found".into()))?;

    for (i, message) in messages.iter().enumerate() {
        let content = message.content.trim();
        match message.role.as_str() {
            "system" => {
                if i != 0 {
                    return Err(ChatError::BadConversation(
                        "system message must be at the beginning".into(),
                    ));
                }
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                let reasoning_content = message
                    .reasoning_content
                    .as_deref()
                    .unwrap_or("")
                    .trim();
                if opts.preserve_thinking || i > last_query {
                    out.push_str("<think>\n");
                    out.push_str(reasoning_content);
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Consecutive tool messages share one user turn.
                if i == 0 || messages[i - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                let next_is_tool = messages.get(i + 1).map(|m| m.role == "tool").unwrap_or(false);
                if !next_is_tool {
                    out.push_str("<|im_end|>\n");
                }
            }
            other => {
                return Err(ChatError::BadConversation(format!(
                    "unexpected message role {other:?}"
                )))
            }
        }
    }

    if opts.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if opts.enable_thinking {
            out.push_str("<think>\n");
        } else {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(out)
}

/// `tool | tojson` as llama.cpp's renderer emits it: single-line JSON with
/// `", "` between members and `": "` after a key, key order preserved. The GGUF
/// template only says `tojson`; the byte format comes from the runtime, so this
/// is the normalisation that makes the rendered prompt byte-identical.
/// Strings are copied verbatim (escapes untouched); whitespace outside strings
/// is collapsed.
pub fn tojson(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 8);
    let mut in_str = false;
    let mut escaped = false;
    for c in raw.chars() {
        if in_str {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push(c);
            }
            ':' => out.push_str(": "),
            ',' => out.push_str(", "),
            c if c.is_whitespace() => {}
            c => out.push(c),
        }
    }
    out
}

/// Reject a template this renderer was not written against. The markers are
/// the branch points used above; they are all present in the Qwen3.8-27B template and
/// absent from a plain ChatML template.
fn validate_template(template: &str) -> Result<(), ChatError> {
    let required = [
        "render_content",
        "reasoning_instructions",
        "preserve_thinking",
        "add_generation_prompt",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
    ];
    for marker in required {
        if !template.contains(marker) {
            return Err(ChatError::Unsupported(format!(
                "template has no {marker:?} marker"
            )));
        }
    }
    Ok(())
}

fn reasoning_instructions(opts: &ChatOptions) -> Result<String, ChatError> {
    if !opts.enable_thinking {
        return Ok(String::new());
    }
    match opts.reasoning_effort.as_str() {
        "xhigh" => Ok(XHIGH.to_string()),
        "medium" => Ok(String::new()),
        "low" => Ok(LOW.to_string()),
        other => Err(ChatError::BadConversation(format!(
            "unexpected reasoning effort {other:?} (xhigh, medium, low)"
        ))),
    }
}

fn last_query_index(messages: &[Message]) -> Option<usize> {
    for i in (0..messages.len()).rev() {
        if messages[i].role == "user" {
            let content = messages[i].content.trim();
            let is_tool_response =
                content.starts_with("<tool_response>") && content.ends_with("</tool_response>");
            if !is_tool_response {
                return Some(i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = "{%- macro render_content(content, do_vision_count) %}\
        {%- if content is string %}{{- content }}{%- endif %}{%- endmacro %}\
        {%- set reasoning_instructions = '' %}\
        {%- if preserve_thinking is undefined or preserve_thinking is true %}{%- endif %}\
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant' }}{%- endif %}\
        {{- '<|im_end|>' }}{{- '<think>' }}";

    #[test]
    fn single_turn_thinking() {
        let msgs = [Message::new("user", "hi")];
        let out = render(TEMPLATE, &msgs, &ChatOptions::default()).unwrap();
        assert!(out.starts_with("<|im_start|>system\nReasoning effort is set to xhigh."));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>\n"));
    }

    #[test]
    fn thinking_off_has_empty_block() {
        let opts = ChatOptions {
            enable_thinking: false,
            ..ChatOptions::default()
        };
        let out = render(TEMPLATE, &[Message::new("user", "hi")], &opts).unwrap();
        assert!(!out.contains("Reasoning effort"));
        assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn rejects_unknown_template() {
        let err = render("plain chatml", &[Message::new("user", "hi")], &ChatOptions::default());
        assert!(matches!(err, Err(ChatError::Unsupported(_))));
    }

    #[test]
    fn tojson_matches_llama_spacing() {
        assert_eq!(
            tojson(r#"{"type":"function","n":[1,2]}"#),
            r#"{"type": "function", "n": [1, 2]}"#
        );
        assert_eq!(
            tojson(r#"{"s":"a, b: c","e":"x\"y"}"#),
            r#"{"s": "a, b: c", "e": "x\"y"}"#
        );
    }
}
