//! Tool-call arguments: parsing and the exact byte format the Qwen3.5
//! template's `arguments | items` loop emits.
//!
//! The template walks an argument object and, for each value, either inserts a
//! string verbatim or runs `tojson` on anything else. llama.cpp's `tojson` is
//! nlohmann's dump with the default `", "` / `": "` separators and raw UTF-8,
//! so a nested object is `{"k": "v"}`, not `{"k":"v"}`.
//!
//! Numbers are kept as their source lexeme class: an integer stays an integer
//! (`1`), a float keeps a decimal point (`1.0`, `1e3` -> `1000.0`) and an
//! exponent is printed C++-style (`1e-07`, not Rust's `1e-7`). That is what
//! the reference emits; a client's `arguments` string is parsed by llama the
//! same way before it reaches the template.

use std::fmt::Write as _;

/// One already-parsed JSON value from a tool call's `arguments`.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Arg>),
    Obj(Vec<(String, Arg)>),
}

impl Arg {
    pub fn get(&self, key: &str) -> Option<&Arg> {
        match self {
            Arg::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Arg::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Arg::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Arg]> {
        match self {
            Arg::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Arg)]> {
        match self {
            Arg::Obj(o) => Some(o),
            _ => None,
        }
    }
}

/// One assistant tool call. `name` is `function.name`; `arguments` is the
/// object's entries in source order (the template preserves insertion order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<(String, Arg)>,
}

/// Parse the `arguments` field: an OpenAI-style JSON string, empty, or absent.
/// A non-object is rejected: the template's `| items` loop cannot render it.
pub fn parse_arguments(s: &str) -> Result<Vec<(String, Arg)>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    let v = parse(s)?;
    match v {
        Arg::Obj(fields) => Ok(fields),
        other => Err(format!(
            "tool arguments must be a JSON object, got {other:?}"
        )),
    }
}

/// Parse one JSON value strictly (no trailing bytes, no comments).
pub fn parse(s: &str) -> Result<Arg, String> {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing bytes at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> Result<(), String> {
        if self.b.get(self.i) == Some(&byte) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at {}", byte as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Arg, String> {
        self.ws();
        match self.b.get(self.i).copied() {
            Some(b'n') => self.lit("null", Arg::Null),
            Some(b't') => self.lit("true", Arg::Bool(true)),
            Some(b'f') => self.lit("false", Arg::Bool(false)),
            Some(b'"') => Ok(Arg::Str(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            other => Err(format!("unexpected {other:?} at {}", self.i)),
        }
    }

    fn lit(&mut self, text: &str, v: Arg) -> Result<Arg, String> {
        if self.b[self.i..].starts_with(text.as_bytes()) {
            self.i += text.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i) != Some(&b'\\')
                                    || self.b.get(self.i + 1) != Some(&b'u')
                                {
                                    return Err("lone high surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("bad low surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or("invalid codepoint")?);
                        }
                        _ => return Err(format!("bad escape \\{} at {}", e as char, self.i)),
                    }
                }
                _ => {
                    // Copy one UTF-8 sequence verbatim.
                    let start = self.i - 1;
                    let len = utf8_len(c);
                    self.i = start + len;
                    if self.i > self.b.len() {
                        return Err("truncated UTF-8".into());
                    }
                    out.push_str(
                        std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?,
                    );
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let s = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let s = std::str::from_utf8(s).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|e| format!("bad \\u {s:?}: {e}"))?;
        self.i += 4;
        Ok(v)
    }

    fn number(&mut self) -> Result<Arg, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        let mut float = false;
        if self.b.get(self.i) == Some(&b'.') {
            float = true;
            self.i += 1;
            while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.b.get(self.i), Some(b'e') | Some(b'E')) {
            float = true;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if text.is_empty() || text == "-" {
            return Err(format!("bad number at {start}"));
        }
        if !float {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Arg::Int(v));
            }
        }
        text.parse::<f64>()
            .map(Arg::Float)
            .map_err(|e| format!("bad number {text:?}: {e}"))
    }

    fn array(&mut self) -> Result<Arg, String> {
        self.eat(b'[')?;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Arg::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Arg::Arr(out));
                }
                other => return Err(format!("expected , or ] at {}: {other:?}", self.i)),
            }
        }
    }

    fn object(&mut self) -> Result<Arg, String> {
        self.eat(b'{')?;
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Arg::Obj(out));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.eat(b':')?;
            let v = self.value()?;
            out.push((key, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Arg::Obj(out));
                }
                other => return Err(format!("expected , or }} at {}: {other:?}", self.i)),
            }
        }
    }
}

fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// `tojson` for a non-string argument value, as llama.cpp emits it.
pub fn tojson(v: &Arg) -> String {
    let mut out = String::new();
    write_arg(&mut out, v);
    out
}

fn write_arg(out: &mut String, v: &Arg) {
    match v {
        Arg::Null => out.push_str("null"),
        Arg::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Arg::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Arg::Float(f) => out.push_str(&fmt_float(*f)),
        Arg::Str(s) => write_string(out, s),
        Arg::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_arg(out, item);
            }
            out.push(']');
        }
        Arg::Obj(fields) => {
            out.push('{');
            for (i, (k, item)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, k);
                out.push_str(": ");
                write_arg(out, item);
            }
            out.push('}');
        }
    }
}

/// nlohmann's dump format: shortest round-trip, always a decimal point for a
/// float, exponent with a sign and at least two digits.
fn fmt_float(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let s = format!("{v:?}");
    if let Some(pos) = s.find(['e', 'E']) {
        let (mant, exp) = s.split_at(pos);
        let exp = &exp[1..];
        let (sign, digits) = match exp.strip_prefix('-') {
            Some(d) => ('-', d),
            None => ('+', exp.strip_prefix('+').unwrap_or(exp)),
        };
        let digits = if digits.len() < 2 {
            format!("0{digits}")
        } else {
            digits.to_string()
        };
        format!("{mant}e{sign}{digits}")
    } else if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append the `<tool_call>` block(s) the template emits after an assistant
/// message's content. `content` is the already-trimmed content string.
pub fn render_calls(out: &mut String, content: &str, calls: &[ToolCall]) {
    for (i, call) in calls.iter().enumerate() {
        if i == 0 {
            if !content.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str("<tool_call>\n<function=");
            out.push_str(&call.name);
            out.push_str(">\n");
        } else {
            out.push_str("\n<tool_call>\n<function=");
            out.push_str(&call.name);
            out.push_str(">\n");
        }
        for (name, value) in &call.arguments {
            out.push_str("<parameter=");
            out.push_str(name);
            out.push_str(">\n");
            match value {
                Arg::Str(s) => out.push_str(s),
                other => out.push_str(&tojson(other)),
            }
            out.push_str("\n</parameter>\n");
        }
        out.push_str("</function>\n</tool_call>");
    }
}
