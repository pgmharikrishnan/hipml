//! GPT-2-style byte-level BPE, as stored in the GGUF tokenizer tables.
//!
//! Ports `llm_tokenizer_bpe_session::tokenize` and
//! `unicode_regex_split_custom_qwen35` (`src/llama-vocab.cpp`,
//! `src/unicode.cpp`): the text is mapped through the GPT-2 byte encoder, split
//! with the `tokenizer.ggml.pre = "qwen35"` pattern, then merged by rank.
//!
//! Scope: the pretokenizer is exact for ASCII. Non-ASCII is handled through the
//! same byte-encoded codepoints, but the general-category tables are Rust's
//! approximations (`char::is_alphabetic` for `\p{L}`, a combining-range test for
//! `\p{M}`) rather than llama.cpp's generated table. Prompts in the acceptance
//! set are ASCII; this is called out as unverified for other scripts.

use std::collections::HashMap;
use std::io::{self, Read, Seek};

use crate::gguf::Gguf;

pub struct Tokenizer {
    pub tokens: Vec<String>,
    id_of: HashMap<String, u32>,
    /// `(left, right) -> rank`, from `tokenizer.ggml.merges`.
    merges: HashMap<(String, String), u32>,
    /// GPT-2 byte encoder: byte -> printable codepoint.
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos_id: u32,
    pub bos_id: u32,
    pub add_bos: bool,
}

impl Tokenizer {
    pub fn from_gguf<R: Read + Seek>(g: &Gguf<R>) -> io::Result<Tokenizer> {
        let model = g
            .meta_req("tokenizer.ggml.model")?
            .as_str()
            .unwrap_or("")
            .to_string();
        if model != "gpt2" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("tokenizer model {model:?} is not gpt2 BPE"),
            ));
        }

        let toks = g
            .meta_req("tokenizer.ggml.tokens")?
            .as_strings()
            .ok_or_else(|| bad("tokenizer.ggml.tokens"))?;
        let tokens: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
        let mut id_of = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            id_of.insert(t.clone(), i as u32);
        }

        let merge_list = g
            .meta_req("tokenizer.ggml.merges")?
            .as_strings()
            .ok_or_else(|| bad("tokenizer.ggml.merges"))?;
        let mut merges = HashMap::with_capacity(merge_list.len());
        for (rank, m) in merge_list.iter().enumerate() {
            // Merges are "left right" over byte-encoded tokens; a space never
            // appears inside a piece because " " is encoded as U+0120.
            let (a, b) = m.split_once(' ').ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("merge {m:?} is not a pair"),
                )
            })?;
            merges.insert((a.to_string(), b.to_string()), rank as u32);
        }

        let (byte_to_char, char_to_byte) = byte_encoder();

        let eos_id = g
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let bos_id = g
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let add_bos = g
            .meta("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Tokenizer {
            tokens,
            id_of,
            merges,
            byte_to_char,
            char_to_byte,
            eos_id,
            bos_id,
            add_bos,
        })
    }

    /// Encode text to token ids. No BOS unless the file says to add one.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // The pretokenizer runs on the raw text; the GPT-2 byte encoder is
        // applied to each resulting word (unicode.cpp: split, then
        // unicode_byte_encoding_process on every word).
        let raw: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        if self.add_bos {
            out.push(self.bos_id);
        }
        for (start, end) in split_qwen35(&raw) {
            let word: String = raw[start..end].iter().collect();
            let encoded: Vec<char> = word
                .as_bytes()
                .iter()
                .map(|b| self.byte_to_char[*b as usize])
                .collect();
            for piece in self.bpe(&encoded) {
                match self.id_of.get(&piece) {
                    Some(&id) => out.push(id),
                    None => {
                        // llama.cpp falls back to the individual byte tokens.
                        for ch in piece.chars() {
                            let s: String = std::iter::once(ch).collect();
                            if let Some(&id) = self.id_of.get(&s) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Rank-ordered merges for one pretokenized word.
    fn bpe(&self, word: &[char]) -> Vec<String> {
        let mut syms: Vec<String> = word.iter().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, String, String)> = None;
            for w in syms.windows(2) {
                if let Some(&rank) = self.merges.get(&(w[0].clone(), w[1].clone())) {
                    if best.as_ref().map(|b| rank < b.0).unwrap_or(true) {
                        best = Some((rank, w[0].clone(), w[1].clone()));
                    }
                }
            }
            let Some((_, a, b)) = best else { break };
            let merged = format!("{a}{b}");
            let mut next = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len() && syms[i] == a && syms[i + 1] == b {
                    next.push(merged.clone());
                    i += 2;
                } else {
                    next.push(syms[i].clone());
                    i += 1;
                }
            }
            syms = next;
        }
        syms
    }

    /// The raw bytes of one token, in the byte-encoded representation.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for ch in self.tokens[id as usize].chars() {
            match self.char_to_byte.get(&ch) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        out
    }

    /// Decode ids to text, lossily for byte fragments that cut a codepoint.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

fn bad(key: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("metadata {key} is missing or malformed"),
    )
}

/// GPT-2's `bytes_to_unicode`: printable bytes map to themselves, the rest to
/// U+0100 onwards.
fn byte_encoder() -> ([char; 256], HashMap<char, u8>) {
    let mut table = ['\0'; 256];
    let mut rev = HashMap::new();
    let mut next = 256u32;
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    for b in 0..256u32 {
        let c = if printable(b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        };
        table[b as usize] = c;
        rev.insert(c, b as u8);
    }
    (table, rev)
}

#[derive(Clone, Copy, Default)]
struct Flags {
    letter: bool,
    number: bool,
    mark: bool,
    whitespace: bool,
}

fn flags(c: char) -> Flags {
    Flags {
        letter: c.is_alphabetic(),
        number: c.is_numeric(),
        mark: is_mark(c),
        whitespace: c.is_whitespace(),
    }
}

/// Approximate `\p{M}` (combining marks). Exact for ASCII (no marks).
fn is_mark(c: char) -> bool {
    let u = c as u32;
    (0x0300..=0x036F).contains(&u)
        || (0x0483..=0x0489).contains(&u)
        || (0x0591..=0x05BD).contains(&u)
        || (0x0610..=0x061A).contains(&u)
        || (0x064B..=0x065F).contains(&u)
        || (0x0900..=0x0903).contains(&u)
        || (0x093A..=0x094F).contains(&u)
        || (0x0E31..=0x0E4E).contains(&u)
        || (0x1AB0..=0x1AFF).contains(&u)
        || (0x1DC0..=0x1DFF).contains(&u)
        || (0x20D0..=0x20FF).contains(&u)
        || (0xFE20..=0xFE2F).contains(&u)
}

/// The `tokenizer.ggml.pre = "qwen35"` splitter, port of
/// `unicode_regex_split_custom_qwen35`. Returns char ranges.
///
/// Pattern:
/// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`
/// `|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}`
/// `| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn split_qwen35(chars: &[char]) -> Vec<(usize, usize)> {
    let n = chars.len();
    let get = |p: usize| -> Option<char> { chars.get(p).copied() };
    let fl = |p: usize| -> Flags { get(p).map(flags).unwrap_or_default() };
    let mut out = Vec::new();
    let mut pos = 0usize;

    while pos < n {
        let start = pos;
        let cpt = chars[pos];
        let f = flags(cpt);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' {
            if let Some(c1) = get(pos + 1) {
                let c1 = c1.to_ascii_lowercase();
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    pos += 2;
                    out.push((start, pos));
                    continue;
                }
                if let Some(c2) = get(pos + 2) {
                    let c2 = c2.to_ascii_lowercase();
                    if (c1 == 'r' && c2 == 'e')
                        || (c1 == 'v' && c2 == 'e')
                        || (c1 == 'l' && c2 == 'l')
                    {
                        pos += 3;
                        out.push((start, pos));
                        continue;
                    }
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if !(cpt == '\r' || cpt == '\n' || f.number) {
            let f1 = fl(pos + 1);
            if f.letter || f.mark || f1.mark || f1.letter {
                pos += 1;
                loop {
                    let fx = fl(pos);
                    if fx.letter || fx.mark {
                        pos += 1;
                    } else {
                        break;
                    }
                }
                out.push((start, pos));
                continue;
            }
        }

        // \p{N}
        if f.number {
            pos += 1;
            out.push((start, pos));
            continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        let f2 = if cpt == ' ' { fl(pos + 1) } else { f };
        let punct = |x: Flags| !(x.whitespace || x.letter || x.mark || x.number);
        if punct(f2) && pos < n {
            if cpt == ' ' {
                pos += 1;
            }
            while pos < n && punct(fl(pos)) {
                pos += 1;
            }
            while let Some(c2) = get(pos) {
                if c2 == '\r' || c2 == '\n' {
                    pos += 1;
                } else {
                    break;
                }
            }
            out.push((start, pos));
            continue;
        }

        // whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
        let mut num_ws = 0usize;
        let mut last_end_r_or_n = 0usize;
        while fl(pos + num_ws).whitespace && pos + num_ws < n {
            let c2 = chars[pos + num_ws];
            if c2 == '\r' || c2 == '\n' {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws += 1;
        }
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            out.push((start, pos));
            continue;
        }
        if num_ws > 1 && pos + num_ws < n {
            pos += num_ws - 1;
            out.push((start, pos));
            continue;
        }
        if num_ws > 0 {
            pos += num_ws;
            out.push((start, pos));
            continue;
        }

        // no match: consume one char (llama.cpp does the same)
        pos += 1;
        out.push((start, pos));
    }
    out
}
