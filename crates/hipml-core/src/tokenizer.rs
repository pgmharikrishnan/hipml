// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//! GPT-2-style byte-level BPE, as stored in the GGUF tokenizer tables.
//!
//! Ports `llm_tokenizer_bpe_session::tokenize` and
//! `unicode_regex_split_custom_qwen35` (`src/llama-vocab.cpp`,
//! `src/unicode.cpp`): the text is mapped through the GPT-2 byte encoder, split
//! with the `tokenizer.ggml.pre = "qwen35"` pattern, then merged by rank.
//!
//! The general-category tests (`\p{L}`, `\p{N}`, `\p{M}`, `\s`) read the same
//! generated table llama.cpp uses (`unicode_tables`, from `unicode-data.cpp`),
//! not Rust's `char::is_alphabetic`/`is_numeric` approximations: the splitter
//! must agree with the reference tokenizer on every script, not only ASCII.

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
    /// `tokenizer.ggml.token_type` per id (1 normal, 3 control, 4 user-defined,
    /// 6 byte; the GGUF llama.cpp enum).
    token_type: Vec<u32>,
    /// Control and user-defined tokens, as (text, id), longest first. These are
    /// the strings llama.cpp treats as special: they are emitted as one id and
    /// never enter the BPE merge loop.
    special: Vec<(String, u32)>,
    special_by_id: Vec<bool>,
    /// `tokenizer.chat_template`, verbatim from the file. `chat::render` refuses
    /// a template it does not understand rather than guessing a chat format.
    pub chat_template: Option<String>,
    pub eos_id: u32,
    pub bos_id: u32,
    pub add_bos: bool,
    pub add_eos: bool,
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

        // llama.cpp's special-token set: CONTROL (3) and USER_DEFINED (4).
        let token_type = g
            .meta("tokenizer.ggml.token_type")
            .and_then(|v| v.as_u32s())
            .unwrap_or_else(|| vec![1u32; tokens.len()]);
        if token_type.len() != tokens.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "token_type has {} entries for {} tokens",
                    token_type.len(),
                    tokens.len()
                ),
            ));
        }
        let mut special_by_id = vec![false; tokens.len()];
        let mut special: Vec<(String, u32)> = Vec::new();
        for (i, (&ty, text)) in token_type.iter().zip(tokens.iter()).enumerate() {
            if ty == 3 || ty == 4 {
                special_by_id[i] = true;
                special.push((text.clone(), i as u32));
            }
        }
        // Leftmost-longest matching needs longer strings first.
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

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
        let add_eos = g
            .meta("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let chat_template = g
            .meta("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        Ok(Tokenizer {
            tokens,
            id_of,
            merges,
            byte_to_char,
            char_to_byte,
            token_type,
            special,
            special_by_id,
            chat_template,
            eos_id,
            bos_id,
            add_bos,
            add_eos,
        })
    }

    /// Encode text to token ids. No BOS unless the file says to add one.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.push(self.bos_id);
        }
        out.extend(self.encode_normal(text));
        out
    }

    /// As [`encode`], but text that spells a control/user-defined token is
    /// emitted as that single id. This is llama.cpp's `parse_special = true`
    /// path and is what a chat template needs: `<|im_end|>` must be 248046,
    /// never a handful of BPE pieces.
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            out.push(self.bos_id);
        }
        let bytes = text.as_bytes();
        let mut pos = 0usize;
        let mut plain = 0usize;
        while pos < bytes.len() {
            let hit = self
                .special
                .iter()
                .find(|(s, _)| bytes[pos..].starts_with(s.as_bytes()));
            match hit {
                Some((s, id)) => {
                    if plain < pos {
                        out.extend(self.encode_normal(&text[plain..pos]));
                    }
                    out.push(*id);
                    pos += s.len();
                    plain = pos;
                }
                None => {
                    // Advance by one UTF-8 character, never into a codepoint.
                    pos += text[pos..].chars().next().map(char::len_utf8).unwrap_or(1);
                }
            }
        }
        if plain < bytes.len() {
            out.extend(self.encode_normal(&text[plain..]));
        }
        out
    }

    /// One token id for a special token's literal text, if the file has it.
    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.special
            .iter()
            .find(|(s, _)| s == text)
            .map(|(_, id)| *id)
    }

    /// Whether id is a control/user-defined token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_by_id.get(id as usize).copied().unwrap_or(false)
    }

    /// The token's GGUF type (1 normal, 3 control, 4 user-defined, 6 byte).
    pub fn token_type(&self, id: u32) -> u32 {
        self.token_type.get(id as usize).copied().unwrap_or(1)
    }

    /// Encode a run of ordinary text: pretokenize, byte-encode, merge.
    fn encode_normal(&self, text: &str) -> Vec<u32> {
        // The pretokenizer runs on the raw text; the GPT-2 byte encoder is
        // applied to each resulting word (unicode.cpp: split, then
        // unicode_byte_encoding_process on every word).
        let raw: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
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

    /// Decode as a chat completion: control/user-defined tokens are not text
    /// and are omitted, matching llama.cpp's default `special = false` detokenize.
    pub fn decode_text(&self, ids: &[u32]) -> String {
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| !self.is_special(id))
            .collect();
        self.decode(&filtered)
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
    let b = super::unicode_tables::bits(c);
    Flags {
        letter: b & super::unicode_tables::LETTER != 0,
        number: b & super::unicode_tables::NUMBER != 0,
        mark: b & super::unicode_tables::MARK != 0,
        whitespace: b & super::unicode_tables::WHITESPACE != 0,
    }
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
