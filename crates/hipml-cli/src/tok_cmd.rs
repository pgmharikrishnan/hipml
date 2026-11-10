// SPDX-License-Identifier: Apache-2.0
//! `hipml tokenize-batch`: tokenize a hex-encoded corpus in one process.
//!
//! The differential harness (bench/tokenizer/diff.py) needs thousands of
//! strings tokenized by hipml and by llama without paying a process start and
//! GGUF open per string. Each input line is `hex(text)` optionally followed by
//! a tab and a label; each output line is `index<TAB>label<TAB>[ids]`.
//!
//! Hex is the wire format because the corpus deliberately contains newlines,
//! NULs, lone combining marks and RTL controls that no line-based text format
//! can carry unambiguously.

use hipml_core::gguf::Gguf;
use hipml_core::tokenizer::Tokenizer;

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd hex length".to_string());
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    let nib = |c: u8| -> Result<u8, String> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(format!("bad hex digit {:?}", c as char)),
        }
    };
    let mut i = 0;
    while i < b.len() {
        out.push((nib(b[i])? << 4) | nib(b[i + 1])?);
        i += 2;
    }
    Ok(out)
}

pub fn run(args: &[String]) -> i32 {
    let mut gguf = None;
    let mut corpus = None;
    let mut out_path = None;
    let mut special = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                gguf = args.get(i + 1).cloned();
                i += 2;
            }
            "--corpus" => {
                corpus = args.get(i + 1).cloned();
                i += 2;
            }
            "--out" => {
                out_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--special" => {
                special = true;
                i += 1;
            }
            other => {
                eprintln!("tokenize-batch: unknown argument {other}");
                return 2;
            }
        }
    }
    let (Some(gguf), Some(corpus)) = (gguf, corpus) else {
        eprintln!("tokenize-batch: --gguf and --corpus are required");
        return 2;
    };
    let g = match Gguf::open(&gguf) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("tokenize-batch: cannot open {gguf}: {e}");
            return 1;
        }
    };
    let tok = match Tokenizer::from_gguf(&g) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenize-batch: {e}");
            return 1;
        }
    };
    let text = match std::fs::read_to_string(&corpus) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenize-batch: read {corpus}: {e}");
            return 1;
        }
    };
    let mut out = String::with_capacity(text.len() * 2);
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (hex, label) = line.split_once('\t').unwrap_or((line, ""));
        let bytes = match unhex(hex) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("tokenize-batch: line {}: {e}", index + 1);
                return 1;
            }
        };
        let s = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("tokenize-batch: line {} is not UTF-8: {e}", index + 1);
                return 1;
            }
        };
        let ids = if special {
            tok.encode_special(&s)
        } else {
            tok.encode(&s)
        };
        out.push_str(&format!("{index}\t{label}\t{ids:?}\n"));
    }
    match out_path {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, out) {
                eprintln!("tokenize-batch: write {path}: {e}");
                return 1;
            }
        }
        None => print!("{out}"),
    }
    0
}
