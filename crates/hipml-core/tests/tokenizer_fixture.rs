//! Permanent round-trip gate for the tokenizer and the chat template.
//!
//! Both fixtures store llama.cpp's own ids (captured by
//! `bench/tokenizer/diff.py` and `bench/tokenizer/template_diff.py`), so this
//! test needs no llama binary at test time. It does need the GGUF whose
//! tokenizer the fixture was captured from; set `HIPML_GGUF`, or run it on a
//! host with the model at the usual path. Without the model the test reports a
//! skip instead of failing every unrelated `cargo test` run.
//!
//! Corpus fixture line: `hex(text) TAB label TAB [ids]`.
//! Template fixture line: `hex(json request) TAB label TAB [ids]`.

use std::path::{Path, PathBuf};

use hipml_core::chat::render;
use hipml_core::chat_request;
use hipml_core::gguf::Gguf;
use hipml_core::json;
use hipml_core::tokenizer::Tokenizer;

const CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/tokenizer/fixture.txt"
);
const TEMPLATE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/tokenizer/template_fixture.txt"
);

fn model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HIPML_GGUF") {
        return Some(PathBuf::from(p));
    }
    let mut candidates = vec![
        PathBuf::from(r"C:\models\Qwen3.8-27B-Q4_K_M.gguf"),
        PathBuf::from("/models/Qwen3.8-27B-Q4_K_M.gguf"),
    ];
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(Path::new(&home).join("models/Qwen3.8-27B-Q4_K_M.gguf"));
    }
    candidates.into_iter().find(|p| p.exists())
}

fn tokenizer() -> Option<Tokenizer> {
    let Some(path) = model() else {
        eprintln!(
            "tokenizer fixture: no GGUF (set HIPML_GGUF), skipping id-equality gate; \
             fixtures are {} and {}",
            CORPUS, TEMPLATE
        );
        return None;
    };
    let g = Gguf::open(&path).expect("open GGUF");
    Some(Tokenizer::from_gguf(&g).expect("tokenizer from GGUF"))
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length");
    let b = s.as_bytes();
    (0..b.len() / 2)
        .map(|i| {
            let nib = |c: u8| match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                _ => panic!("bad hex digit {}", c as char),
            };
            (nib(b[2 * i]) << 4) | nib(b[2 * i + 1])
        })
        .collect()
}

fn read_ids(text: &str) -> Vec<u32> {
    let v = json::parse(text).expect("ids array");
    v.as_arr()
        .expect("ids array")
        .iter()
        .map(|x| x.as_u64().expect("non-negative id") as u32)
        .collect()
}

fn lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"))
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn multilingual_corpus_matches_llama_ids() {
    let Some(tok) = tokenizer() else { return };
    let mut checked = 0;
    for line in lines(CORPUS) {
        let mut f = line.splitn(3, '\t');
        let hex = f.next().expect("hex");
        let label = f.next().expect("label");
        let want = read_ids(f.next().expect("ids"));
        let text = String::from_utf8(unhex(hex)).expect("fixture is UTF-8");
        let got = tok.encode(&text);
        assert_eq!(got, want, "tokenizer mismatch for {label:?}: text={text:?}");
        checked += 1;
    }
    assert!(checked >= 200, "fixture unexpectedly small: {checked}");
    eprintln!("tokenizer fixture: {checked} strings match llama");
}

#[test]
fn tool_call_and_preserve_thinking_templates_match_llama_ids() {
    let Some(tok) = tokenizer() else { return };
    let template = tok
        .chat_template
        .as_deref()
        .expect("model has tokenizer.chat_template");
    let mut checked = 0;
    for line in lines(TEMPLATE) {
        let mut f = line.splitn(3, '\t');
        let hex = f.next().expect("hex");
        let label = f.next().expect("label");
        let want = read_ids(f.next().expect("ids"));
        let body = String::from_utf8(unhex(hex)).expect("request is UTF-8");
        let (messages, opts) =
            chat_request::load(&body).unwrap_or_else(|e| panic!("{label}: request: {e}"));
        let text =
            render(template, &messages, &opts).unwrap_or_else(|e| panic!("{label}: render: {e}"));
        let got = tok.encode_special(&text);
        assert_eq!(got, want, "template mismatch for {label:?}: {text:?}");
        checked += 1;
    }
    assert!(
        checked >= 16,
        "template fixture unexpectedly small: {checked}"
    );
    eprintln!("template fixture: {checked} conversations match llama");
}
