# Tokenizer and chat-template fidelity (WP48)

Two gates against llama.cpp for inputs the earlier packages never exercised.

## Differential corpus

`gen_corpus.py` writes a deterministic corpus of 6,789 hex-encoded strings:

| class | strings | coverage |
| --- | ---: | --- |
| `cjk`, `hangul`, `kana`, `tangut`, … | 214+ | every **assigned** codepoint of the CJK, Tangut, kana, bopomofo and Hangul blocks, chunked at 512 |
| `all-assigned` | 550 | every assigned codepoint in Unicode, chunked at 512 |
| `emoji` | 53 + 676 | emoji blocks, ZWJ families/professions, skin tones, keycaps, tag flags, VS15/VS16, all regional-indicator flags, ZWJ at string edges |
| `mark` | 2,513 | every Mn/Mc/Me codepoint as `a<mark>b`, mark chunks, random clusters |
| `arabic`, `hebrew`, `syriac`, `thaana`, `nko`, … | 57 | RTL blocks, presentation forms, bidi controls, mixed LTR/RTL |
| `box`, `block`, `geometric`, `arrows`, `math`, `tech`, `legacy` | 21 | box drawing and diagram characters |
| `ident` | 308 | mixed ASCII/Unicode identifiers, snake/camel, combining forms |
| `ws` | 293 | leading/trailing/runs of space, tab, CR/LF/CRLF, NBSP, thin/ideographic space, ZWSP, BOM, NEL, LS/PS, empty |
| `random` | 1,999 | random valid-UTF-8, including unassigned codepoints |

Run the diff (llama-server on `8092`, hipml built for the host):

```bash
python3 bench/tokenizer/gen_corpus.py /tmp/corpus.txt
target/release/hipml tokenize-batch --gguf MODEL --corpus /tmp/corpus.txt --out /tmp/hipml.txt
python3 bench/tokenizer/diff.py --corpus /tmp/corpus.txt --hipml-ids /tmp/hipml.txt
```

`diff.py` prints the mismatch count by class and every minimal reproducer.
Baseline on the old `is_mark`/`is_alphabetic` approximation: **1 mismatch**
(`👩‍❤️‍👨`, a ZWJ before VS16). After adopting llama's generated category
table: **0/6,789**.

The tables in `crates/hipml-core/src/unicode_tables.rs` are generated from
llama.cpp `d7725699` (`src/unicode-data.cpp`) by `gen_unicode_table.py`; only
`\p{N}`, `\p{L}`, `\p{M}` and `\s` are kept, which are the bits the qwen35
splitter reads.

## Chat-template branches

`template_diff.py` sends the same request body to `/apply-template` and to
`hipml chat --json-file`, then compares rendered bytes and ids. It covers
tool schemas, `tool_calls` arguments (string and object form), JSON escaping
(newline/tab/quote/backslash/control/U+2028), nested arrays/objects/null/bool/
int/float/exponent, multiple calls, empty content, `preserve_thinking`
false/true, `enable_thinking` false, `reasoning_effort` low/medium, and
image/video rejection.

```bash
python3 bench/tokenizer/template_diff.py --gguf MODEL
# 18/18 cases pass
```

## Permanent gate

`bench/tokenizer/fixture.txt` (266 multilingual strings) and
`bench/tokenizer/template_fixture.txt` (16 conversations) store llama's own
ids. `crates/hipml-core/tests/tokenizer_fixture.rs` replays them without a
llama binary:

```bash
HIPML_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  cargo test -p hipml-core --test tokenizer_fixture
```

Without a GGUF the test prints a skip; on the GPU host the model is found at
the default path.

## Coding-agent request

`coding_task.py` renders a multi-turn conversation with a tool call and a Rust
file containing `café`, `世界`, `👋` and `⚠️`, checks prompt identity, then
generates greedily from the same ids on both engines and reports the first
output divergence with llama's margin.
