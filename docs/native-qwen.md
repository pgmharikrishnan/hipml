# Native Qwen3.8-27B (stock Q4_K_M, Unsloth UD variants)

`feat/native-qwen`, base `6a1207f` (`feat/release`). Every earlier hipml
measurement used a private fine-tune with the same geometry. This package
points the engine at the **stock** model family and removes the fine-tune
constants that made it work only there. The release model is
`Qwen3.8-27B-Q4_K_M.gguf`, the file [`release-run.md`](release-run.md)
measured; the Unsloth dynamic-quant (UD) variants of the same model load and
generate as well — see [Status of the UD types](#status-of-the-ud-types).

| file | bytes | tensors |
| --- | ---: | ---: |
| `Qwen3.8-27B-Q4_K_M.gguf` (the release model) | 16,810,714,336 (15.65 GiB) | 866 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,559,178,144 (16.35 GiB) | 866 |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25,299,061,664 (23.56 GiB) | 866 |
| private fine-tune (`<model>.gguf`) | 17,192,421,376 (16.01 GiB) | 851 |

Both stock files carry the MTP (`nextn`) block **inline** as `blk.64.*`;
`block_count` is 65 and `nextn_predict_layers` is 1.

## Status of the UD types

A Unsloth UD build mixes nine tensor types into one file. hipml now reads all
nine; what differs is *how*, and that is the whole of the cost:

| type | GGUF id | how hipml serves it | resident cost |
| --- | ---: | --- | --- |
| Q4_K | 12 | block form: `gemv_q4_k`, batched `_b`, `prefill_q4_k(_w2h)` tile | 1:1 |
| Q5_K | 13 | block form: `gemv_q5_k`, batched `_b`, `prefill_q5_k(_w2h)` tile | 1:1 |
| Q6_K | 14 | block form: `gemv_q6_k` (+ batched), `prefill_q6_k(_w2h)`, `embed_q6_k_row` | 1:1 |
| Q8_0 | 8 | block form: `gemv_q8_0`, batched `_b`, `prefill_q8_0(_w2h)` | 1:1 |
| IQ4_XS | 23 | block form: `gemv_iq4_xs` (+ `_b`), `prefill_iq4_xs(_w2h)` | 1:1 |
| IQ4_NL | 20 | block form: `gemv_iq4_nl` (+ `_b`), `prefill_iq4_nl(_w2h)` | 1:1 |
| F32 | 0 | as stored (`gemv_f32`, `prefill_f32`) | 1:1 |
| Q3_K | 11 | **expanded to F32 at load** (host `dequant_q3_k`), then the F32 path | 4.00× |
| IQ3_S | 21 | **expanded to F32 at load** (host `dequant_iq3_s`), then the F32 path | 4.00× |

- The six block types stay in their GGUF blocks in VRAM; the dequant happens
  inside the GEMV or the dequant-in-shared prefill tile, never into a resident
  copy.
- Q3_K and IQ3_S are the exception: there is no block-form decode kernel for
  them, so the loader dequantises those rows on the host once at load and
  uploads F32. They are 4× their file size.
- Both UD files load with `unsupported: none` and round-trip 866/866
  byte-identical. On `UD-Q4_K_M` the expansion is **4 tensors,
  153,190,400 B on disk → 1,426,063,360 B resident (+1,272,872,960 B,
  +1.19 GiB)**. `UD-Q4_K_XL` expands the same four tensors, 153,190,400 B →
  612,761,600 B (**+459,571,200 B, +0.43 GiB**; from the table below, not
  re-measured in this tree).
- Still unmeasured: Q3_K and IQ3_S never run in block form (by design — the
  row count is small enough that expansion is the cheap choice), the 64-row
  f32-tile `prefill_iq4_*` entries are wired but the default WMMA path selects
  `_w2h` instead, and nothing beyond 8,192 tokens, batching, MTP or the cache
  gates has been run on a UD file.

## Architecture config (read from the GGUF by `hipml load`)

```
config (qwen35): 65 blocks (64 decoder + 1 nextn), hidden 5120, ffn 17408,
  heads 24 / kv 4, head_dim 256, vocab 248320, eps 1e-6, rope base 1e7 rot 64
  sections [11, 11, 10, 0], ctx 262144
  ssm: conv 4, state 128, groups 16, dt_rank 48, inner 6144 (head_v 128)
  full-attention layers (16): [3, 7, ..., 63]
```

| field | stock Qwen3.8-27B | private fine-tune |
| --- | --- | --- |
| `general.architecture` | `qwen35` | `qwen35` |
| `block_count` | **65** | **64** |
| `nextn_predict_layers` | **1** | absent |
| decoder blocks | 64 | 64 |
| `embedding_length` | 5120 | 5120 |
| `feed_forward_length` | 17408 | 17408 |
| `attention.head_count` | 24 | 24 |
| `attention.head_count_kv` | 4 | 4 |
| `attention.key_length` / `value_length` | 256 / 256 | 256 / 256 |
| `n_vocab` (tensor rows) | 248320 | 248320 |
| `rope.freq_base` | 1e7 | 1e7 |
| `rope.dimension_count` | 64 | 64 |
| `rope.dimension_sections` | [11, 11, 10, 0] | [11, 11, 10, 0] |
| `attention.layer_norm_rms_epsilon` | 1e-6 | 1e-6 |
| `full_attention_interval` | 4 | 4 |
| `attention.recurrent_layers` | absent (interval derived) | explicit 64-entry mask |
| `ssm.conv_kernel` / `state_size` | 4 / 128 | 4 / 128 |
| `ssm.group_count` / `time_step_rank` | 16 / 48 | 16 / 48 |
| `ssm.inner_size` | 6144 | 6144 |
| `context_length` | 262144 | 262144 |
| `tokenizer.ggml.model` / `pre` | gpt2 / qwen35 | gpt2 / qwen35 |
| `bos` / `eos` / `padding` | 248044 / 248046 / 248055 | 248044 / 248046 / 248044 |
| `add_bos` / `add_eos` / `clean_spaces` | absent (false defaults) | false / false / false |
| `tokenizer.chat_template` | 9,993 B, sha256 `12827f24…aaadce` | 8,952 B, sha256 `c3cf9e34…d7a81041` |

Only four fields differ: `block_count`, `nextn_predict_layers`,
`attention.recurrent_layers`, and the tokenizer section (padding id plus the
template body). Heads, hidden, ffn, head_dim, vocab, rope and all SSM sizes are
identical. The template adds merged leading system/developer turns,
`high` → `xhigh`, and mapping-only tool arguments; the text path is otherwise
the same.

## Tensor type histogram

`Qwen3.8-27B-UD-Q4_K_M.gguf` — the UD target, measured in this tree:

| type | tensors | bytes | GiB | resident | engine |
| --- | ---: | ---: | ---: | ---: | --- |
| IQ4_XS (23) | 142 | 5,695,897,600 | 5.30 | 5.30 | block form |
| Q5_K | 142 | 5,230,100,480 | 4.87 | 4.87 | block form |
| Q4_K | 74 | 3,142,287,360 | 2.93 | 2.93 | block form |
| Q6_K | 28 | 1,804,185,600 | 1.68 | 1.68 | block form |
| IQ4_NL (20) | 9 | 430,571,520 | 0.40 | 0.40 | block form |
| Q8_0 | 107 | 86,343,680 | 0.08 | 0.08 | block form |
| Q3_K (11) | 2 | 76,595,200 | 0.07 | 0.66 | expanded to F32 |
| IQ3_S (21) | 2 | 76,595,200 | 0.07 | 0.66 | expanded to F32 |
| F32 | 360 | 10,686,464 | 0.01 | 0.01 | as stored |
| total | 866 | 16,553,263,104 | 15.42 | **16.60** | |

`Qwen3.8-27B-UD-Q4_K_XL.gguf` (bytes as stored; the four Q3_K/IQ3_S tensors
become 4× resident at load, see above):

| type | tensors | bytes | GiB | engine |
| --- | ---: | ---: | ---: | --- |
| Q5_K | 191 | 7,926,251,520 | 7.38 | block form |
| IQ4_XS (23) | 70 | 3,125,084,160 | 2.91 | block form |
| Q4_K | 69 | 3,059,712,000 | 2.85 | block form |
| Q6_K | 56 | 2,862,182,400 | 2.67 | block form |
| IQ4_NL (20) | 6 | 280,166,400 | 0.26 | block form |
| Q8_0 | 110 | 130,908,160 | 0.12 | block form |
| Q3_K (11) | 3 | 114,892,800 | 0.11 | expanded to F32 |
| IQ3_S (21) | 1 | 38,297,600 | 0.04 | expanded to F32 |
| F32 | 360 | 10,686,464 | 0.01 | as stored |
| total | 866 | 17,548,181,504 | 16.34 | |

`Qwen3.8-27B-UD-Q6_K_XL.gguf`:

| type | tensors | bytes | GiB |
| --- | ---: | ---: | ---: |
| Q8_0 | 310 | 13,525,319,680 | 12.60 |
| Q6_K | 168 | 10,663,833,600 | 9.93 |
| Q5_K | 27 | 1,070,530,560 | 1.00 |
| Q4_K | 1 | 17,694,720 | 0.02 |
| F32 | 360 | 10,686,464 | 0.01 |
| total | 866 | 25,288,065,024 | 23.55 |

The fine-tune, for comparison: Q4_K 257 / 10.66 GiB, Q6_K 79 / 3.84, Q8_0 64 / 1.30,
Q5_K 2 / 0.11, F32 449 / 0.10; 851 tensors, 16.01 GiB.

Every type in all three UD-relevant files is now served, so the mix no longer
decides whether a file runs — only how its Q3_K/IQ3_S share is stored.

## Load result

```
UD-Q4_K_M: 866 tensors  16,553,263,104 B (15.42 GiB)  read+upload 5.51 s (3.01 GB/s)
           hipMemGetInfo free 34,023,604,224 -> 16,194,994,176 B
           resident delta 17,828,610,048 B (16.60 GiB)
           expanded at load: 4 tensors, 153,190,400 B -> 1,426,063,360 B (+1.19 GiB)
           unsupported: none
           round-trip 866/866 byte-identical
           embedding rows 0/1/1234/100000/248319: 5120/5120 exact, max_abs 0
           lm_head Q6_K 248,320 logits in 5.002 ms (208.5 GB/s)
Q4_K_XL: 866 tensors  17,548,181,504 B (16.34 GiB)  read+upload 10.42 s (1.68 GB/s)
         hipMemGetInfo free 34,023,604,224 -> 16,448,159,744 B
         resident delta 17,575,444,480 B (16.37 GiB)  [before the F32 expansion]
Q6_K_XL: 866 tensors  25,288,065,024 B (23.55 GiB)  read+upload 16.55 s (1.53 GB/s)
         resident delta 25,263,079,424 B (23.53 GiB); 8.76 GB free after
```

## Generation and speed, `UD-Q4_K_M`

The file that used to stop at `type_23 has no GEMV kernel` now runs end to end.
169-token fixture, chunk 2048, q8 KV, against the CPU f32 reference:

```
prefill logits: max |gpu - cpu| = 0.1019 at 29832 (gpu argmax 248068 / cpu argmax 248068)
PASS: 20 ids identical
```

Greedy, `--device --chunk 2048 --kv q8`, two prompts, each run twice:
`The capital of France is` → `` Paris.\nThe capital of Germany is Berlin.\n…``
(24 ids, identical on rerun); `Explain in one sentence why the sky is blue.`
→ "The sky appears blue because the Earth's atmosphere scatters shorter
wavelengths…" (32 ids, identical on rerun).

```
resident: 15.42 GiB in 866 tensors; 15.61 GiB streamed/token; q8 KV 0.266 GiB at 8,192
                          prefill tok/s            decode tok/s
depth 0                    880.45 (373 tok)         27.84
depth 8,192              1,104.00 (8,192 tok)       27.07

with HIPML_Q5K_COPY_GB=0 (no Q5_K tile kernel, batched GEMV instead):
depth 0                    289.76                   27.62
depth 8,192                297.68                   26.92
```

The Q5_K dequant-in-shared tile kernel is the whole prefill difference on this
file: 3.0× / 3.7× over the batched-GEMV fallback, with the same last id at both
depths (220 and 292).

The stock `Qwen3.8-27B-Q4_K_M.gguf` is unchanged by all of this — same oracle
gate (`PASS 20/20`, `max |gpu - cpu| = 0.1481 at 34114`, the published value)
and prefill 1,000.31 / 1,235.14 tok/s with decode 30.06 / 29.21 tok/s at depths
0 and 8,192.

`Qwen3.8-27B-UD-Q6_K_XL.gguf` is not measured in this tree, but needs no format
of its own: Q6_K embedding rows and the Q5_K attention/prefill entries it was
missing both exist now.

## Hard-coded fine-tune values replaced

| was | now | GGUF source |
| --- | --- | --- |
| `n_layer = block_count` (65) | `block_count - nextn_predict_layers` (64) | `qwen35.nextn_predict_layers` |
| `MTP_BLK = 64` const | `cfg.n_layer` | `block_count - nextn_predict_layers` (64 in both files) |
| probe token `248_319` ("the 248320-row vocab") | last row of `token_embd.weight` | tensor shape |
| `lm_head_reps` required `output.weight == Q6_K` | GEMV chosen from the head's own type | tensor type |
| `supported()` returned false for Q5_K | true (decode/batched/prefill kernels exist) | kernel manifest |
| IQ4_XS/IQ4_NL/IQ3_S aborted the parser | sized in `size_only`, so `load` reports them | type ids 23/20/21 |
| plain `encode()` never split specials | splits user-defined tokens (type 4), as llama `parse_special=false` | `tokenizer.ggml.token_type` |

`full_attention_interval`, rope base/sections, eps, SSM sizes, heads and
`n_vocab` were already read from GGUF; the interval is only a fallback when
`attention.recurrent_layers` is absent, which is exactly the stock file's case.
With the old code the stock file would have run `blk.64` as a 65th decoder
layer and produced wrong logits; `hipml load` now prints
`65 blocks (64 decoder + 1 nextn)`.

`hipml load` also grew a config summary so the metadata is visible without a
separate `info` command.

## Tokenizer and template exactness

20 multilingual/chat strings × `parse_special` false/true = 40 cases,
`hipml tokenize` vs llama-server `/tokenize` (`add_special=false`) on the
stock Q4_K_XL file:

```
cases 40  mismatches 0
```

Before the user-defined-token fix it was 39/40: plain mode disagreed on
`<tool_call>…</tool_call>` (ids 248058/248059, token type 4). llama splits
type-4 tokens even with `parse_special=false`; hipml did not. The corpus:
CJK, Hangul, kana, Arabic, Hebrew, Cyrillic, Devanagari, Thai, accented Latin,
combining marks, ZWJ emoji/flags, code, JSON, template fragments, whitespace,
mixed bidi, tool-call XML, URL escapes.

Committed fixtures also pass against the stock file:

```
$ HIPML_GGUF=<HIPML_GGUF> \
    cargo test -p hipml-core --test tokenizer_fixture
2 passed (266 multilingual strings, 16 template conversations)
```

One chat turn, native template vs `/apply-template`:

| case | text | ids |
| --- | --- | --- |
| user "hi" | byte-identical (297 B) | 53/53 identical |
| system "  be terse  " + user "hi" | byte-identical (307 B) | 56/56 identical |
| tool call round trip (`ping`, empty args) | byte-identical (466 B) | 85/85 identical |

Only mismatch found: two leading system turns. hipml rejects
(`system message must be at the beginning`); the stock template merges them
(`…\n\nfirst\nsecond<|im_end|>`). Minimal reproducer:

```json
{"messages":[{"role":"system","content":"first"},
             {"role":"system","content":"second"},
             {"role":"user","content":"hi"}]}
```

## Native-model speed

Same fixture and harness as `bench/` (`job.py`, 3 cold reps, 33 generated
tokens, q8/q8_0 KV, no cache reuse), **stock Q4_K_XL**, `prose_explanation`,
llama.cpp only:

| depth | prompt | prefill tok/s (median) | decode tok/s (median) | TTFT (median) | peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 373 | 1,006.6 | 30.20 | 371 ms | 16,485 MiB |
| 8,192 | 8,192 | 1,284.7 | 29.50 | 6,389 ms | 16,760 MiB |

Reps: prefill 870.4 / 1,006.6 / 1,029.3 and 1,284.7 / 1,287.7 / 1,279.2;
decode 29.71 / 30.21 / 30.20 and 29.44 / 29.57 / 29.50.
llama.cpp build `11820`, commit `d7725699`, `-ngl 99`.

hipml's own stock numbers are in
[`benchmarks.md`](benchmarks.md) (1,198 / 1,252 tok/s prefill, 29.2 / 28.8
decode); the UD files are measured above.

## Commands

```bash
# sync and build
git archive HEAD -o /tmp/x.tar && scp /tmp/x.tar gpu-host:x.tar
ssh gpu-host "tar -xf x.tar -C <REPO>"
ssh gpu-host "cd <REPO> && set HIP_PATH=<ROCM>& \
  set HIPML_ARCH=gfx1201& set PATH=<ROCM>\bin;%PATH%& \
  cargo build --release --features hip"

# config + residency + type table + unsupported list
target\release\hipml.exe load --gguf <HIPML_GGUF> --roundtrip 4

# greedy generation over the chunked prefill graph
target\release\hipml.exe generate --device --gguf <HIPML_GGUF> \
  --ids 1,2,3 --tokens 2 --ctx 64 --kv q8

# tokenizer differential (llama-server cpu on 8092, add_special=false)
python bench/job.py --engine llama --prompt prose_explanation --depth 0 --reps 3 \
  --model <HIPML_GGUF>
```

## Checks on this tree

| check | result |
| --- | --- |
| `cargo test --workspace` | local and gpu-host: all pass (0 failed) |
| `cargo check --workspace --features hip` | clean |
| `hipml parity --deep` | 56 passed, 0 failed, 1 skipped |
| `hipml oracle-gate` (fine-tune, 169 ids, chunk 2048) | PASS, 20/20 ids identical |
| `hipml oracle-gate` (stock Q4_K_M, 169 ids, chunk 2048) | PASS, 20/20, max abs 0.1481 at 34114 |
| `hipml oracle-gate` (UD-Q4_K_M, 169 ids, chunk 2048) | PASS, 20/20, max abs 0.1019 at 29832 |
| Fine-tune generation after the config change | 64 layers, 8 tokens emitted |
| Q4_K_XL load checks | 4/4 byte-identical, embeddings 5120/5120 exact |
| UD-Q4_K_M load checks | 866/866 byte-identical, embeddings 5120/5120 exact |
| tokenizer, 20 strings × 2 modes | 40/40 ids identical |
| template, 3 single-turn cases | text and ids identical |

The fine-tune still reports 64 layers and generates, so subtracting
`nextn_predict_layers` did not change the fine-tune path.

## Unverified

- 128-token greedy parity and logit margins on a stock file: the 20-step
  oracle gate above is the only published logit comparison.
- `UD-Q6_K_XL` end to end; it needs no new format, but it was not run here.
- Anything past 8,192 tokens on a UD file: batching, MTP, prefix cache and the
  KV/SSM cache gates are measured on the stock model only.
- Chat-template branches specific to the stock template: developer-role
  messages, merged multi-system turns (one reproducer above), and the stricter
  tool-argument mapping rule. The committed template fixture passes, but these
  branches are not in it.
