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
| Q3_K | 11 | block form: `gemv_q3_k` (+ `_b`), `prefill_q3_k(_w2h)` tile | 1:1 |
| IQ3_S | 21 | block form: `gemv_iq3_s` (+ `_b`), `prefill_iq3_s(_w2h)` tile | 1:1 |

- The eight block types stay in their GGUF blocks in VRAM; the dequant happens
  inside the GEMV or the dequant-in-shared prefill tile, never into a resident
  copy. Q3_K and IQ3_S used to be the exception (host dequant at load, F32
  resident, 9.31× their file size); both now have a decode GEMV, a batched GEMV
  and a prefill tile, so a 110-byte 256-value block stays 110 bytes.
- `HIPML_EXPAND_SMALL_QUANTS=1` restores the old host expansion for those two
  types. It is a debug fallback, off by default; set it to compare against the
  pre-kernel path, not to run the model.
- Both UD files load with `unsupported: none` and round-trip 866/866
  byte-identical. `UD-Q4_K_M` now holds 16,553,263,104 B (15.42 GiB) resident
  (was 17,826,136,064 B / 16.60 GiB, the difference being the 1.19 GiB the two
  Q3_K and two IQ3_S tensors used to gain), and `UD-Q4_K_XL` 17,548,181,504 B
  (16.34 GiB, was 17.53 GiB).
- **Batched decode reaches the skinny tile for all eight block types.** The tile
  has an entry per resident format (`gemm_skinny_{q4_k,q5_k,q6_k,q8_0,iq4_xs,
  iq4_nl,q3_k,iq3_s}`), chosen from one table (`BATCH_RESIDENT` in
  `crates/hipml-kernels`). Before that, Q5_K (32% of this file's bytes) and the
  four IQ/small-K formats had no entry and every batched projection fell back to
  the exact `_b` GEMV, which is why 16 slots on UD ran at 97.42 tok/s against
  216.88 for the stock file. With the entries, 16 slots reach 201.59 on the
  approximate tile; the follow-up's Q5_K load-order rewrite took it to 220.65
  (2.27x its own baseline), and the re-verification on `409f4ad` measures
  **222.13** tok/s in cell c; the exact path is 96.87.
  See [`notes/serve-ud-batch.md`](notes/serve-ud-batch.md).
- Batching runs past 8,192 tokens per slot on a UD file: cell b is 8 slots at
  16,384 tokens each and cell c is 16 slots at 8,192, with the prefix cache
  restoring 2,396 of the 2,423-token prompt. Still unmeasured: MTP over a batch,
  and `cache-gate` on a UD file. The 64-row f32-tile `prefill_iq4_*` entries are
  wired but the default WMMA path selects `_w2h` instead.

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
| Q3_K (11) | 2 | 76,595,200 | 0.07 | 0.07 | block form |
| IQ3_S (21) | 2 | 76,595,200 | 0.07 | 0.07 | block form |
| F32 | 360 | 10,686,464 | 0.01 | 0.01 | as stored |
| total | 866 | 16,553,263,104 | 15.42 | **15.42** | |

`Qwen3.8-27B-UD-Q4_K_XL.gguf`, measured in this tree:

| type | tensors | bytes | GiB | resident | engine |
| --- | ---: | ---: | ---: | ---: | --- |
| Q5_K | 191 | 7,926,251,520 | 7.38 | 7.38 | block form |
| IQ4_XS (23) | 70 | 3,125,084,160 | 2.91 | 2.91 | block form |
| Q4_K | 69 | 3,059,712,000 | 2.85 | 2.85 | block form |
| Q6_K | 56 | 2,862,182,400 | 2.67 | 2.67 | block form |
| IQ4_NL (20) | 6 | 280,166,400 | 0.26 | 0.26 | block form |
| Q8_0 | 110 | 130,908,160 | 0.12 | 0.12 | block form |
| Q3_K (11) | 3 | 114,892,800 | 0.11 | 0.11 | block form |
| IQ3_S (21) | 1 | 38,297,600 | 0.04 | 0.04 | block form |
| F32 | 360 | 10,686,464 | 0.01 | 0.01 | as stored |
| total | 866 | 17,548,181,504 | 16.34 | **16.34** | |

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

Every type in all three UD-relevant files is now served in block form, so the
mix no longer decides whether a file runs, and Q3_K/IQ3_S no longer cost more
resident bytes than their share of the file.

## Load result

```
UD-Q4_K_M: 866 tensors  16,553,263,104 B (15.42 GiB)  read+upload 8.90 s (1.86 GB/s)
           hipMemGetInfo free 34,023,604,224 -> 17,467,703,296 B
           resident delta 16,555,900,928 B (15.42 GiB)
           expanded at load: 4 tensors, 153,190,400 B -> 1,426,063,360 B (+1.19 GiB)   [HIPML_EXPAND_SMALL_QUANTS=1 only]
           unsupported: none
           round-trip 866/866 byte-identical
           embedding rows 0/1/1234/100000/248319: 5120/5120 exact, max_abs 0
           lm_head Q6_K 248,320 logits in 5.782 ms (180.4 GB/s)
Q4_K_XL: 866 tensors  17,548,181,504 B (16.34 GiB)  read+upload 10.83 s (1.62 GB/s)
         hipMemGetInfo free 34,023,604,224 -> 16,448,159,744 B
         resident delta 17,575,444,480 B (16.37 GiB)
         unsupported: none
         round-trip 866/866 byte-identical
Q6_K_XL: 866 tensors  25,288,065,024 B (23.55 GiB)
         resident delta 25,263,079,424 B (23.53 GiB); 8.76 GB free after
```

## Generation and speed, `UD-Q4_K_M`

The file that used to stop at `type_23 has no GEMV kernel` now runs end to end.
169-token fixture, chunk 2048, q8 KV, against the CPU f32 reference:

```
prefill logits: max |gpu - cpu| = 0.1086 at 101748 (gpu argmax 248068 / cpu argmax 248068)
PASS: 20 ids identical
```

Greedy, `--device --chunk 2048 --kv q8`, two prompts, each run twice:
`The capital of France is` → `` Paris.\nThe capital of Germany is Berlin.\n…``
(24 ids, identical on rerun); `Explain in one sentence why the sky is blue.`
→ "The sky appears blue because the Earth's atmosphere scatters shorter
wavelengths…" (32 ids, identical on rerun).

```
resident: 15.42 GiB in 866 tensors; 14.42 GiB streamed/token; q8 KV 0.268 GiB at 8,192
```

The current numbers are the re-verification on `409f4ad`: one binary, one
session, 3 reps in each of 2 separate processes, llama.cpp build 11820 in the
same window on the same ids. The full table, its spreads and its caveats are in
[`benchmarks.md`](benchmarks.md).

| cell | hipml | llama.cpp | Δ |
| --- | ---: | ---: | ---: |
| prefill @512 tok/s | 1,346.1 | 1,215.2 | +10.8% |
| prefill @8,192 tok/s | 1,326.1 | 1,352.5 | −1.9% |
| decode @0 tok/s | 33.15 | 32.24 | +2.8% |
| decode @8,192 tok/s | 31.98 | 31.41 | +1.8% |

### The Q5_K tile's fallback comparison (earlier tree)

These rows were measured on an earlier tree, before the re-verification, and are
not re-run here. They are kept because the Q5_K dequant-in-shared tile's whole
justification is the comparison against the batched-GEMV fallback, not the
current headline; the current numbers are the table above.

```
                          prefill tok/s            decode tok/s
depth 0 (--prefill 512)     1,186.27 (512 tok)       29.20
depth 8,192                 1,207.27 (8,192 tok)     28.43

with HIPML_EXPAND_SMALL_QUANTS=1 (the old F32 expansion):
depth 0 (--prefill 512)     1,059.21                 27.58
depth 8,192                 1,086.31                 26.93

with HIPML_Q5K_COPY_GB=0 (no Q5_K tile kernel, batched GEMV instead):
depth 0                       289.76                  27.62
depth 8,192                   297.68                  26.92
```

`HIPML_Q5K_COPY_GB=0` still selects the old batched-GEMV route, so the
comparison stays reproducible. The variable, and why an fp16 Q5_K copy could
never have been resident on this file, are in
[`notes/serve-vram.md`](notes/serve-vram.md).

Serving Q3_K and IQ3_S in block form is worth 1.19 GiB of residency and a few
percent on both axes: the four tensors stream 9.31× fewer bytes per token, and
the decode GEMV reads 110-byte blocks at 237–263 GB/s against 390 GB/s for the
Q4_K-sized `gemv_iq4_xs`. The prefill tile is at parity with the other K-quants
(95.6 TFLOPS for Q3_K, 89.5 for IQ3_S, 95.5 for Q6_K at 17408x5120, 2,048
tokens).

The Q5_K dequant-in-shared tile kernel is the whole prefill difference on this
file: 3.0× / 3.7× over the batched-GEMV fallback, with the same last id at both
depths (220 and 292).

The stock `Qwen3.8-27B-Q4_K_M.gguf` passes the same gate — `PASS 20/20`,
`max |gpu - cpu| = 0.1481 at 34114`, the published value. Its speed was measured
on the earlier release tree; the current head-to-head is the UD table above.
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

## llama.cpp's settings

llama.cpp is build 11820, commit `d7725699`. It ran with

```
-ngl 99 -c <ctx> --parallel <slots> -ctk q8_0 -ctv q8_0
--no-webui --no-cache-idle-slots
```

and with no `-fa`, `-b` or `-ub` flag, so the build defaults applied (`-fa
auto`, which resolves to on, `-b 2048`, `-ub 512`). Its cells were measured in
the same session and the same window as hipml's, on the same token ids; the
numbers are in [`benchmarks.md`](benchmarks.md) and are not repeated here.

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

## Checks

Re-run on `409f4ad`, the tree the table above was timed on, before any cell:

| check | result |
| --- | --- |
| `hipml probe` | `built with hip feature: true` |
| greedy ids, "The capital of France is" (UD) | 12/12 identical |
| `hipml parity --deep` | 74 passed, 0 failed, 1 skipped (`sample_top_p_f32`) |
| `hipml parity prefill --deep` | 15 passed, 0 failed, 0 skipped |
| `hipml oracle-gate` (UD-Q4_K_M, 20 steps, chunk 2048, q8) | PASS, 20/20, max abs 0.1086 at 101748 |
| `hipml oracle-gate` (stock Q4_K_M, 20 steps) | PASS, 20/20, max abs 0.1481 at 34114 |
| `hipml batch-bench --slots 16 --check` (UD, q8) | 16/16 ids on all 16 slots, "all slots reproduce the B=1 path" |
| `hipml spec-bench` drafts 0–3, q8 | 128/128 exact on every row |
| `hipml spec-bench` drafts 0–3, f32 | 128/128 exact on every row |

The same binary is the one timed. Both oracle gates reproduce the reference
max-error to four decimal places and the same index on this tree, so the
prefill path is numerically unchanged.

Load, tokenizer and template checks, from the tree where UD support landed:

| check | result |
| --- | --- |
| `cargo test --workspace` | local and gpu-host: all pass (0 failed) |
| `cargo check --workspace --features hip` | clean |
| `hipml oracle-gate` (fine-tune, 169 ids, chunk 2048) | PASS, 20/20 ids identical |
| Fine-tune generation after the config change | 64 layers, 8 tokens emitted |
| Q4_K_XL load checks | 866/866 byte-identical, embeddings 5120/5120 exact |
| UD-Q4_K_M load checks | 866/866 byte-identical, embeddings 5120/5120 exact |
| tokenizer, 20 strings × 2 modes | 40/40 ids identical |
| template, 3 single-turn cases | text and ids identical |

The fine-tune still reports 64 layers and generates, so subtracting
`nextn_predict_layers` did not change the fine-tune path.

## Unverified

- 128-token greedy parity and logit margins on a stock file: the 20-step
  oracle gate above is the only published logit comparison.
- `UD-Q6_K_XL` end to end; it needs no new format, but it was not run here.
- MTP over a batch, and `cache-gate` on a UD file. Batched decode past 8,192
  tokens per slot and the prefix cache are now measured on UD (cells b and c),
  but the cache gate itself was not re-run there.
- Chat-template branches specific to the stock template: developer-role
  messages, merged multi-system turns (one reproducer above), and the stricter
  tool-argument mapping rule. The committed template fixture passes, but these
  branches are not in it.
