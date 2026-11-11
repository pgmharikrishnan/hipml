# Native Qwen3.8-27B (stock Q4_K_M, Unsloth UD variants)

`feat/native-qwen`, base `6a1207f` (`feat/release`). Every earlier hipml
measurement used a private fine-tune with the same geometry. This package
points the engine at the **stock** model family and removes the fine-tune
constants that made it work only there. It records why the Unsloth dynamic
quantisations do not run, and what the release does instead: it runs
`Qwen3.8-27B-Q4_K_M.gguf`, the file
[`release-run.md`](release-run.md) measured, which contains no IQ or Q3_K type.

| file | bytes | tensors |
| --- | ---: | ---: |
| `Qwen3.8-27B-Q4_K_M.gguf` (the release model) | 16,810,714,336 (15.65 GiB) | 866 |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 17,559,178,144 (16.35 GiB) | 866 |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25,299,061,664 (23.56 GiB) | 866 |
| private fine-tune (`<model>.gguf`) | 17,192,421,376 (16.01 GiB) | 851 |

Both stock files carry the MTP (`nextn`) block **inline** as `blk.64.*`;
`block_count` is 65 and `nextn_predict_layers` is 1.

## Bottom line

- The Q4_K_XL file **loads completely** into VRAM (all 866 tensors, 16.34 GiB,
  byte-identical readback, embedding and lm_head checks pass) but **cannot
  generate**: its Unsloth dynamic quant mixes in four GGML types with no
  dequant path (`IQ4_XS`, `IQ4_NL`, `Q3_K`, `IQ3_S`, 80 tensors / 3.31 GiB).
  The first failure is `generate: graph: type_23 has no GEMV kernel`.
- The Q6_K_XL file avoids every IQ type, loads 23.55 GiB, and stops at its
  Q6_K embedding (`token_embd.weight: expected Q4_K, found Q6_K`). Its next
  blocker is three Q5_K attention projections.
- Implementing IQ4_XS is out of scope: v0.1.0 adds no new quant format.
  Q6_K_XL needs no new *format* — it needs a Q6_K embedding-row kernel and a
  Q5_K attention path — but neither was added here. See "If this is unblocked".

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

`Qwen3.8-27B-UD-Q4_K_XL.gguf`:

| type | tensors | bytes | GiB | engine |
| --- | ---: | ---: | ---: | --- |
| Q5_K | 191 | 7,926,251,520 | 7.38 | supported |
| IQ4_XS (23) | 70 | 3,125,084,160 | 2.91 | **no dequant** |
| Q4_K | 69 | 3,059,712,000 | 2.85 | supported |
| Q6_K | 56 | 2,862,182,400 | 2.67 | supported |
| IQ4_NL (20) | 6 | 280,166,400 | 0.26 | **no dequant** |
| Q8_0 | 110 | 130,908,160 | 0.12 | supported |
| Q3_K (11) | 3 | 114,892,800 | 0.11 | **no dequant** |
| IQ3_S (21) | 1 | 38,297,600 | 0.04 | **no dequant** |
| F32 | 360 | 10,686,464 | 0.01 | supported |
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

So Q6_K_XL **does** avoid every IQ/Q3 type; its only Q5_K tensors are 27
mixtures, and Q5_K already has decode, batched and prefill kernels in this
tree. Q4_K_XL cannot be made to run without IQ4_XS (2.91 GiB), which is
explicitly out of scope.

## Load result

```
Q4_K_XL: 866 tensors  17,548,181,504 B (16.34 GiB)  read+upload 10.42 s (1.68 GB/s)
         hipMemGetInfo free 34,023,604,224 -> 16,448,159,744 B
         resident delta 17,575,444,480 B (16.37 GiB)
         4/4 round-trip tensors byte-identical
         embedding rows 0/1/1234/100000/248319: 5120/5120 exact, max_abs 0
         lm_head Q6_K 248,320 logits in 2.419 ms (431.2 GB/s)
Q6_K_XL: 866 tensors  25,288,065,024 B (23.55 GiB)  read+upload 16.55 s (1.53 GB/s)
         resident delta 25,263,079,424 B (23.53 GiB); 8.76 GB free after
```

## Generation blocker

```
$ hipml generate --device --gguf <HIPML_GGUF> \
    --ids 1,2,3 --tokens 2 --ctx 64 --kv q8
prompt ids: [1, 2, 3]
resident: 16.34 GiB in 866 tensors on AMD Radeon AI PRO R9700 (upload 5.9s)
prefill:    chunked, 2048-token tiles
generate: graph: type_23 has no GEMV kernel
```

`type_23` is IQ4_XS; the first one the layer loop reaches is
`blk.0.ffn_down.weight`. The full unsupported set is 80 tensors / 3.31 GiB:
IQ4_XS 70, IQ4_NL 6, Q3_K 3, IQ3_S 1.

`Qwen3.8-27B-UD-Q6_K_XL.gguf` gets one step further and fails at
`token_embd.weight: expected Q4_K, found Q6_K` (the only embedding-row kernel
is `embed_row_q4_k_dev`). Behind that, three attention projections are Q5_K
(`blk.3.attn_v`, `blk.7.attn_k`, `blk.11.attn_output`) and `AttnLayer` has no
Q5_K GEMV/prefill entry.

### If this is unblocked

- Q4_K_XL: requires IQ4_XS/IQ4_NL/Q3_K/IQ3_S dequant + GEMV. Ruled out.
- Q6_K_XL: requires (a) a Q6_K embedding-row dequant kernel (the format is
  already supported everywhere else), (b) Q5_K entries in `AttnLayer`
  (decode entry `gemv_q5_k`; prefill `prefill_q5_k`, or `prefill_q5_k_w2h`
  on the WMMA path). Until then it loads but does not generate.

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

## Native-model speed (llama.cpp only; hipml is blocked)

Same fixture and harness as `bench/` (`job.py`, 3 cold reps, 33 generated
tokens, q8/q8_0 KV, no cache reuse), **stock Q4_K_XL**, `prose_explanation`:

| depth | prompt | prefill tok/s (median) | decode tok/s (median) | TTFT (median) | peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 373 | 1,006.6 | 30.20 | 371 ms | 16,485 MiB |
| 8,192 | 8,192 | 1,284.7 | 29.50 | 6,389 ms | 16,760 MiB |

Reps: prefill 870.4 / 1,006.6 / 1,029.3 and 1,284.7 / 1,287.7 / 1,279.2;
decode 29.71 / 30.21 / 30.20 and 29.44 / 29.57 / 29.50.
llama.cpp build `11820`, commit `d7725699`, `-ngl 99`.

There are **no hipml prefill/decode numbers for the stock model**: it does not
reach a token. The 128-token greedy comparison and the depth-0/8K timing table
cannot be produced until a stock file runs, and no such number is published
here.

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

# the blocker
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
| Fine-tune generation after the config change | 64 layers, 8 tokens emitted |
| Q4_K_XL load checks | 4/4 byte-identical, embeddings 5120/5120 exact |
| tokenizer, 20 strings × 2 modes | 40/40 ids identical |
| template, 3 single-turn cases | text and ids identical |

The fine-tune still reports 64 layers and generates, so subtracting
`nextn_predict_layers` did not change the fine-tune path.

## Unverified

- hipml prefill/decode, 128-token greedy parity and logit margins on any stock
  file: blocked above.
- Q6_K_XL generation: two named blockers, not exercised past the embedding.
- Chat-template branches specific to the stock template: developer-role
  messages, merged multi-system turns (one reproducer above), and the stricter
  tool-argument mapping rule. The committed template fixture passes, but these
  branches are not in it.
