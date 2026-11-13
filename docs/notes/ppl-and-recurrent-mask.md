# The recurrent mask's MTP entry, and a perplexity the engine can call its own

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

Two prerequisites for the multi-model table. Neither touches the forward pass,
the kernels, the tile or the MTP path: one is a config-validation rule, the
other a read-only command.

1. **Two models could not load.** `Swift-1.5-Qwen3.8-27B` and
   `OrcaCyber-Nano-1.0` declare a 65-entry `attention.recurrent_layers` mask for
   a 64-layer decoder plus one inline MTP block, and `Config::from_gguf`
   rejected it. Truncating the MTP entry fixes it; the rule that decides when to
   truncate is the interesting part.
2. **The engine had no quality number.** `hipml logits` applies `lm_head` to a
   caller-supplied 5,120-float vector: one position, no ids, no text. This adds
   `hipml ppl`, which prefills a corpus in windows and scores every true next
   token, and it reproduces `llama-perplexity` to **0.23%** on Arete and
   **0.41%** on the stock file.

## 1. The mask

### 1.1 What the files carry

All three files are `qwen35`, dense, hidden 5,120, 24 heads / 4 KV, head_dim
256, vocab 248,320, 16 full-attention layers at `[3, 7, ... 63]`.

| file | `block_count` | `nextn_predict_layers` | declared mask | decoder |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3.8-27B-Q4_K_M` (stock) | 64 | absent | absent (interval fallback) | 64 |
| `Qwen3.8-27B-UD-Q4_K_M` | 64 | absent | absent (interval fallback) | 64 |
| `Arete-27B-Q4_K_M` | 64 | absent | 64 entries | 64 |
| `Swift-1.5-Qwen3.8-27B-Q4_K_M` | 65 | 1 | **65 entries** | 64 |
| `OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G` | 65 | 1 | **65 entries** | 64 |

`Config::from_gguf` subtracts the MTP block first (`n_layer = 65 - 1 = 64`) and
then compared the declared mask against `n_layer`, so the two 65-entry files
failed with `config: recurrent layer mask length != block_count`. The mask was
right; the check was comparing it to the wrong number.

### 1.2 The rule

`trim_mtp_entry` accepts exactly two lengths and rejects everything else:

```rust
fn trim_mtp_entry(mut mask: Vec<bool>, n_layer: usize, nextn: usize) -> Option<Vec<bool>> {
    if nextn > 0 && mask.len() == n_layer + nextn {
        mask.truncate(n_layer);
    }
    (mask.len() == n_layer).then_some(mask)
}
```

* `len == n_layer`: the mask is a decoder mask, keep it.
* `len == n_layer + nextn` **and** `nextn > 0`: it covers the inline MTP block
  too, drop that block's entry.
* anything else: `None`, and the caller keeps the existing
  `recurrent layer mask length != block_count` error.

This is not "ignore the length". A 63-entry mask on a 64-layer file, a 66-entry
mask, or a 65-entry mask on a file with no MTP block are all still refused. The
three unit tests in `config.rs` pin all three cases, including that a 65-entry
mask on a 64-layer file with `nextn = 0` is *not* truncated.

### 1.3 What it produced

`hipml load` prints the resolved mask, so the truncation is checked against the
interval fallback that the files without a mask take:

```
Swift  config (qwen35): 65 blocks (64 decoder + 1 nextn), ... ctx 262144
       full-attention layers (16): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
Orca   config (qwen35): 65 blocks (64 decoder + 1 nextn), ... ctx 262144
       full-attention layers (16): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
Arete  config (qwen35): 64 blocks (64 decoder + 0 nextn), ... ctx 262144
       full-attention layers (16): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
```

The truncated 65-entry mask and Arete's declared 64-entry mask give the same 16
indices, and so does the interval fallback (`GGG A` repeating, `(i + 1) % 4`).
That is the check that the entry dropped was the MTP block's and not a decoder
layer's.

### 1.4 Loading is not output

Greedy 12 ids on `bench/fixtures/p2_math_169.ids`, same file, same ids, through
`llama-server` (build 11820, `d7725699`, `temperature 0`, `return_tokens`) and
through `hipml generate --device`:

| file | llama.cpp | hipml |
| --- | --- | --- |
| Swift-1.5-Qwen3.8-27B | `248068 198 1596 1144 4087 1156 25 12870 9885 61695 1228 440` | identical |
| OrcaCyber-Nano-1.0 | `248068 198 1596 1144 4087 1156 579 17754 9885 61695 1228 440` | identical |

12/12 on both, and the two files legitimately differ at id 6 (`25` vs `579`:
"answer user:" against "answer user's"), which is what a different fine-tune
should do. The tokenizer agreement behind the comparison is in §3.1.

## 2. `hipml ppl`

### 2.1 Why `logits` could not do it

`hipml logits` takes a 5,120-value f32 file and runs `lm_head` on it. There is no
token, no position, no sequence and no tokenizer in that path, so it cannot
produce a next-token likelihood at all, let alone a mean over a corpus.

### 2.2 The convention, read off `llama-perplexity`'s source

`tools/perplexity/perplexity.cpp` at `d7725699`, the `ppl_stride == 0` path
(the default):

* `common_tokenize(ctx, prompt, true)`; `add_bos` is `llama_vocab_get_add_bos`,
  which is false for Qwen3.8, so no BOS is prepended. Same on our side.
* `n_chunk_max = tokens.size() / n_ctx` — **whole windows only; the tail shorter
  than a window is never scored.**
* `const int first = n_ctx / 2;` — **only the second half of each window is
  scored**, so every scored position has at least 256 tokens of context.
* `count += n_ctx - first - 1` — 255 positions per 512-token window: the last
  position of a window has no next token inside that window.
* the NLL is `-(logit[t] - max - log(Σ exp(logit - max)))` with f32 logits, f32
  `expf`, and the sum in f64.

`hipml ppl` reproduces all of it, prints llama's per-window running estimate
(`[1]2.3723,`) and llama's final line (`Final estimate: PPL = X +/- Y`), and
prints the token count and the scored-position count so the convention is
visible in the log rather than assumed.

It does not tokenize: `--ids-file` takes the ids the other commands read. That is
deliberate — the ids that reach the model are the ids the caller measured.

### 2.3 The 61-versus-62 token question

Recon saw llama report 61 tokens where both tokenizers say 62. The cause is in
`common/arg.cpp`, in the `-f/--file` handler:

```cpp
params.prompt = read_file(value);
params.prompt_file = value;
if (!params.prompt.empty() && params.prompt.back() == '\n') {
    params.prompt.pop_back();          // <-- the trailing newline
}
```

`llama-perplexity -f` tokenizes the file **minus its final newline**. A 62-token
text file that ends in `\n` therefore reaches the model as 61 ids. It is not the
scoring window and it is not the tokenizer: `llama-tokenize -f` keeps that
newline (its own reader says so in a comment) and returns 62.

Measured on the corpus this note uses, `calib_longctx.txt` (3,006,539
B, 95,254 lines):

| tool | ids |
| --- | ---: |
| `hipml tokenize --text-file` | 849,455 |
| `llama-tokenize -f --no-escape` | 849,455, **0 differing** |
| `llama-tokenize -f` (escape processing on) | 848,698 |
| `llama-perplexity -f` | 849,454 (the above minus the final `198`) |

**Decision: `hipml ppl` does not drop anything.** It scores exactly the ids it
is given, because its input is ids and silently deleting one would be a hidden
edit to the corpus. The count is printed so the difference is visible, and the
comparison below feeds both engines the *same* ids — llama's own, taken from its
`--save-all-logits` dump — so the two numbers are computed over an identical
token stream by construction.

The `848,698` row is a separate llama quirk worth knowing before anyone diffs
token streams: `llama-tokenize` applies `string_process_escapes` by default in
this build, so the literal two-character `\n` inside source code becomes a real
newline (757 such sequences in this corpus, and exactly the 757-token
difference). `llama-perplexity` does not do this: a file containing `alpha \n
beta` and one containing `alpha <LF> beta` produce different ids through
`--save-all-logits` (`1088, 77` against `695`), so the reference engine reads
the file's bytes.

### 2.4 How it runs

* one `prefill_chunked_capture` per window: the production chunked-prefill path,
  so a 512-token corpus is **one** invocation, not 511, and the capture buffer
  already holds every position's post-`output_norm` row;
* one `lm_head` GEMV per scored position on that row, the same kernel and the
  same entry the decode loop uses, so the numbers come from the shipped path;
* `engine.reset_state()` per window. **This was a bug the acceptance test
  caught**: `GpuState` is only a position, and the 48 gated-delta-net layers'
  carried state lives in the engine. Without the reset, window 1 matched llama
  and windows 2-4 drifted to `[1]2.3843 [2]2.1048 [3]2.4042 [4]2.5749` against
  llama's `2.3723 2.0250 2.2250 2.1791` — a +18% aggregate. A single-window
  check would have passed. That is the argument for running the comparison over
  more than one window.

Cost: ~1 s per 512-token window on the R9700 (prefill 0.4 s, 255 `lm_head`
GEMVs at ~1.7 ms each because the 248,320-row head is 1.04 GiB of Q6_K), against
0.6 s for llama's batched head.

## 3. The acceptance test

### 3.1 Same file, same ids

`llama-perplexity --save-all-logits` writes the ids it tokenized ahead of the
per-position log-probs, so the comparison can use exactly those. Both engines
then see the same 512-token windows.

### 3.2 PPL

`-c 512 -b 512 -ub 512 --ppl-stride 0`, corpus `calib_longctx.txt`, four
windows (2,048 tokens, 1,020 scored positions). llama's two attention paths are
both listed because llama moves between them by more than the gap to us:

| file | hipml `ppl` (f32 KV) | llama default (`-fa auto`, f16 KV) | Δ | llama `-fa off`, f32 KV | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arete-27B-Q4_K_M | **2.1824** | 2.1791 | **+0.15%** | 2.1874 | **−0.23%** |
| Qwen3.8-27B-Q4_K_M (stock) | **2.2210** | 2.2270 | **−0.27%** | 2.2301 | **−0.41%** |

One window, the run recon measured:

| file | hipml | llama default | Δ | llama `-fa off` f32 KV | Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arete, window 1 | 2.3843 | 2.3723 | +0.51% | 2.3860 | −0.07% |

Per window, Arete: hipml `2.3843 2.0300 2.2261 2.1824`, llama `2.3723 2.0250
2.2250 2.1791`.

**Tolerance claimed: 1%, achieved 0.41% worst case; 0.23% on Arete and 0.41% on
the stock file against the like-for-like non-flash-attention path, and 0.15% /
0.27% against llama's default.**

Why it is not bit-identical, and why 0.4% is the floor rather than a bug:

* the prefill projections use the tiled fp16 GEMM, whose operands are fp16 and
  whose k-accumulation is 32 values at a time; `oracle-gate` already measures
  the whole-model consequence of that as `max |gpu - cpu| = 0.1086` on logits of
  order 10, i.e. ~1e-3 relative. A 1e-3 perturbation of a 248,320-way softmax is
  worth a few 1e-3 nats, and 0.4% of PPL is 0.004 nats;
* llama's own two attention paths differ from each other by **0.38%** (Arete,
  4 windows) and **0.55%** (stock, window 1). Our residual is smaller than the
  spread llama shows against itself, which is the honest scale for "two correct
  engines, different kernels";
* the direction is not systematic: we are +0.15% on Arete and −0.27% on stock
  against llama's default, so it is accumulation order, not a missing term.

Had this been 3% or 18%, one of the two would have been wrong. It was, once: see
the `reset_state` bug in §2.4, which the four-window run exposed and the
single-window run did not.

## 4. Gates

Every row below was measured on the binary this branch ships (SHA-256
`8F65FE83…77FC6B`), built from the final tree with `--features hip` on the
R9700, binary asserted `built with hip feature: true`, UD file asserted at
16,564,259,744 B and SHA-256 `1BCE504D…7DC529`, `HIPML_MTP_GGUF` set explicitly
where a drafter is wanted, every model passed with `--gguf`.

| gate | expected | observed |
| --- | --- | --- |
| `parity --deep` | 74 passed | 74 passed, 0 failed, 1 skipped (`sample_top_p_f32`) |
| `parity prefill --deep` | 15 passed | 15 passed, 0 failed |
| `oracle-gate` UD, 20 steps, chunk 2048, q8 | PASS 20/20, 0.1086 at 101748 | PASS 20/20, **0.1086 at 101748** |
| `oracle-gate` stock, 20 steps | PASS 20/20, 0.1481 at 34114 | PASS 20/20, **0.1481 at 34114** |
| greedy 12 ids, UD | `11751 13 198 760 6511 314 9564 369 19241 13 198 760` | identical |
| `batch-bench --slots 16 --check` | 16/16 on every slot | 16 slots, each 16/16 ids |
| `spec-bench --drafts 3 --kv q8` | 128/128 exact every row | 33.33 → 51.95 / 60.78 / 63.78 tok/s, 128/128 on all four rows |
| `spec-bench --drafts 3 --kv f32` | 128/128 exact every row | 32.70 → 50.96 / 59.64 / 62.35 tok/s, 128/128 on all four rows |
| `column-bench --check` | 0 differ | 0 differ on logits (7,946,240 values), hidden and residual |
| `probe_exact.py --ctx 32768` | PASS | PASS, 48 replies id-for-id across drafts 1,2,3 |
| `cargo test --workspace` | clean | clean, including the three new `config` tests |
| `cargo fmt --all -- --check` | clean | clean |
| `scripts/check-public-tree.sh` | clean | clean |

The two oracle maxima are the same numbers every slice since §6 of the
prefill-tile note reports, which is the expected result for a change that does
not touch the forward pass.

### 4.1 The cells that must not regress

| cell | reference | measured |
| --- | ---: | ---: |
| prefill @512 | 1,353.8 tok/s | 1,337.49 / 1,316.04 / 1,302.71 |
| prefill @8,192 | 1,328.0 tok/s | 1,330.54 / 1,321.97 / 1,314.74 |
| decode @0 | 33.22 tok/s | 33.07 (spec-bench plain row 33.33) |
| decode @8,192 | 31.99 tok/s | 31.99 |
| serve a, 1 slot | 31.64 tok/s | 31.77 |
| serve b, 8 slots | 131.96 tok/s | 132.58 |
| serve c, 16 slots | 222.42 tok/s | 222.57 |
| MTP serve a, 3 drafts | 52.36 tok/s | **52.48**, peak 21,071.5 MiB against the reference's 21,072 |

The prefill rows carry the widest spread, so they were re-measured interleaved
against the base tree (`a6afea2`, built in a second directory, alternating
base/new within one hold, three rounds):

| round | @512 base | @512 this branch | @8,192 base | @8,192 this branch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,338.07 | 1,337.49 | 1,340.06 | 1,330.54 |
| 2 | 1,313.50 | 1,316.04 | 1,327.96 | 1,321.97 |
| 3 | 1,299.64 | 1,302.71 | 1,321.56 | 1,314.74 |

Both trees drift down across the hold (1,338 → 1,300 at 512) and the second run
of each pair reads slightly below the first: the pairs are −0.04%, +0.19%,
+0.24% at 512 and −0.71%, −0.45%, −0.52% at 8,192, i.e. the 512 pairs bracket
zero and the 8,192 pairs carry the same consistent −0.5% that the previous hold
showed (−0.69%, −0.31%, −0.37%). The prefill path is untouched by this branch
(no change in `gpu/prefill.rs`, the kernels, the tile or the MTP path), and the
shipped binary's standalone reading at 512 was 1,307.94, inside this hold's own
1,299–1,338 spread. The interleaved table is the number that compares code; the
reference 1,353.8 is a reading from outside this hold.

One trap worth recording, because it produced a wrong number before it was
caught: `serve_bench.py` loads the MTP drafter from `HIPML_MTP_GGUF` when the
environment sets it, even with no `--mtp` flag, so the first serve-a run read
48.11 tok/s (a speculative-decode cell) against a plain reference of 31.64. The
plain cells must be run with `HIPML_MTP_GGUF` removed from the process
environment. Clearing it with `set NAME= && …` in `cmd` does not remove it: the
space before `&&` becomes the value, so the server is handed `--mtp " "` and
dies in `load_sidecar`.

## 5. What is unverified

* The PPL agreement is measured on one corpus, four windows, two files. It says
  the engine's likelihoods are within 0.5% of llama's on English prose and code;
  it does not bound the agreement on another distribution.
* The `-fa auto` path on this build is llama's flash-attention kernel; whether
  the 0.38-0.55% spread between llama's own paths is the kernel or the f16 KV
  cache it pairs with was not separated (`-ctk f32` alone moved PPL by 0.00%
  with FA on and 0.12% with FA off, so the KV dtype is not the cause).
* `hipml ppl` scores with f32 KV. llama's default is f16. The table lists both,
  but only the f32-vs-f32 column is like-for-like.
* The corpus tail (849,454 mod 512 = 446 tokens) is not scored, by llama's
  convention. A different `--ctx` would score a different tail.
* The 757-token `llama-tokenize` escape difference is characterised by one
  example (`\n` in source code) and the count on this corpus; no other escape
  sequence was tested.
