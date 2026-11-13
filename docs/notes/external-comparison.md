# EXT: two published claims on this model, recorded as published

Base `409f4ad` (origin/main). Box: R9700 (gfx1201, 64 CUs, 32 GB, 640 GB/s
spec), `Qwen3.8-27B-UD-Q4_K_M.gguf`, q8 KV.

This note runs no benchmark of its own. It records two other projects'
published numbers beside ours, with their provenance, and states what separates
them from ours. Every external figure below is **vendor-published and not
independently reproduced**; nothing in this note was measured on their hardware,
and none of their figures was measured on ours.

## 0. Question and answer

Question: two external projects publish numbers on Qwen 3.8 27B and on a 125B
Qwen MoE. What do they claim, what do we measure, and what separates the two?

Answer: **their decode advantage is their format's size, not their engine.**
Normalised by bandwidth, their engine and ours run at the same efficiency per
byte — **80% for them, 81% for us** (§5). Their `ROCmFP4_FAST` file moves 14.9%
fewer bytes than their stock `Q4_K_M` file and they measure 13.5% more decode
from it: near, but short of, the +17.5% a pure byte reduction would predict at
equal efficiency. The two things they
have that we do not are **prompt-lookup drafting** and **confidence-gated draft
depth**; both claims are unverified on this card (§7).

Nothing here ranks the engines. The numbers were produced on different silicon,
different formats, different backends and different harnesses, and §6 lists why.

## 1. Provenance

| | project | licence | hardware | engine and backend | date of the claim | link |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `julianmb/halofpx` | Apache-2.0 | AMD Ryzen AI Max+ 395, Radeon 8060S, `gfx1151`, 40 CU @ 2.9 GHz, 128 GB 256-bit LPDDR5X, Linux 7.0, Mesa 26.0 RADV Wave64 | wraps `charlie12345/ROCmFPX` (pinned `e87d53e (213)`); decode on Vulkan/RADV Wave64, prefill on ROCm/HIP | README "Verified Date" for the 27B row: **2026-08-25**; repository last pushed 2026-09-24 | <https://github.com/julianmb/halofpx> |
| 2 | `Niko1221/Strata` | MIT | MoE-offload engine; its own tables span NVIDIA (CUDA) and AMD (HIP) cards, including a Ryzen AI Max+ 395 and an R9700 | its own engine on ggml; CUDA and HIP | docs read **2026-11-13**; repository last pushed 2026-11-12 | <https://github.com/Niko1221/Strata> |
| — | hipml (ours) | Apache-2.0 | Radeon AI PRO R9700, `gfx1201`, 64 CU, 32 GB, 640 GB/s spec | HIP | measured **2026-11-13**, one session, one binary, `409f4ad` and after | this repository |

Licence notes: halofpx ships a short-form Apache-2.0 `LICENSE` (the notice, not
the full text), so GitHub's licence detector reports `NOASSERTION`; the file
itself is Apache-2.0 and the README says so. Strata is MIT, with `third_party/ggml`
under MIT and the model files under their own licences.

Quoted from halofpx, the hardware line above its benchmark table:

> Measurements taken directly on **AMD Ryzen AI Max+ 395 (40 CU Radeon 8060S @
> 2.9 GHz, 128 GB 256-bit LPDDR5X, Linux 7.0, Mesa 26.0 RADV Wave64 / Vulkan0)**

## 2. Source 1 — `julianmb/halofpx`, the 27B numbers

**Vendor-published, not independently reproduced.** From the README's "Measured
Speed Increase Over Standard GGUF" table, measured "with identical prompts":

| model | stock `Q4_K_M` | `ROCmFP4` / `ROCmFP4_FAST` | claimed speedup | claimed size saving |
| --- | --- | --- | --- | --- |
| **Qwen 3.8 27B** | 15.92 GiB — **12.35 tok/s** | 13.55 GiB — **14.02 tok/s** | **+13.5%** | **−14.9%** |
| **Ornith 1.5 35B-A3B** | 21.80 GiB — **71.5–71.7 tok/s** | 18.16 GiB — **76.9 tok/s** | **+7.5%** | **−16.7%** |

The same 27B row in their model-zoo table, with the columns that table adds:

| field | published value |
| --- | --- |
| quant and size | `ROCmFP4_FAST`, 13.5 GiB (the format-comparison table above gives 13.55 GiB; the zoo table rounds it) |
| minimum VRAM | 16 GB |
| prefill `pp512` | **385.65 tok/s** |
| decode, bare | **14.02 tok/s** |
| decode, with MTP | **33.80 tok/s** |
| verified date | `2026-08-25` |

Their MTP sentence, in full:

> **Combined with MTP speculative decoding** (Qwen 3.8 27B, `n4/p0.0`): **33.8
> tok/s sustained = 2.40× over stock baseline** (12.35 tok/s).

Their mechanism claim, in full:

> ROCmFP4 quant blocks map directly to RDNA 3.5 cooperative-matrix
> (`KHR_coopmat`) operands — faster prompt evaluation at equal context, without
> the multi-scale dequantization overhead of `Q4_K` blocks.

Their two backends, from the same README:

> **Vulkan0 (Mesa RADV Wave64):** Fastest token decode and MTP speculative tree
> verification (**up to 36 tok/s** on 27B).
> **ROCm0 (HIP):** High-throughput prompt evaluation / prefill processing (**up
> to 390+ tok/s**).

### 2a. Arithmetic on their published figures

Three checks, done on their numbers and nothing else:

- `14.02 / 12.35 = 1.135` and `13.55 / 15.92 = 0.851`. The two headline deltas
  are consistent with their own table.
- `76.9 / 71.5 = 1.075` and `18.16 / 21.80 = 0.833`. The Ornith row is
  consistent too.
- **The 2.40× does not resolve against the number in its own parentheses.**
  `33.80 / 12.35 = 2.74`, not 2.40; `33.80 / 14.02 = 2.41`. So the 2.40× is
  against their own `ROCmFP4_FAST` bare decode, not against the stock
  `Q4_K_M` baseline the sentence names. We quote the figure as published and
  record the ambiguity rather than pick a reading.

## 3. Source 2 — `Niko1221/Strata`, technique only

**Vendor-published, not independently reproduced.** Strata is a MoE-offload
engine for Qwen3.8-Flash-Next, a **125B MoE** that streams experts from RAM and
SSD; its absolute numbers are therefore not comparable to a dense 27B and are
not recorded here. Only what transfers as a technique is, from
`docs/HOW_IT_WORKS.md` and `docs/DETAILS.md`.

| technique | their published claim | where |
| --- | --- | --- |
| MTP drafting | "the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them, 2.4-3.2 tokens per pass on average"; "you get the same quality answer, 1.6-1.8x sooner" | `HOW_IT_WORKS.md` |
| prompt lookup | "prompt lookup drafts up to 5 tokens from the earlier copy, but only where its measured acceptance and cost say it pays: code edits 6-11% faster, other text unchanged" | `HOW_IT_WORKS.md` |
| draft depth and gate | "`--spec N`, `--spec-min-p P` — Draft depth and the confidence needed to add a guess. Setup's default is `--spec 4 --spec-min-p 0.5`." | `DETAILS.md` |
| draft vocabulary subset | "`--draft-vocab en` … `en` keeps the English and code subset (~110 MiB less VRAM, English 1-2% faster, CJK gets almost no drafts)" | `DETAILS.md` |
| KV, default | "`--kv int8` (default) — 8-bit KV cache." | `DETAILS.md` |
| KV, 4-bit | "`--kv q4_0` — Half the KV memory with a Hadamard rotation before 4-bit rounding. About 4% faster at 128K, measurably less precise on long documents (perplexity +8-12%; needle tests still pass)." | `DETAILS.md` |
| KV, mixed | "`--kv k8v4` — Keys at 8 bits, values as rotated 4-bit: 23% less KV memory than 8-bit … Prompts are 2-5% slower." | `DETAILS.md` |
| KV streaming | "Keeps the context's KV cache in RAM and only the part attention reads in VRAM … Setup turns it on from 64K when the RAM has room" | `DETAILS.md` |
| acceptance reporting | "`STRATA_SPEC_DEPTH=1` prints acceptance by draft depth" | `DETAILS.md` |

They also publish a verify profile and per-depth acceptance tables, which is the
reporting shape that makes the confidence gate auditable rather than asserted.

## 4. Our numbers

**Measured on R9700, one session, one binary**, `409f4ad` and after. The
head-to-head is the same fixture-token harness as
[`../release-run.md`](../release-run.md): prompt ids tokenized once and fed to
both engines, same file, same KV class (hipml `q8`, llama.cpp `-ctk q8_0 -ctv
q8_0`), cold reps. Serve cells are `bench/serve/serve_bench.py`; the MTP row is
`hipml spec-bench --drafts 3`; the stream row is `hipml stream-shape` and
`hipml gemv-entries`.

| axis | hipml | llama.cpp (same file, same harness) |
| --- | ---: | ---: |
| prefill @512, tok/s | 1,349.6 | 1,209.8 |
| prefill @8,192, tok/s | 1,335.1 | 1,353.9 |
| decode @0, tok/s | 33.31 | 32.20 |
| decode @8,192, tok/s | 32.04 | 31.44 |
| serve, 1 slot, tok/s | 30.98 | 32.17 |
| serve, 8 slots, tok/s | 130.0 | 107.6 |
| serve, 16 slots, tok/s | 219.4 | 220.0 |
| MTP drafts 3, serve cell a, tok/s | 51.8 | 62.3 |
| weight stream | 15.48 GB/token at 96.6% of a pure read | — |

The last row is the one that makes the rest interpretable: the decode stream is
15.48 GB of weights per token, and the routed GEMV moves them at **96.6% of what
a pure read of the same bytes at the same shapes achieves** (26.98 vs 26.08 ms,
[`decode-roofline.md`](decode-roofline.md) §2). The read is not where our time
goes.

No row here is a claim about their card, and no row in §2 or §3 is a claim about
ours.

## 5. Bandwidth normalisation

The two machines differ in memory bandwidth by 2.5×, so the tok/s columns cannot
be compared directly. Dividing bytes per token by bandwidth gives the time the
token would take if the engine did nothing but read weights; the ratio of that
ceiling to the measured time is the engine's efficiency per byte.

| | theirs, 27B `ROCmFP4_FAST` (published) | ours, UD-Q4_K_M (measured) |
| --- | ---: | ---: |
| bytes per token | 13.55 GiB = **14.55 GB** | **15.48 GB** (measured stream) |
| bandwidth | **256 GB/s** (256-bit LPDDR5X) | **640 GB/s** (GDDR6 spec) |
| read ceiling | 14.55 ÷ 256 = **56.8 ms** | 15.48 ÷ 640 = **24.19 ms** |
| achieved | 1000 ÷ 14.02 = **71.33 ms** | 1000 ÷ 33.31 = **30.02 ms** |
| efficiency | 56.8 ÷ 71.33 = **79.7% → 80%** | 24.19 ÷ 30.02 = **80.6% → 81%** |

The same arithmetic on their stock `Q4_K_M` row, for the comparison inside their
own table:

| | stock `Q4_K_M` (published) | `ROCmFP4_FAST` (published) |
| --- | ---: | ---: |
| bytes per token | 15.92 GiB = 17.09 GB | 13.55 GiB = 14.55 GB |
| read ceiling at 256 GB/s | 66.77 ms | 56.83 ms |
| achieved | 80.97 ms | 71.33 ms |
| efficiency | **82.5%** | **79.7%** |

**So the engines are equally efficient per byte of bandwidth — 80% against
81% — and their advantage is the format's size, not the engine.** Their own two
rows say the same thing from the other side: swapping the format moved the
bytes by −14.9% and the decode by +13.5%, and the efficiency did not improve
(82.5% → 79.7%). The win is the 2.54 GB they stopped reading, not a faster read.

One caveat that runs in their favour: their numerator is the published *file*
size, which includes the embedding and output tensors the decode does not
stream, while ours is the measured *stream*. Their per-token bytes are therefore
overstated, which raises their ceiling and flatters their efficiency. Their true
efficiency is at or below 80%, so the "equally efficient" conclusion survives a
correction in our favour; it does not depend on it.

**This is why a 4-bit-with-hardware-mapping format is the one remaining decode
lever for us.** Our GEMV already runs at 96.6% of a pure read at its own shapes,
and every attempt to raise that read — pipelining, split-K, streaming loads,
wider grids, 128-bit loads — measured neutral or slower
([`decode-roofline.md`](decode-roofline.md) §2–§3). What is left is fewer bytes
per token. On a whole-file basis their `ROCmFP4_FAST` file is 14.55 GB against
our UD-Q4_K_M file's 16.56 GB (15,788.9 MiB resident,
[`serve-vram.md`](serve-vram.md)) — **12.1% fewer bytes** — and the trade they
measured for that size was +13.5% decode. Applied to our stream, 12.1% is
**2.9 ms** off a 24.19 ms read ceiling, the largest single term in a 30.02 ms
token. That is a bound on the read, not a projection of the token: their own
run returned +13.5% for a −14.9% size change, so even a working format would
realise less than the byte ratio, and no hardware-mapped 4-bit format exists for
`gfx1201` in this tree.

## 6. Why these are not like-for-like

1. **Different GPU architecture and bandwidth.** Theirs is an RDNA 3.5 APU
   (`gfx1151`, 40 CU) on unified 256-bit LPDDR5X at roughly 256 GB/s; ours is a
   discrete RDNA 4 card (`gfx1201`, 64 CU) on GDDR6 at a 640 GB/s spec. That is
   2.5× the bandwidth and 1.6× the CUs, and it is why the raw tok/s columns
   differ before any engine is involved.
2. **Different model, for Strata.** Source 1 runs the same dense 27B we do, so
   that row is the closest thing here to a like-for-like comparison. Strata runs
   a **125B MoE** that streams 24,576 experts from RAM and SSD, so none of its
   absolute numbers transfer; only its techniques do.
3. **Different quantisation.** They ship a custom `ROCmFP4`/`ROCmFP4_FAST` block
   format built for `KHR_coopmat` operands. We run the Unsloth dynamic
   `UD-Q4_K_M` file, whose per-tensor type mix is chosen by Unsloth, not by us.
   The comparison is "their format against our format", not "their format
   against stock `Q4_K_M`".
4. **Different backends.** Their decode runs on Vulkan/RADV Wave64 and their
   prefill on ROCm/HIP — two engines with a hand-off between them. Ours is HIP
   for both.
5. **Self-reported versus measured.** Every figure in §2 and §3 is the
   project's own, published with its own choice of prompt, repetition count,
   KV type and definition of "decode". Ours in §4 is one session on one binary
   with the fixture committed in the tree. Neither side has been reproduced by
   the other.
6. **None of their numbers were run on our card.** No figure in §2 or §3 was
   measured on an R9700, and no figure in §4 was measured on a Ryzen AI Max+
   395. The `gfx1201` kernels those projects publish (Strata does list R9700
   rows for its prompt path) are not the kernels behind the numbers quoted here.
7. **Different measurement axes.** Their 27B row reports bare decode, MTP decode
   and `pp512` only; they publish no serve cell, no depth sweep and no 8-slot or
   16-slot number for that model. Our table has axes they do not report and
   theirs has a `pp512` we report at 512 and 8,192. The rows that exist on both
   sides are decode and prefill; the rest is not a comparison.
8. **Different dates and different trees.** Their claim is dated 2026-08-25 on a
   tree last pushed 2026-09-24; ours is 2026-11-13. Both projects have moved
   since.

## 7. What we take from them

Two techniques, each with the claim it comes with and the fact that the claim is
unverified on our hardware.

**Prompt-lookup drafting.** They draft up to 5 tokens copied from earlier in the
context and use them "only where its measured acceptance and cost say it pays",
claiming **6–11% faster on code edits and unchanged on other text**. We have no
lookup drafter: our speculation is the model's MTP head only. This is the
interesting one for us because it targets exactly the case where our acceptance
is weakest — serve cell a accepts **0.70** of its drafts
([`mtp.md`](mtp.md), "Four-column verify"), and a repeated-context
draft is cheap to verify and nearly free to propose. The 6–11% is unverified on
this card, on this model, and at our acceptance profile.

**Confidence-gated draft depth.** They expose `--spec N --spec-min-p P` — the
depth and the confidence needed to add another guess — and their setup default
is `--spec 4 --spec-min-p 0.5`, with acceptance published per depth. We run a
fixed depth (`spec-bench --drafts 0..3`: 33.32 / 51.88 / 60.50 / 63.22 tok/s at
[`decode-launch-merge.md`](decode-launch-merge.md)), so every token pays for the
deepest draft whether or not the shallow ones were accepted. A gate that stops
extending a tree once the draft head is unsure is the obvious next shape, and
their per-depth acceptance table is the reporting we would need to justify it.
The gate's benefit on our card is unverified; our own numbers say depth 3 is
worth 1.90× plain decode, so a gate that truncates too eagerly could cost more
than it saves.

Their remaining items are recorded in §3 but not proposed: the draft-vocabulary
subsets (English-only, ~110 MiB less VRAM, 1–2% faster), the `q4_0` KV with a
Hadamard rotation (~4% faster at 128K, perplexity +8–12%) and `k8v4` (23% less
KV, 2–5% slower prompts) are all trades we would have to measure on our own KV
path before adopting, and none of them is a decode-bandwidth lever.

## 8. What this note does not establish

- It does not rank the engines. Efficiency per byte is 80% against 81%, a
  one-point difference at this precision, and neither side has run the other's
  harness.
- It does not reproduce anything. No external number here was re-run, and no
  external binary was built or executed.
- It does not predict what an FP4-class format would do on `gfx1201`. §5 bounds
  the read at 2.9 ms and says nothing about whether the format's kernels would
  reach their own byte ratio on this architecture.
- It does not claim the 6–11% or the 1.6–1.8× for us. Both are their numbers on
  their model and their card.
