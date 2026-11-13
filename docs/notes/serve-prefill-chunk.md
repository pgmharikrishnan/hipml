# SCH: the serve prefill chunk — 256 cost cell a 469 ms of TTFT, and every larger chunk flips one greedy id

Sweep base `5bdf904`; the default moved on `233b3ca` (branch
`agent/serve-chunk-512`). Box: R9700 (gfx1201, 64 CUs, 32 GB),
`Qwen3.8-27B-UD-Q4_K_M.gguf` (16,564,259,744 B, sha256 `1bce504d…7dc529`) with
the MTP drafter `mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 B, sha256
`50d9ce5a…1c6e`), q8 KV, port 8093. Every GPU run held the `serve-chunk` lock.
Binary `built with hip feature: true`, arch gfx1201.

## 0. Question and answer

Question: `serve` defaulted to `--chunk 256` while the CLI prefills at 2048.
Cell a's TTFT of 2,500 ms for a 2,423-token prompt implies ~970 tok/s of
prefill, against the CLI's measured 1,346. Is the 256-token tile the cause, what
does a larger tile cost the other cells, and can the default move?

Answer: **the tile is the whole cause, and the default is now 512.** Raising it
to 2048 cuts cell a's TTFT from 2,495 ms to 2,026 ms (−469 ms, −18.8%) with
cells b and c flat, at +1,229 MiB of peak. **Every chunk above 256 changes one
greedy reply**: at chunk ≥ 512, chat 15 of the 16-conversation probe diverges at
token 5, and 122 of its 128 ids differ downstream. The two candidates sit
0.008–0.018 nats apart, so this is a tie broken by last-digit rounding, and
`oracle-gate` passes at every chunk. The owner accepted that tie as a trade and
chose **512**: it is llama-server's own `-ub 512`, so the comparison is
like-for-like instead of hipml having chosen a smaller chunk, and it recovers
372 of the 469 ms (79%) for 179 MiB. 2,048 was offered and declined — it buys 3%
more TTFT for +1,229 MiB and beats llama on a parameter llama does not set.

## 1. The sweep

`bench/serve/serve_bench.py`, one lock hold, interleaved chunk → cell, two
rounds of all twelve cells, `--tokens 64 --reps 3`. Cell a is the plain
single-stream cell (`--drafts 0`, matching the comparison table's 31.64 row);
b and c keep the harness default (`--drafts 1`). Cell a's two values are the
two rounds; p95 is the larger of the two rounds; the tok/s column is the
aggregate (a: median per-stream decode, b/c: wave goodput — do not compare
across them).

| chunk | a TTFT p50 | a p95 | a tok/s | b TTFT p50 | b p95 | b tok/s | c TTFT p50 | c p95 | c tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2,495 / 2,492 | 2,503 | 31.65 / 31.68 | 485 / 486 | 490 | 132.64 / 132.56 | 735 / 735 | 739 | 223.24 / 223.33 |
| 512 | 2,123 / 2,125 | 2,134 | 31.66 / 31.66 | 490 / 487 | 491 | 132.51 / 132.58 | 735 / 734 | 738 | 223.35 / 223.46 |
| 1,024 | 2,058 / 2,058 | 2,069 | 31.66 / 31.66 | 485 / 487 | 491 | 132.71 / 132.50 | 736 / 736 | 739 | 223.30 / 223.15 |
| 2,048 | 2,026 / 2,026 | 2,034 | 31.67 / 31.66 | 487 / 490 | 491 | 132.53 / 132.31 | 738 / 739 | 740 | 222.68 / 222.69 |

Round-to-round spread is ≤0.15% on every TTFT cell and ≤0.4% on every tok/s
cell, so these differences are results, not noise.

**The 469 ms is per-tile fixed cost, not prefill throughput.** A 2,423-token
prompt is 10 tiles at 256 and 2 at 2,048. Fitting the four points gives
**≈59 ms per extra tile** (intercept 1,909 ms, i.e. the floor including HTTP,
admission and the first token). Cell a's TTFT at 2,048 is 2,026 ms against
llama-server's 1,921 — **+5.5%** instead of +30.1%. The CLI's 1,346 tok/s
prefill for the same 2,423 tokens is 1,800 ms, which leaves 226 ms of HTTP and
first-token in the 2,026 — the server and the CLI prefill at the same rate once
the tile stops being 256 tokens.

**What it costs the other cells: nothing measurable.** b's TTFT p50 moves
485→487 ms and c's 735→739 ms across the whole range; b's aggregate is flat
within 0.4%; c's falls 0.3% (223.33 → 222.69), consistently in both rounds but
inside llama's own 11–12% spread on that cell. A larger tile delays the *first
token of a second request* only by the remainder of the tile in flight; at 8
and 16 slots the queue behind fifteen prefills dominates, so the tile size
does not show.

**Peak VRAM is the real price.** The prefill scratch is sized by `cap = chunk`,
not by the tile actually issued, so it scales 8× between 256 and 2,048:

| peak MiB | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: |
| cell a (drafts 0) | 20,091 | 20,271 | 20,623 | 21,321 |
| cell b (8 slots) | 24,349 | 24,528 | 24,881 | 25,578 |
| cell c (16 slots) | 25,287 | 25,466 | 25,819 | 26,516 |

At 16 slots and 2,048 the peak is 26,516 MiB of 32,665 — 6.1 GiB of headroom, so
it fits, but it buys 469 ms of cell-a latency for 1.2 GiB of the headroom that
keeps 16 × 8K slots off the page file.

## 2. Does the tile change the output?

Two probes, both greedy, both at every chunk.

**The 20-id fixture is tile-invariant.** `generate --device --chunk N` on
`--ids 760,6511,314,9338,369` returns
`[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314,
14898, 369, 21047, 13, 198, 760]` at 256, 512, 1,024 and 2,048, byte-identical
to the release gate. The CLI's prefill is not the thing that moves.

**The server's 16-conversation probe is not.** `probe_exact.py --ctx 32768
--tokens 128 --drafts=` (plain decode, `cache_prompt: false`) gives 16 replies
of 128 ids. Chunk 512, 1,024 and 2,048 agree with each other exactly. Chunk 256
differs from all three in **one reply**, chat 15, at position 5:

```
chunk 256   ... 16561, 728,  310, 10033, 3069, 264 ...
chunk 512+  ... 16561, 3069, 264, 791,   43318, 9830 ...
```

Six of the 128 ids still coincide; 122 diverge, because greedy decoding from a
different token is a different sentence. This is not run-to-run noise: 256 was
run twice and produced sha1 `14a417aad54dbd29` both times, 512 twice and
`c77939a1ff1fc722` both times.

**It is a tie, not a numerical regression.** The same prompt through
`generate` (which prints the top-3 log-probs) has the same two candidates at
position 5, and their gap is a few hundredths of a nat at every chunk:

| chunk | 728 | 3069 | gap |
| ---: | ---: | ---: | ---: |
| 256 | −0.7322 | −0.7398 | 0.0076 |
| 512 | −0.7271 | −0.7450 | 0.0179 |
| 1,024 | −0.7271 | −0.7450 | 0.0179 |
| 2,048 | −0.7287 | −0.7434 | 0.0147 |

The log-prob of 728 moves 0.005 nats between tiles; the two candidates are
0.008–0.018 apart, so a last-digit move decides the argmax. (The CLI keeps 728
at every chunk because its decode is the single-sequence path; the server
decodes at `b=1` through `BatchEngine`, and that is where the tie lands on the
other side.)

**llama.cpp takes 256's branch.** The same raw-text prompt to llama-server
(`-ctk q8_0 -ctv q8_0`, greedy, same box) at position 5:
`728 −0.6939`, `3069 −0.7698` — margin 0.0759, token **728**. So on the one
reply where the two tiles disagree, the 256-token tile is the one that matches
the reference implementation, and 2048 is not "more correct".

## 3. The decision

The card's rule was that a chunk that changes ids is a product decision, not a
benchmark win. The owner made that decision: **512 is the default**. What the
choice was between, in one place:

| | keep 256 | 512 | 1,024 | 2,048 |
| --- | --- | --- | --- | --- |
| cell a TTFT p50 | 2,495 ms | 2,123 | 2,058 | **2,026** |
| vs llama 1,921 | +30.1% | +10.5% | +7.1% | **+5.5%** |
| cell b / c tok/s | 132.6 / 223.3 | 132.5 / 223.4 | 132.6 / 223.2 | 132.4 / 222.7 |
| cell c peak | 25,287 MiB | 25,466 | 25,819 | 26,516 |
| greedy probe | reference, matches llama | **1 reply flips** | same flip | same flip |

512 is the defensible value: it is llama-server's own `-ub 512`, so it is
like-for-like rather than a win taken on a parameter the comparison does not
set, and it recovers 372 of the 469 ms (79%) for 179 MiB. 1,024 and 2,048 buy a
further 65 and 97 ms; 2,048 exceeds llama's ubatch and costs 1,229 MiB for 3%
more, so it was declined. The change is one line in
`crates/hipml-cli/src/serve/mod.rs` (`chunk: 256` → `chunk: 512`) plus the help
text and this note. The flip is accepted, not an accident: both candidates are
equally likely under the model at 0.008–0.018 nats, and `oracle-gate` passes at
every chunk.

Two things this does not establish. The probe is 16 conversations at one
context; a wider sweep would put a number on how often a tie like this is
reachable (one in sixteen replies here, at one position in 128 tokens). And the
59 ms per tile is a property of the prefill tile kernel, which this slice was
told not to touch — if the per-tile fixed cost came down, 256 would stop
costing 469 ms and the trade would disappear.

## 4. The harness

`bench/serve/serve_bench.py` hardcoded `--chunk 256`. That pinned the value
under test: no cell run by this script could ever have shown a better default,
and the 2,500 ms TTFT in `benchmarks.md` was the harness's constant, not a
measurement of the server. The flag is gone; the script now passes `--chunk`
only when asked (`--chunk N`), so a cell measures the server's default.

`probe_exact.py` gained `--chunk` (default: the server's default) and `--dump`
(write the ids per draft as JSON), so the same exactness gate can be run at a
fixed chunk and two chunk runs can be diffed id for id. Neither script changed
what it measures.

## 5. Gates on this tree

All on the UD file, q8 KV unless stated, one binary (`built with hip feature:
true`, gfx1201):

| gate | result |
| --- | --- |
| greedy fixture, `--ids 760,6511,314,9338,369 --tokens 20 --chunk 2048` | 20/20, matches the release ids |
| `oracle-gate --steps 20 --chunk 2048 --kv q8`, UD | PASS 20/20, max \|gpu − cpu\| **0.1086 at 101748** |
| `oracle-gate --steps 20 --chunk 2048 --kv q8`, stock Q4_K_M | PASS 20/20, max \|gpu − cpu\| **0.1481 at 34114** |
| `parity --deep` | **74** passed, 0 failed, 1 skipped |
| `parity prefill --deep` | **15** passed, 0 failed |
| `batch-bench --slots 16 --tokens 16 --kv q8 --check` | **16/16** on every slot |
| `spec-bench --tokens 128 --kv q8` | 33.34 / 51.92 / 60.82 / 63.77 tok/s, **128/128** exact each |
| `spec-bench --tokens 128 --kv f32` | 32.69 / 51.02 / 59.73 / 62.35 tok/s, **128/128** exact each |
| `probe_exact.py --ctx 32768 --drafts 1,2,3` | PASS, 48 replies id-for-id |
| `column-bench --check` | **0 differ** (32 columns × 248,320 logits) |
| `cargo test --workspace` | every suite ok, 0 failed |
| `cargo fmt --all -- --check` | clean (run on the dev box; the GPU host has no rustfmt component) |

## 6. The cells at 256 (the old default)

Same session, default chunk, compared with the comparison table:

| cell | recorded | this run | |
| --- | ---: | ---: | --- |
| serve a, plain | 31.68 | 31.65 / 31.68 | equal |
| serve a, d3 | 52.48 | 52.84 | +0.7% |
| serve b, 8 slots | 132.43 | 132.64 / 132.56 | +0.1% |
| serve c, 16 slots | 223.03 | 223.24 / 223.33 | +0.1% |
| CLI prefill @512 tok/s | 1,346 | 1,352.8 (1,353.4 / 1,352.1 / 1,352.9) | +0.5% |
| CLI prefill @8,192 tok/s | 1,326 | 1,342.8 (1,351.4 / 1,342.8 / 1,333.8) | +1.3% |
| CLI decode @0 tok/s | 33.15 | 33.09 | −0.2% |
| CLI decode @8,192 tok/s | 31.98 | 32.14 | +0.5% |

The CLI rows are `generate --device --chunk 2048 --tokens 33 --kv q8`, three
reps in separate processes, on a 512-token prefix, the 8,192-token
`code_edit@8192` fixture and the 672-token `code_edit@0` fixture. Peak VRAM for
MTP d3 is 20,892 MiB, byte-for-byte the recorded figure.

## 7. Re-verification at 512 (the new default)

After `chunk: 256` → `chunk: 512`, `bench/serve/serve_bench.py` was run with no
`--chunk` flag, so each cell measures the server's default. The startup banner
and the recorded argv confirm it: `chunk 512` in every server log, no `--chunk`
in any command. One lock hold, two rounds interleaved a → b → c, then cell a
with three drafts, `--tokens 64 --reps 3`. Cell a is plain (`--drafts 0`); b and
c keep `--drafts 1`.

| cell | round | TTFT p50 | TTFT p95 | tok/s | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| a, plain | 1 | 2,111 | 2,128 | 31.80 | 20,271 |
| a, plain | 2 | 2,124 | 2,129 | 31.67 | 20,271 |
| b, 8 slots | 1 | 489 | 490 | 132.86 | 24,528 |
| b, 8 slots | 2 | 486 | 491 | 132.61 | 24,528 |
| c, 16 slots | 1 | 739 | 740 | 222.77 | 25,466 |
| c, 16 slots | 2 | 736 | 738 | 223.39 | 25,466 |
| a, d3 | 1 | 2,509 | 2,521 | 52.31 | 21,072 |
| a, d3 | 2 | 2,495 | 2,502 | 52.50 | 21,072 |

Every number lands on the 512 row of the sweep: cell a's p50 is 2,111–2,124 ms
against the sweep's 2,123–2,125, and the peak VRAM figures are the sweep's
20,271 / 24,528 / 25,466 MiB exactly. Cell b's goodput is 132.61–132.86 against
the recorded 132.43, cell c's 222.77–223.39 against 223.03, and cell a's plain
decode 31.67–31.80 against 31.68. The MTP d3 cell is 52.31–52.50 tok/s against
the recorded 52.48, and its TTFT falls 2,879 → 2,495–2,509 ms, the same 370 ms
the tile change predicts.

The CLI cells do not use the serve chunk and do not move. On a fresh lock hold,
`generate --device --chunk 2048 --tokens 33 --kv q8`, three processes per cell:
prefill @512 **1,356.6 / 1,349.2 / 1,347.6** tok/s (recorded 1,346), prefill
@8,192 **1,347.8 / 1,337.9 / 1,332.9** (recorded 1,326), decode @0
**33.29 / 33.30 / 33.31** (recorded 33.15), decode @8,192
**32.20 / 32.09 / 32.06** (recorded 31.98). The first pass of the same cell,
immediately after the gates on a hot card, read 1,308–1,331 @512; the fresh
hold reproduces the recorded figure, so the low pass was thermal, not the
change.

Gates, all on the new default and the UD file unless stated, one binary
(`built with hip feature: true`, gfx1201):

| gate | result |
| --- | --- |
| greedy fixture, `--ids 760,6511,314,9338,369 --tokens 20 --chunk 2048` | **20/20**, first 12 ids `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]` |
| `oracle-gate --steps 20 --chunk 2048 --kv q8`, UD | **PASS 20/20**, max \|gpu − cpu\| **0.1086 at 101748** |
| `oracle-gate --steps 20 --chunk 2048 --kv q8`, stock Q4_K_M | **PASS 20/20**, max \|gpu − cpu\| **0.1481 at 34114** |
| `parity --deep` | **74** passed, 0 failed, 1 skipped |
| `parity prefill --deep` | **15** passed, 0 failed, 0 skipped |
| `batch-bench --slots 16 --tokens 16 --kv q8 --check` | **16/16** on every slot |
| `spec-bench --tokens 128 --kv q8` | 33.35 / 51.96 / 60.86 / 63.87 tok/s, **128/128** exact each |
| `spec-bench --tokens 128 --kv f32` | 32.72 / 51.01 / 59.73 / 62.41 tok/s, **128/128** exact each |
| `probe_exact.py --ctx 32768 --drafts 1,2,3`, **server default** | **PASS, 48 replies id-for-id**; d0 chat 15 is `c77939a1ff1fc722`, the 512 branch |
| `column-bench --check` | **0 differ** (32 columns × 248,320 logits) |
| `cargo test --workspace` | every suite ok, 0 failed |
| `cargo fmt --all -- --check` | clean |

The last exactness gate is the one that had to hold: speculation must equal
plain decode at 512 exactly as it did at 256, and it does, for all three draft
depths on the same 16 conversations. The tie flip is present and accepted —
chat 15's reply is the 512 branch the sweep recorded — but it is identical
between plain decode and every speculative depth.
