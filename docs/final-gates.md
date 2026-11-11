# Final gates: stock Qwen3.8-27B-Q4_K_M on the release tree

This is the GPU verification of the release candidate **`dd62925`** on the stock
release model, one window per gate group, with every raw log under
[`bench/release/gates2/`](../bench/release/gates2/). The tree is the one
[`release-run.md`](release-run.md) describes; this file records that its gates
still pass on that tree and on that file.

## Machine, model, binary

| field | value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 (gfx1201), 32 GiB (an iGPU is also present; `probe` enumerates two HIP agents) |
| GPU driver | 32.0.32015.7029 (2026-10-18) |
| OS | Windows 11 Pro (build 26200) |
| ROCm | 7.2.1 (`C:\rocm-sdk\_rocm_sdk_core`), hipcc AMD clang 22.0.0git |
| llama.cpp | build 11820, commit `d7725699` (not re-run; matrix unchanged) |
| model | `Qwen3.8-27B-Q4_K_M.gguf` |
| model bytes | **16,810,714,336** |
| model SHA-256 | `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` |
| MTP drafter | `mtp-Qwen3.8-27B-Q4_0.gguf`, 1,369,590,656 B (not hashed) |
| hipml tree HEAD | `dd62925` (`feat/final3`) |
| hipml binary SHA-256 | `9d7ff8502cee7df4b9c30b4e730be34a7ee86e249c6bd6f81c3105e286a9dc06` |

Build: `cargo build --release -p hipml-cli --features hip` with `HIP_PATH`,
`HIPML_ARCH=gfx1201`, clean in 28.0 s. `hipml --help` lists every documented
command including the new `cache-gate` ([`cli-help.out`](../bench/release/gates2/cli-help.out)).
`hipml --version` is **not a command**: it prints "unknown command: --version"
and exits 2. No doc references `--version`; `--help` prints `hipml 0.1.0`.

## Gates

| gate | expected | observed | verdict |
| --- | --- | --- | --- |
| `load` on the stock GGUF | 866 tensors, 16,799,719,424 B, `unsupported: none` | 866 / 16,799,719,424 B / none; 20/20 round-trip byte-identical | PASS |
| `oracle-gate --ids-file p2_math_169.ids --steps 20 --chunk 2048 --kv q8` | 20/20, step 19 = **1420** | 20/20, step 19 = **1420** | PASS |
| `parity --deep` | 56 passed, 0 failed, 1 skipped | 56 / 0 / 1 (`sample_top_p_f32`) | PASS |
| `parity prefill --deep` | 6/6 | 6 / 0 / 0 | PASS |
| 20-id acceptance | `11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760` | identical | PASS |
| `gemv-check` 4 tensors x R=1/2/4 | 12/12, 0 differing | 12/12, 0 differing, max abs 0.000e0 | PASS |
| `spec-bench` drafts 0/1/2/3, stock model | 128/128 exact every row | 29.46 / 46.47 / 55.54 / 56.03 tok/s; 128/128 exact on all four rows | PASS |
| `batch-bench --check` B=8 | every slot 16/16 vs B=1 | 8/8 slots 16/16 | PASS |
| `batch-bench --check` B=16 | every slot 16/16 vs B=1 | 16/16 slots 16/16 | PASS |
| `rows-check --slots 16` | 112 decode ids match | 112 ids match; SSM max abs 2.33e-2 | PASS |
| `tokenizer_fixture` | 2/2 | 2 passed, 0 failed | PASS |
| `cache-gate --len 8192 --shared 4096 --chunk 2048` | undersized capture refused; output ids equal cold | refused (1,174.6 MiB > 1,024 MiB); legacy diverges at id 0; fallback ids equal cold | PASS |
| chunk boundary, 8,193 tokens | prefill succeeds, last id **293** | 6.77 s (1,210 tok/s), last id **293** | PASS |
| `splitk-gate` synthetic | PASS, 44/44 | PASS 44/44, ids identical, worst abs 0.014487 <= 0.05 | PASS |
| `splitk-gate --ids-file p2_math_169.ids` | PASS, 44/44 | PASS 44/44, ids identical, worst abs 5.439910 <= 16 | PASS |

Raw output: [`load.out`](../bench/release/gates2/load.out),
[`oracle-gate.out`](../bench/release/gates2/oracle-gate.out),
[`parity-deep.out`](../bench/release/gates2/parity-deep.out),
[`parity-prefill-deep.out`](../bench/release/gates2/parity-prefill-deep.out),
[`acceptance-20id.out`](../bench/release/gates2/acceptance-20id.out),
[`gemv-check.out`](../bench/release/gates2/gemv-check.out),
[`spec-bench.out`](../bench/release/gates2/spec-bench.out),
[`batch-b8.out`](../bench/release/gates2/batch-b8.out),
[`batch-b16.out`](../bench/release/gates2/batch-b16.out),
[`rows-check.out`](../bench/release/gates2/rows-check.out),
[`tokenizer-fixture.out`](../bench/release/gates2/tokenizer-fixture.out),
[`cache-gate.out`](../bench/release/gates2/cache-gate.out),
[`chunk-8193.out`](../bench/release/gates2/chunk-8193.out),
[`splitk-gate.out`](../bench/release/gates2/splitk-gate.out),
[`splitk-gate-ids.out`](../bench/release/gates2/splitk-gate-ids.out).

### Spec-bench on the stock model

`spec-bench --tokens 128 --ids-file bench/fixtures/p2_math_169.ids`, MTP
drafter, f32 KV, fresh engine per row:

| drafts | tok/s | vs plain | accept | exact |
| ---: | ---: | ---: | --- | --- |
| 0 | 29.46 | 1.00x | - | - |
| 1 | 46.47 | 1.58x | 0.969 | 128/128 |
| 2 | 55.54 | 1.89x | 0.978, 0.864 | 128/128 |
| 3 | 56.03 | 1.91x | 0.946, 0.829, 0.931 | 128/128 |

This replaces the gate-model `spec-bench` row in [`release-run.md`](release-run.md).

## Server smoke (merged binary)

`hipml serve --gguf <stock> --mtp <drafter> --host 127.0.0.1 --port 8093
--slots 8 --ctx 16384` (q8 KV is the server's only mode; 8 slots/16,384 ctx is
the release command).

| check | result | log |
| --- | --- | --- |
| 1. greedy completion | **200**, sane `chat.completion` body | [`server-greedy.out`](../bench/release/gates2/server-greedy.out) |
| 2. non-greedy (`temperature 0.7`) | **400**, greedy-only message | [`server-nongreedy.out`](../bench/release/gates2/server-nongreedy.out) |
| 3. `X-Hipml-Batch-Exact` | present, **0**, with `X-Hipml-Batch-Tile-Min: 4` (documented default: WMMA tile at B>=4) | `server-greedy.out` |
| 4. repeated prompt | TTFT **3,477.6 -> 681.3 ms**; `prefix_hit_tokens` 2,400; ids `[271, 2064, 7701, 488]` identical | [`server-prefix-cache.out`](../bench/release/gates2/server-prefix-cache.out) |
| 5. `hipml_prefix_refused_total` | present (value 0) | [`server-metrics.out`](../bench/release/gates2/server-metrics.out) |
| 6. prompt > slot context | **500** with `admit: prompt 20059 + 1 tokens exceeds the 16384 token slot` | [`server-oversize.out`](../bench/release/gates2/server-oversize.out) |
| 7. shutdown | no `hipml` and no `llama-server` process; no listener on 8093 (only TIME_WAIT sockets) | [`server-shutdown.out`](../bench/release/gates2/server-shutdown.out) |

## Deviations from the recorded run

- `hipml --version` is unsupported (exit 2). The docs never claim it; `--help`
  is the version-bearing command and lists `cache-gate`.
- `splitk-gate` synthetic worst abs is 0.014487 (recorded 0.011199); the
  ids-file cell is 5.439910 (recorded 9.309817). Both pass their bounds
  (0.05 and 16); the ids are identical. This is run-to-run/GEMV-order movement,
  not a verdict change.
- `rows-check` SSM last-bit movement is 2.33e-2 (recorded 2.45e-2); the 112
  decoded ids are identical, which is what the gate asserts.
- `cache-gate` cold fallback is 7,594.5 ms (recorded 7,397.7 ms); the refusal,
  the legacy divergence at id 0, and the equal ids are unchanged.
- Server oversize: [`server.md`](server.md) documents failure scopes but no
  status for an over-context prompt. The code fails the job (`admit: ... exceeds
  the ... token slot`), which surfaces as **500** on a non-streamed request and
  as a 200 SSE with an `error` event when streaming. No doc contradicts this;
  the status is simply unspecified.
- The earlier `release-run.md` draft cited `bench/release/gates/gateE.out`,
  which is not in the tree. The stock-model spec-bench evidence is
  `gates2/spec-bench.out`.
- With `--mtp`, the startup banner prints `0.00 GiB drafter` although MTP ran
  (`hipml_mtp_requests_total 5`). Reporting only; decode was token-exact.

## Unverified

- MTP at q8 KV, batch decode at depth > 0, the full `mtp-batch-bench` matrix,
  and server cells above 16 slots (unchanged from [`release-run.md`](release-run.md)).
- The MTP drafter SHA-256 was not computed; only its byte size is recorded.
- The head-to-head matrix, chat divergence, and server throughput cells were
  not re-run; this run only re-runs the gate list.
- The second HIP agent (the iGPU) was never used; all work ran on device 0
  (R9700).

## GPU lock

The GPU lock, status checked before every acquire, released between windows and FREE
at the end. Holds: 23:49:18-00:09:31Z (20m13s, 13 s over the 20-minute target
because the oracle and both deep-parity passes ran in one window), 00:09:42-
00:13:36Z (3m54s), 00:14:11-00:21:54Z (7m43s), and a sub-minute probe window.

## Re-run on the published tree

Re-verified at tip `c816388` with the binary built from that tree, SHA-256 `eb4cf305e92543e51742b3f93fd2c853f56851e53e0ba2659efd04c9e2b2698d`; raw logs in [`bench/release/gates3/`](../bench/release/gates3/).

- `load` on `Qwen3.8-27B-Q4_K_M.gguf`: 866 tensors, 16,799,719,424 B, `unsupported: none`, 20/20 round-trip byte-identical — [`load.out`](../bench/release/gates3/load.out).
- `oracle-gate --ids-file bench/fixtures/p2_math_169.ids --steps 20 --chunk 2048 --kv q8`: 20/20, step 19 = **1420** — [`oracle-gate.out`](../bench/release/gates3/oracle-gate.out).
- `cache-gate`: capture @8192 refused (needs 1,174.6 MiB, cap 1,024 MiB), legacy diverges at id 0, fallback ids equal cold ids (20 tokens) — [`cache-gate.out`](../bench/release/gates3/cache-gate.out).
- `generate --text "The capital of France is" --tokens 64 --stop-eos --device`: 63 greedy ids, `stopped=false`, text `Paris. ... Lisbon.`, last ids `14227 369 31785 13 198 760 6511 314` — [`generate-stock.out`](../bench/release/gates3/generate-stock.out).
- 20-id acceptance (`--ids 760,6511,314,9338,369`): `11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760` — [`acceptance-20id.out`](../bench/release/gates3/acceptance-20id.out).

`crates/**` is byte-identical to the earlier full gate run (`dd62925`), so the other gates above carry over.
