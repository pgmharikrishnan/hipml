# Reproduce the release run

This is the GPU runbook behind [`release-run.md`](release-run.md). Every expected
value below is a line from that file; if your card disagrees, record the whole
log (GPU, driver, ROCm, binary SHA-256, command line) and report it as new data.

The run is one GPU, one OS, one driver: Radeon AI PRO R9700 (`gfx1201`, 32 GiB),
Windows 11 Pro build 26200, driver 32.0.32015.7029, ROCm 7.2.1. Close every
other GPU process first; the numbers are cold and single-tenant.

## 0. Environment

```powershell
$env:HIP_PATH     = "C:\rocm-sdk\_rocm_sdk_core"   # the directory with bin\ and lib\
$env:HIPML_ARCH   = "gfx1201"
$env:PATH         = "$env:HIP_PATH\bin;$env:PATH"

$env:HIPML_GGUF     = "D:\models\Qwen3.8-27B-Q4_K_M.gguf"
$env:HIPML_MTP_GGUF = "D:\models\mtp-Qwen3.8-27B-Q4_0.gguf"   # optional; MTP cells only
$env:LLAMA_SERVER   = "D:\llama.cpp\build\bin\llama-server.exe"

# optional: a lock script for a shared box (`acquire <label>` / `release <label>`)
# $env:HIPML_GPU_LOCK = "C:\tools\gpu-lock.ps1"
```

The variables and their failure behaviour are in [`usage.md`](usage.md): an
unset `HIPML_GGUF` fails with "set HIPML_GGUF", never with somebody else's path.
`HIPML_MTP_GGUF` and `LLAMA_SERVER` only skip the cells that need them, and they
say so. Budget ~40 GB of disk for the model files and `bench/out/`.

Check the model before using it: `Qwen3.8-27B-Q4_K_M.gguf` is 16,810,714,336
bytes with SHA-256
`e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`. `hipml load`
should report 866 tensors and 16,799,719,424 bytes of tensor data.

llama.cpp must be build **11820**, commit **`d7725699`**, so the comparison is
the one in the release run. Its settings in the head-to-head are `-ngl 99 -c
<ctx> --parallel 1 -ctk q8_0 -ctv q8_0 --no-webui --no-cache-idle-slots`, no
draft model and no speculative decoding. No `-fa`, `-b` or `-ub` flag was
passed, so the build defaults applied: `-fa auto`, `-b 2048`, `-ub 512`.

One distinction: the correctness gates in §§1–3 were measured on the gate model
named in [`release-run.md`](release-run.md) — the fine-tune the gate corpus and
the MTP drafter were built for — while the batch throughput cells in §3 and the
performance matrix and the chat and server cells in §§4–6 use the stock
`Qwen3.8-27B-Q4_K_M.gguf` above. Set `HIPML_GGUF` to the gate model for the
gates, then back to the stock file.

The 20-id acceptance string is **model-specific**. On the stock
`Qwen3.8-27B-Q4_K_M.gguf`, the prompt `760,6511,314,9338,369` decodes to
`11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047
13 198 760`. The gate model produces a different string; compare against the one
for the file you loaded, and record which file it was.

```powershell
cargo build --release -p hipml-cli --features hip
.\target\release\hipml.exe probe        # must list gfx1201
```

## 1. Kernel and model gates

Run these before any benchmark. Each one exits non-zero on failure.

```powershell
.\target\release\hipml.exe parity --deep
# 56 passed, 0 failed, 1 skipped (sample_top_p_f32, host-only)

.\target\release\hipml.exe parity prefill --deep
# 6 passed, 0 failed, 0 skipped

.\target\release\hipml.exe gemv-check --gguf $env:HIPML_GGUF --tensor blk.0.ffn_gate.weight
.\target\release\hipml.exe gemv-check --gguf $env:HIPML_GGUF --tensor blk.0.attn_qkv.weight
.\target\release\hipml.exe gemv-check --gguf $env:HIPML_GGUF --tensor blk.0.ffn_down.weight
.\target\release\hipml.exe gemv-check --gguf $env:HIPML_GGUF --tensor output.weight
# for each tensor and R=1/2/4: 0 differing, max |d| 0.000e0 (12/12 in total)

.\target\release\hipml.exe oracle-gate --gguf $env:HIPML_GGUF `
  --ids-file bench\fixtures\p2_math_169.ids --steps 20 --chunk 2048 --kv q8
# PASS, 20/20; step 19 = 41020

cargo test -p hipml-core --test tokenizer_fixture
# 2 passed, 0 failed (with HIPML_GGUF set; otherwise the id gate is a visible skip)

cargo test --workspace
# 82 passed, 0 failed

.\target\release\hipml.exe splitk-gate --gguf $env:HIPML_GGUF `
  --ids-file bench\fixtures\p2_math_169.ids
# PASS 44/44, ids identical, worst |dlogit| 9.309817 <= 16
```

A chunk boundary is its own case: a prompt of 8,193 tokens must complete in
one prefill and produce last id `293` (6.67 s on the release run).

## 2. Speculative decode

```powershell
.\target\release\hipml.exe spec-bench --gguf $env:HIPML_GGUF `
  --mtp $env:HIPML_MTP_GGUF --ids-file bench\fixtures\p2_math_169.ids --tokens 128
```

Drafts 0/1/2/3 should read 29.15 / 43.07 / 47.84 / 45.59 tok/s, with 128/128
exact on every row. The release MTP cells use f32 KV; MTP at q8 KV is
implemented but was not measured.

## 3. Batched decode

```powershell
.\target\release\hipml.exe batch-bench --gguf $env:HIPML_GGUF --slots 8  --tokens 16 --kv q8 --check
.\target\release\hipml.exe batch-bench --gguf $env:HIPML_GGUF --slots 16 --tokens 16 --kv q8 --check
```

`--check` validates the exact path (`_b` kernels): every slot must reproduce the
B=1 path 16/16 ids over the full stream. Measured exact/skinny aggregate:

| B | exact tok/s | skinny tok/s | llama.cpp `-np B` tok/s |
| ---: | ---: | ---: | ---: |
| 8 | 107.46 | 149.43 | 97.54 |
| 16 | 107.33 | 272.97 | 227.89 |

The skinny path is the opt-in WMMA tile (`HIPML_SKINNY=1`). It is **not**
bit-identical; `batch-bench --check` reports `TOKENS DIFFER` on it. Quote it as
approximate. Run both checks so the skip is explicit rather than implied:

```powershell
.\target\release\hipml.exe batch-bench --gguf $env:HIPML_GGUF --slots 8 --tokens 16 --kv q8 --check
# exact path: every slot 16/16 ids match B=1

$env:HIPML_SKINNY = "1"
.\target\release\hipml.exe batch-bench --gguf $env:HIPML_GGUF --slots 8 --tokens 16 --kv q8 --check
# skinny path: TOKENS DIFFER, as expected; the throughput is approximate
Remove-Item Env:\HIPML_SKINNY     # or start a fresh shell
```

## 4. Head-to-head matrix

```powershell
bash bench/run.sh --refresh-fixture --validate --llama-kv q8_0 --hipml-kv q8
# 5 prompts x depths 0, 8192, 32768, 100000; 3 cold reps at 0 and 8192, 1 at the deep cells
```

If the GPU is on another host, add `--host <gpu-host> --remote-dir <checkout>`;
the harness copies the tree, runs both engines, and writes `bench/results.json`
and `bench/RESULTS.md` with the GPU, driver, binary SHA-256 and llama.cpp version
in the metadata.

Expected `code_edit` cells, prefill / decode / TTFT:

| depth | hipml | llama.cpp |
| ---: | ---: | ---: |
| 0 (672 tokens) | 1,172 tok/s / 29.7 tok/s / 574 ms | 1,000 tok/s / 31.8 tok/s / 698 ms |
| 8,192 | 1,225 / 29.0 / 6,689 ms | 1,160 / 30.9 / 7,080 ms |
| 32,768 | 1,133 / 27.4 / 28,923 ms | 1,048 / 28.7 / 31,300 ms |
| 100,000 | 920 / 23.7 / 108,698 ms | 813 / 22.7 / 122,997 ms |

The full 5-prompt matrix, including the one prefill loss (`copy_200_words` at
depth 0, 831 vs 848 tok/s), is in [`release-run.md`](release-run.md).

## 5. MTP chat and chat divergence

```powershell
python bench\chat\mtp_chat.py --tokens 384 --drafts 3
python bench\chat\compare.py --tokens 256 --llama-fa on
```

`mtp_chat.py` uses f32 KV, greedy, stop at EOS. Expected speedups over plain
greedy on the same ids: 1.69×, 1.53×, 2.00×, 1.83×, 1.91×; exact in every cell.

`compare.py` takes llama.cpp's own `/apply-template` and `/tokenize` as the
reference. The rendered template ids match on all five conversations. Two of
the five greedy streams are identical to EOS; the other three diverge at a
near-tie (0.0059–0.2064 nats) and are listed in [`release-run.md`](release-run.md).

## 6. Server cells

```powershell
python bench\serve\serve_bench.py --engine hipml --cell a --reps 10
python bench\serve\serve_bench.py --engine llama --cell a --reps 10
# repeat for cells b and c (reps 6 and 3)
```

Same five conversations, greedy, 64 new tokens each, warm shared ~2.4K-token
system prompt. Expected aggregate tok/s (hipml vs llama.cpp): cell a (1 slot)
41.19 vs 31.58; cell b (8 slots) 125.25 vs 94.75; cell c (16 slots) 212.14 vs
214.41. Errors: 0 in all six cells.

The harness does not read llama-server's `cache_n` counter, so its
`prefix_hit_tokens` column reads 0 even when the shared prefix was cached; judge
that from its TTFT, not that column.

## 7. Prefix cache

```powershell
.\target\release\hipml.exe cache-bench --gguf $env:HIPML_GGUF `
  --len 8192 --shared 4096 --chunk 2048 --bytes-mib 4096
```

On the patched build (`bench/release/cache/cache_8192_4gib_fixed.txt`): cold
8,192-token prefill 6,724.4 ms; exact repeat 4.6 ms (1,460×); 50% shared prefix
3,498.8 ms (1.94×); hit latency 4.59 ms at 8,192 tokens.

At the default 1 GiB cap the same snapshot needs 1,174.6 MiB, so it is
**refused at capture** (counted in `refused`, never truncated) and the exact
repeat falls back to a full prefill: 6,766.2 ms against 6,691.7 ms cold, with
token-identical ids. That is the fixed behaviour, not a measurement error.

Then run the gate. It replays the old consumer (a partial hit used as exact) and
requires it to diverge, and checks that the fixed fallback matches the uncached
ids:

```powershell
.\target\release\hipml.exe cache-gate --gguf $env:HIPML_GGUF `
  --len 8192 --shared 4096 --chunk 2048
# capture @8192 refused: needs 1174.6 MiB, cap 1024 MiB
# legacy consumer (partial hit @4096 used as exact) diverges at id 0
# fallback ids equal cold ids (20 tokens); stats: 1 entries, 1 refused, 1 hits, 1 misses
# cold 7397.7 ms; an undersized cap is a miss, not a divergent hit: pass
```

Peak process VRAM at the default cap is 20,340.9 MiB (21,270.9 before the fix);
at 4 GiB it is 21,462.9 MiB (22,553.5 before).

## 8. Tokenizer

```powershell
python bench\tokenizer\diff.py --corpus corpus.txt --hipml-ids ids.txt
python bench\tokenizer\template_diff.py --gguf $env:HIPML_GGUF
```

Expected: 6,912 strings, 0 mismatches, 0 llama errors; and 18/18 template cases
pass (tool schemas, tool-call argument escaping, `preserve_thinking`,
`enable_thinking`, reasoning effort, image/video rejection).

## What is not covered

The release run did not measure MTP at q8 KV, batch decode at depth > 0, the full
`mtp-batch-bench` matrix, the skinny path's logit error, server cells above 16
slots, or any GPU other than gfx1201 and any OS other than Windows 11. Do not
extrapolate those rows; add them as new measurements.

## What will differ on your hardware

- **Bandwidth decides the decode cells.** The R9700's spec is 640 GB/s and the
  release GEMV streams at the card's limit, so a card with more bandwidth (an RX
  7900 XTX is rated 960 GB/s) can beat the R9700 on the same kernel, and a card
  with less will lose. That is the memory system, not the engine; say which card
  you used.
- **VRAM decides whether the 100K cell runs at all.** The 100K row needs the
  weights plus a long KV; a 24 GB card will spill or refuse, and a spill is
  reported as a spill, not as a slow number. The release peaks at 21.26 GiB.
- **VRAM accounting differs by OS.** The peak column on Windows is WDDM "GPU
  Process Memory \ Dedicated Usage" for the process. Linux reports differently,
  so a Linux number is not directly comparable to that column.
- **The ROCm version changes codegen.** The wheels here are 7.2.1. A different
  HIP compiler can move a tolerance-edge case such as the q8 chunk attention by
  a few e-4; check the max abs, not just PASS.
- **The host CPU moves the wall clock, not the verdicts.** `parity --deep` and
  `oracle-gate` run the f32 reference on the CPU, so a slower host takes longer
  and prints the same verdicts.
- **The llama.cpp column depends on the build.** 11820 (`d7725699`) is the
  comparison baseline; a newer build is a different competitor, so do not mix
  them in one table.
- **Clocks and power plans are part of the result.** Run the matrix with the
  same power plan you reported, and do not compare a laptop-on-battery number
  with a desktop one.

## If a step fails

Attach the command, the output tail, your GPU, driver and ROCm version, and the
model SHA-256 to a [bug report](../.github/ISSUE_TEMPLATE/bug.yml). Different
hardware is welcome as a new row: open a
[hardware-result issue](../.github/ISSUE_TEMPLATE/hardware-result.yml) with the
`results.json` attached. For anything that looks like a memory-safety or server
issue, read [`../SECURITY.md`](../SECURITY.md) first and report it privately.
