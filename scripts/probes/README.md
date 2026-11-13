# Standalone probes and the decode ablation harness

Small programs that measure one thing on the GPU, outside the engine. The C++
probes are not part of the build; each needs `hipcc` on `PATH` and `HIP_PATH` set
to a ROCm SDK root, and the ones that include a shipped kernel compile the tree's
own source, so the A/B between two versions of that kernel is "run the probe
against each tree".

```powershell
hipcc -O3 --offload-arch=$env:HIPML_ARCH probe_wmma_gemm.cpp -o probe_wmma_gemm.exe
.\probe_wmma_gemm.exe
```

They contain no machine paths and are cited by the notes in `docs/notes/`.

| file | what it measures |
|---|---|
| `probe_wmma2.cpp` | WMMA throughput with two accumulators in flight |
| `probe_wmma_ceiling.cpp` | the achievable WMMA ceiling at one shape, to compare a kernel against |
| `probe_wmma_cmp.cpp` | the same GEMM through WMMA and through plain `v_mfma`/FMA paths |
| `probe_wmma_gemm.cpp` | a full quantized GEMV/GEMM tile through WMMA, end to end |
| `probe_prefill_tiles.cpp` | the shipped prefill tile per format at the real ffn shapes, and its staging split through `PWW2_PROBE_*` |
| `probe_wmma_int4_fp8.cpp` | register-only issue rate of the gfx1201 fp8 and int4 matrix instructions against the fp16 one |
| `probe_prefill_fp8_acc.cpp` | the fp8 staging tile against an f64 host reference, to price its accuracy |
| `probe_wmma_layout.cpp` | fragment layout and bank-conflict behaviour of the WMMA load/store |
| `probe_norm.cpp` | the decode's `rmsnorm_f32` at 5,120 floats: µs per launch, independent and chained, and the GB/s one launch gets |

`probe_prefill_tiles.cpp` builds the tile with `-DPWW2_W8=1` to stage the two
LDS tiles as fp8 and multiply with `v_wmma_f32_16x16x16_fp8_fp8`; `-DPWW2_W8=0`
is the shipped fp16 tile and is the only setting the engine can reach. Section
10 of [`prefill-tile-ceiling.md`](../../docs/notes/prefill-tile-ceiling.md) has
the A/B and why the fp8 path is a probe and not a route.

## The decode ablation harness

`hipml decode-ablate` measures the decode stage budget by **removing** a stage's
launches and timing the rest, untraced, over N tokens. It is the counterpart of
`probe_prefill_tiles.cpp`'s `PWW2_PROBE_*` switches for the token path: a stage
that is not launched cannot be over-reported by an event, which is what happened
to two earlier trace-based budgets ([`decode-residual-gemv.md`](../../docs/notes/decode-residual-gemv.md) §4).

The switches live in `crates/hipml-model/src/gpu/ablate.rs` and are default-off:
`HIPML_ABLATE=a,b` seeds the set (so `gpu-bench` can reproduce one cell in a fresh
process) and `hipml decode-ablate` sweeps the table in one process, interleaving
the cells across reps so a clock ramp cannot favour one. `--cells` picks rows,
`--ctx` sizes the caches independently of the depth, `--graph` measures the same
cells through one captured launch per token.

| file | what it does |
|---|---|
| `ablate-sweep.ps1` | the whole table at depth 0 and 8,192 under one lock hold, then the cache-footprint pair |
| `ablate-run.ps1` | one run: prints and clears the ambient `HIPML_*`, checks the model sha256, acquires and releases the lock |
| `ablate-launch.ps1` | detaches `ablate-sweep.ps1` so an ssh drop cannot kill it |
| `norm-probe.ps1` | builds and runs `probe_norm.cpp` under the lock |
| `ab-final.ps1` | the interleaved A/B against a base checkout: 3 reps x 2 processes, both depths, plus the id streams |
| `gates.ps1` | every invariant the card requires, on one tree, under one lock hold |

`ablate-run.ps1` exists because the box used for these runs had a stale ambient
`HIPML_GGUF` pointing at the *stock* file and a globally set `HIPML_MTP_GGUF`, and
`call env.cmd && hipml … %HIPML_GGUF%` expands before `env.cmd` runs, so the stale
value silently wins. The GGUF path is therefore passed literally, `HIPML_MTP_GGUF`
is cleared on its own line, and the model is checked to be 16,564,259,744 B with
sha256 `1BCE504D…7DC529` before the lock is taken.

The drivers take `-Models` (the directory holding the GGUFs and `gpu-lock.ps1`,
default `$env:HIPML_MODELS`) and `-Tree` (the checkout to test, default the one
the script lives in), so they carry no machine paths. `ab-final.ps1` also needs
`-Base`, a checkout of the parent commit to A/B against.
