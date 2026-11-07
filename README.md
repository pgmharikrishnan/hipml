# hipml

A small inference engine for AMD Radeon GPUs on Windows, in Rust with HIP kernels.

Named after what it replaces: MLX is a thin framework over Metal, and this is a
thin engine over HIP. The goal is the same shape — one host language, one driver
API, kernels you can read — on a card where nobody has built that yet.

```
Python (optional, later)  →  Rust core  →  HIP  →  driver
```

No PyTorch, no ONNX, no MLIR, no graph compiler, no Python in the hot path, and
no external crates at all.

## Status

**M1 complete.** Six kernels run on the card and match the reference; a naive
fp32 GEMV reaches 634 GB/s, which is the card's spec.

| Milestone | What it proves | State |
|---|---|---|
| M0 | Every op has a reference, a parity case and a manifest row | done |
| M1 | Kernels produce the same numbers as the reference on real hardware | done |
| M2 | Quantized weights + KV cache; tokens match llama.cpp | next |
| M3 | Faster than llama.cpp Vulkan on the same card, or the project stops | |

Measured on a Radeon AI PRO R9700 (gfx1201, 32 GB, Windows 11, ROCm 7.2.1):

```
case                 result  detail
gemv_f32             PASS    max_abs 5.96e-7  max_rel 4.77e-6
rmsnorm_f32          PASS    max_abs 9.54e-7  max_rel 4.08e-7
rope_neox_f32        PASS    max_abs 2.38e-7  max_rel 9.58e-6
silu_mul_f32         PASS    max_abs 5.96e-8  max_rel 2.02e-7
softmax_f32          PASS    max_abs 8.94e-8  max_rel 5.01e-7
residual_add_f32     PASS    max_abs 0.00e0  max_rel 0.00e0
```

| Case | Launch | Effective bandwidth |
|---|---:|---:|
| gemv `[40960, 5120]` fp32 (838 MB) | 1349 µs | **621.7 GB/s** |
| attn decode, Ripple at 12K (100.7 MB) | 14868 µs | 6.8 GB/s |
| gemv `[64, 512]` | 1.14 µs | launch-bound |

The attention kernel is correct (parity `1.34e-7`) and deliberately unoptimised:
one block per head with a block reduction per cache row serialises 12,288 rows.
Splitting the KV range across blocks — the flash-decoding shape — is the next
piece of work, and the floor is ~157 µs, so there is room for a 10–30x gain.

### What that means for M3

A 12-line naive kernel already streams weights at 99% of the card's 640 GB/s
spec, so the memory path is not where the remaining speed lives. For reference,
measured on the same box: llama.cpp Vulkan moves 558 GB/s (87% of roofline) on a
27B Q4 model, and 422 GB/s (66%) on Ripple-65B, whose architecture has no Flash
Attention path. The headroom is the missing attention kernel, not the GEMV — so
M3 targets the case where llama.cpp leaves 34% on the table, not the mature one.

## Quickstart

On any machine, no GPU needed — this runs the reference half of the suite:

```bash
cargo test --workspace
cargo run -p hipml-cli -- kernels
cargo run -p hipml-cli -- parity     # every case reports SKIP without a GPU
```

On the GPU host, with the HIP SDK installed:

The toolchain comes from AMD's ROCm Windows wheels — `rocm_sdk_core` and
`rocm_sdk_devel` from `repo.radeon.com/rocm/windows/`, installed with pip into a
directory. That gives you `hipcc` without the driver-touching installer:

```powershell
pip install --no-deps --target C:\rocm-sdk `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl

$env:HIP_PATH  = "C:\rocm-sdk\_rocm_sdk_core"     # the dir with bin/ and lib/
$env:HIPML_ARCH = "gfx1201"
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"          # amdhip64.dll at run time

cargo build --release --features hip
cargo run --release -p hipml-cli --features hip -- probe
cargo run --release -p hipml-cli --features hip -- parity
cargo run --release -p hipml-cli --features hip -- bench
```

`parity` exits non-zero if any case fails, so it is the CI gate as well as the
developer loop. A case that cannot run is reported as `SKIP` with the reason —
never as a pass.

## Layout

```
crates/hipml-core      tensors, deterministic RNG, reference kernels, parity harness
crates/hipml-hip       raw amdhip64 bindings: memory, streams, modules, launches
crates/hipml-kernels   .hip sources, hipcc build step, the kernel manifest
crates/hipml-cli       probe / kernels / parity
```

Adding a kernel means four things in step: a `.hip` source, a reference function,
a parity case, and a manifest row. The CLI walks the manifest, so nothing can be
launched that is not also testable.

## Design rules

- **The reference is the specification.** Kernels are compared against plain Rust
  with the same signature and layout, not against golden files.
- **Explicit over clever.** The argument arrays for `hipModuleLaunchKernel` are
  written out per kernel, because they are the ABI and they should be readable.
- **Skips are visible.** No silent zero-device fallbacks.
- **One vendor API.** `amdhip64` only, no wrappers, no rocBLAS yet.

See [`docs/architecture.md`](docs/architecture.md) for the layer budget, the
memory model and the milestones.

## License

Apache-2.0.
