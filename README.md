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

**M0 complete.** The parity harness, the reference kernels and the HIP plumbing
exist and are tested. No kernel has run on a GPU yet — that is M1, on the box.

| Milestone | What it proves | State |
|---|---|---|
| M0 | Every op has a reference, a parity case and a manifest row | done |
| M1 | Kernels produce the same numbers as the reference on real hardware | next |
| M2 | Quantized weights + KV cache; tokens match llama.cpp | |
| M3 | Faster than llama.cpp Vulkan on the same card, or the project stops | |

M3 is the go/no-go. The bar is measured, not aspirational: llama.cpp Vulkan
already runs a 65B dense model at 67% of this card's memory roofline, so a
hand-written engine has to reach roughly 85% to be worth its maintenance.

## Quickstart

On any machine, no GPU needed — this runs the reference half of the suite:

```bash
cargo test --workspace
cargo run -p hipml-cli -- kernels
cargo run -p hipml-cli -- parity     # every case reports SKIP without a GPU
```

On the GPU host, with the HIP SDK installed:

```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\6.4"
$env:HIPML_ARCH = "gfx1201"
cargo build --release --features hip
cargo run --release -p hipml-cli --features hip -- probe
cargo run --release -p hipml-cli --features hip -- parity
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
