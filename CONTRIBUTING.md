# Contributing to hipml

hipml is a small inference engine for one target: an AMD Radeon GPU on Windows,
one binary, no external crates, no Python in the hot path. Patches that keep it
that way are easy to merge. The rules below are the ones the maintainers apply
to their own work.

If you are unsure whether a change fits, open a discussion first — new models,
new GPUs and Linux support are large changes to an engine built for one target,
not a PR.

## Build

### CPU only (any OS)

This is the path CI takes. No GPU, no ROCm, no model.

```bash
cargo check --workspace
cargo test --workspace
cargo run -p hipml-cli -- kernels
cargo run -p hipml-cli -- parity      # every case reports SKIP without a GPU
```

Without `HIP_PATH` both `build.rs` files skip kernel compilation and print one
warning. That is by design: the CPU reference engine and the CLI must build and
test on a laptop. The tokenizer fixture test also skips, loudly, unless
`HIPML_GGUF` points at the model the fixture was captured from.

### GPU (Windows 11 + ROCm 7.2.1)

`hipcc` comes from AMD's Windows wheels, installed with pip into a directory —
no installer, nothing written outside it:

```powershell
pip install --no-deps --target C:\rocm-sdk `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl

$env:HIP_PATH   = "C:\rocm-sdk\_rocm_sdk_core"   # the dir with bin/ and lib/
$env:HIPML_ARCH = "gfx1201"
$env:PATH       = "$env:HIP_PATH\bin;$env:PATH"   # amdhip64.dll at run time

cargo build --release -p hipml-cli --features hip
.\target\release\hipml.exe probe                  # must list gfx1201
```

Model paths are environment variables, never literals: `HIPML_GGUF` (the
target GGUF), `HIPML_MTP_GGUF` (the drafter), `LLAMA_SERVER` (the llama.cpp
binary for the comparison cells). The full list is in
[`docs/usage.md`](docs/usage.md).

## Gates a PR must pass

CI (`.github/workflows/ci.yml`) runs five jobs on Ubuntu with no GPU, and all
five are required:

| job | command |
|---|---|
| `fmt` | `cargo fmt --all -- --check` |
| `check` | `cargo check --workspace` (and the kernels-skipped warning is asserted) |
| `test` | `cargo test --workspace` (and the tokenizer skip is asserted to be a skip) |
| `guard` | `bash scripts/check-public-tree.sh` |
| `docs-links` | `python3 scripts/check-doc-links.py` |

Plus, when the change touches a kernel, the forward pass, prefill, decode, MTP,
batching, the server or a number in the README, the relevant steps of
[`docs/reproduce.md`](docs/reproduce.md) on real hardware. A kernel without a
parity case is not merged; a change to the prefill, decode, MTP or batch path
must pass `oracle-gate`.

Correctness claims need a reference: every kernel has a plain-Rust twin, a
parity case and a manifest row (see [`docs/architecture.md`](docs/architecture.md)).
Skips are reported as skips — a case that cannot run on your machine is never
counted as a pass.

## GPU performance claims

**A GPU performance claim needs a committed `results.json`.** The file must
carry the hardware and the software that produced the number:

```json
"meta": {
  "gpu": "Radeon AI PRO R9700", "arch": "gfx1201",
  "driver": "31.0.x", "rocm": "7.2.1",
  "model_sha256": "<sha256 of the GGUF>", "hipml_sha256": "<sha256 of hipml.exe>"
}
```

`bench/run.sh` writes that block; `python bench/validate_results.py` checks it.
Compute a model hash with `Get-FileHash .\model.gguf -Algorithm SHA256` on
Windows or `sha256sum model.gguf` anywhere else. Numbers in a prose summary
without a results file are not merged, and a table shows every cell, including
the ones hipml loses.

**Hardware.** The maintainers run one GPU: a Radeon AI PRO R9700 (gfx1201,
32 GB) on Windows 11 with ROCm 7.2.1. We cannot verify any other card, driver
or operating system. Results from your hardware are welcome as a new row —
[open a hardware-result issue](.github/ISSUE_TEMPLATE/hardware-result.yml) —
but they never replace the R9700 row, and a regression there is not a
regression here.

## Code size

Aim for files under ~400 lines. When a module passes that, split it along the
seams the code already has (`serve/sched/` is one example: admit, run, step,
rows, fail). Generated tables (`crates/hipml-core/src/unicode_tables.rs`) and
the kernel manifest in `crates/hipml-kernels/src/lib.rs` are the intended
exceptions. The workspace has no external crates; adding one needs a discussion
before a PR.

## Commit style

- One concern per commit. A formatting-only change is its own commit
  (`style: cargo fmt`), never mixed into a feature.
- Subject: `area: imperative`, at most ~72 characters — `kernels: split the KV
  range across blocks`, `docs: correct the prefill table`.
- Body: what the change does and **which check proves it** (`parity --deep`,
  `oracle-gate`, the bench cell). If a number changed, name the file it is in.
- No machine paths: no home directories, no drive-letter model paths, nothing
  that only resolves on your own box. `scripts/check-public-tree.sh` runs in CI
  and rejects them, and `docs/usage.md` shows the environment variables to use
  instead. No build output, no weights, no `.exe`, no `.hsaco`/`.co` files.
- Never commit a GGUF. `HIPML_GGUF` exists so the repository never has to.

## Adding a quant type

A new quant type is merged only with a parity case. The order that works, all
six in one PR:

1. `TensorType::from_id` and the block-size/byte arithmetic in
   `crates/hipml-core/src/quant.rs`.
2. The scalar dequantiser and the plain-Rust `gemv_*` reference next to the
   existing ones.
3. A hand-computed vector in `crates/hipml-core/tests/quant_reference.rs`: the
   input bytes and the expected float derived on paper, asserted exactly. This
   is the test that catches a misread block layout.
4. The `.hip` kernel and its manifest row in `crates/hipml-kernels/src/lib.rs`,
   with the launch parameters written out explicitly.
5. A case in `crates/hipml-core/src/quant_parity.rs` with a stated tolerance.
6. A row in `hipml gemv-check`, then `oracle-gate` on a real model to prove the
   whole path, not just the kernel.

If step 6 is not green on the R9700, the type is not added. Half a format is
worse than none: a tensor that loads and computes the wrong number is a silent
corruption bug.

## Reporting bugs

Use the issue forms. A useful bug report has the command, the full output tail,
your GPU, driver and ROCm version, and the model's SHA-256. Security issues go
to [`SECURITY.md`](SECURITY.md), never to a public issue.
