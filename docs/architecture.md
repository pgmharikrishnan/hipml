# Architecture

## Why this exists

On Apple silicon, MLX gives you one host language, one driver API (Metal) and
kernels you can read. On Windows with a Radeon there is nothing equivalent: every
option is llama.cpp underneath, and the research stacks (vLLM, SGLang, ExLlama)
are Linux-only because PyTorch has no ROCm wheels for Windows.

`hipml` is an attempt at the missing layer for one card: an inference engine whose
host is Rust, whose kernels are HIP, and whose dependency list is empty.

## Layer budget

The point of the project is the number of layers between a token and the GPU.

| Stack | Layers |
|---|---|
| PyTorch / vLLM | Python → dispatcher → ATen → HIP → HSA → driver |
| llama.cpp | server → ggml graph → backend → Vulkan → ICD → driver |
| MLX | Python → C++ → Metal → driver |
| **hipml** | **Rust core → HIP (`amdhip64`) → driver** |

One layer of our own code. Everything above it is ours too, which is the cost:
there is no ecosystem to borrow from.

What cannot be removed: a discrete GPU means explicit host-to-device copies. MLX
does not pay that cost because Apple's memory is unified. hipml hides the copies
behind asynchronous staging buffers rather than pretending they are free.

## Decisions

**HIP, not Vulkan.** Vulkan adds an ICD layer, an explicit descriptor and barrier
model, and no first-class device-memory semantics — more host code and fewer
kernels. HIP gives device allocations, streams, events and `hipcc` compiling the
kernels directly. Vulkan stays as the fallback if `gfx1201` proves flaky under
HIP on Windows.

**No external crates.** No `rand`, no `bytemuck`, no `half`. Each dependency is a
layer, a version constraint and an audit surface. The RNG is xorshift64\*, the
tensors are `Vec<f32>`.

**The reference is the specification.** Every kernel has a plain-Rust twin with
the same name, signature and layout in `hipml-core`. Parity is a tolerance
comparison against that twin, driven by one table of cases, so "does it work" is
one command with one verdict.

**Explicit kernel ABI.** `hipModuleLaunchKernel` takes an array of pointers to
arguments, so each launcher writes its argument array out by hand. A macro would
be shorter and would hide exactly the thing that breaks.

**Deterministic inputs.** Cases are seeded, and a test asserts that rebuilding a
case reproduces it bit-for-bit. A failure on the GPU host has to be replayable.

## Memory model (M2)

Two arenas, no general allocator:

- **Weights:** one persistent device allocation per tensor, uploaded once at load
  and never moved. Kept quantized in VRAM; dequantized inside the GEMV.
- **Activations:** a per-graph arena, sized at load from the maximum sequence
  length, reused every token. No `hipMalloc` in the decode loop.

Copies go through pinned host staging buffers and an explicit stream. The
allocator is a bump allocator over the arena with no free list, because nothing
inside a graph frees.

## Quantization (M2)

Weights stay compressed in VRAM and are decompressed inside the kernel that
consumes them — that is the entire bandwidth win on a dense model. The reference
implementation defines the block format first, and the kernel must match it
exactly, block scale rounding included.

Own pack format: one file per GPU, 4096-byte-aligned records, a sha256 per record.
Corruption then becomes a load-time error instead of a wrong answer.

## Milestones

**M0 — harness (done).** Reference kernels for gemv, rmsnorm, rope, silu_mul,
softmax, residual add and top-p sampling; a manifest; a CLI; a HIP layer that
typechecks without a GPU. 10 tests, no GPU required.

**M1 — first real numbers (done).** Kernels compile to a single 24 KB code
object with `hipcc --genco`, load with `hipModuleLoad`, and all six device cases
`PASS` within tolerance on gfx1201. A naive fp32 GEMV at the `ffn_up` shape
(`[40960, 5120]`, 838 MB) runs at **633.6 GB/s**, i.e. 99% of the card's 640 GB/s
spec.

Two things had to change to get there, both recorded in the log rather than
hidden: `hipcc` refuses one `-o` with several sources, so the sources are
amalgamated into one translation unit; and rope parity failed at 7.1e-5 until
both sides built the rotation angle in f64, which is a precision-path fix rather
than a widened tolerance.

The M1 result reframes M3. Since the memory path already reaches the roofline,
the remaining speed is in the parts llama.cpp does not implement for a given
architecture — attention without Flash Attention, above all.

**M2 — a model.** Quantized GEMV, KV cache, a tokenizer, and one small model
generating tokens that match llama.cpp's output on the same prompt. This is where
the engine stops being a kernel library.

**M3 — the verdict.** Run the same model as llama.cpp Vulkan on the same card and
compare tokens per second at a deep context. The measured baseline to beat is a
65B dense model at 67% of the memory roofline; below roughly 85% there is no
reason to keep the second engine.

## Non-goals

Training. Autograd. Multi-GPU. Portability beyond one architecture and one card.
Loading arbitrary checkpoints — packs are built for this engine, not imported.

## Risks

- **`gfx1201` under HIP on Windows is unproven for us.** If the runtime is slower
  than Vulkan, the premise weakens before any kernel is written. The cheapest test
  is running the existing llama.cpp HIP build on the same box.
- **Tolerance drift.** RDNA4 fuses operations differently from the scalar
  reference. Tolerances may need to widen; they must never be widened silently.
- **Scope.** An engine that only runs one model family is a research project, not
  infrastructure. M3 is deliberately a stop-or-continue gate.
