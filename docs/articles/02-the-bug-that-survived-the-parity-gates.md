# The bug that survived every parity gate: 40 of 48 heads running on zeros

*One launch argument left five sixths of a projection unwritten, and every correctness check the project owned was blind to it.*

At 169 prompt tokens the logits at the last prefill row were **9.74 nats** from the CPU f32 reference, with an rms error of 1.77. At step 19 the engine emitted token `1420` where the reference emitted `41020`. The same engine, same weights, same ids, prefilling one token at a time, was **0.0001 nats** from the reference.

That is the shape of the failure: not a crash, not a NaN, not a wrong answer that looks wrong. The production path was wrong by enough to change what the model said, and right enough to keep saying it fluently.

## The suite was green

The suite that shipped it was green:

- `parity --deep`: 56 passed, 0 failed, 1 deliberate skip.
- `parity prefill --deep`: 6 cases.
- `gemv-check`: 15/15, max absolute difference 0.
- `attn-parity`: 7/7.
- `spec-bench`: 128/128 ids exact on every row.
- `batch-bench --check`: every slot matched B=1 at B=8 and B=16.
- The 20-id acceptance string, `cargo test --workspace`, `cargo check --features hip`.

None of that is a story about lazy testing. It is a story about what the tests hold constant. Parity compares a kernel with a plain-Rust twin at one shape; the other harnesses compare the GPU with itself, or with a differently shaped launch; all of them go through the same grid helper. And the 20-id acceptance prompt is five tokens long: it exercises the token loop, not the chunk.

The projection that broke had no parity case at all. It was 48 rows wide — the two f32 alpha and beta tensors each SSM layer uses to build its gates — and it only ran on the chunked path.

## The mistake, in three lines

The kernel wants one warp per `(token, output row)`. For a one-token chunk and 48 rows that is 48 warps, or six blocks of eight warps. The code computed the block count correctly and then passed it to a helper that expects an element count:

```
let warps = 8;
LaunchConfig::linear((tokens * m + warps - 1) / warps, (warps * 32) as u32)
// tokens = 1, m = 48 -> linear(6, 256) -> grid = ceil(6 / 256) = 1 block
// one block is 8 warps, so 8 of 48 rows are written; 40 stay zero
```

`LaunchConfig::linear(n, block)` derives its grid as `ceil(n / block)`. It was handed a block count, divided by the block size a second time, and launched one block. The output buffer was zeroed at allocation, so the missing rows did not fault; they read as zeros.

If alpha and beta are zero, the gated-delta-net gates for 40 of 48 heads are built from zeros. The attention blocks and the FFN still compute normally, which is why the text stayed fluent: eight heads had real gates, forty had none.

## What found it

The gate that caught it is deliberately not composed of kernels. `oracle-gate` runs the CPU f32 engine and the production GPU path on the same token ids for the same number of greedy steps. It compares the two id streams and the prefill logits, and it exits non-zero when the streams disagree *and* the oracle's own margin at the first disagreement is at least 0.05 nats. A smaller margin is a near-tie that fp16 prefill operands and a q8 KV cache may legitimately reorder.

The first run failed. Isolating the cause took a table of hypotheses, each measured alone:

| hypothesis | measurement | verdict |
|---|---|---|
| the chunk path is faithful | chunk vs oracle **9.74** | false |
| the oracle is wrong | token loop vs oracle **0.0001** | false |
| KV size or context length | 0.192 with f32 KV, one layer | false |
| fp32 prefill or split-K | no change | false |
| the alpha/beta projection | forcing one f32 row per launch: 0.192 → **0.00027**; alpha/beta up to **12.19** off, **40 of 48 rows zero**, qkv identical | true |

The qkv projection matching is what closed it: alpha and beta are the only tensors in that block that skip the tiled fp16 GEMM.

## The fix, and what it left

The grid is now computed by a host-side helper, `gemm_rows_grid`, with a unit test that asserts every `(token, row)` gets a warp and that the grid does not over-launch by a full block. The chunk gained a tiled f32 projection so the alpha/beta path stops being a special case, and the prefill parity runner follows the engine's split-K decision, so the shape that broke is inside the suite.

The residue after the fix, worst logit error against the CPU oracle at 169 tokens:

| path | error |
|---|---|
| token loop, f32 weights | 0.0001 |
| chunk, f32 weights + f32 KV | 0.0165 |
| chunk, quantized weights + q8 KV | 0.113 |
| shipped: the same, with fp16 alpha/beta | 0.128 |
| shipped, k-sliced eight ways | 0.161 |

The shipped number is larger than 0.128 because slicing k changes the summation order, and both are dominated by the q8 cache rather than by the projection. The gate's real result is not the logit budget: it is `20/20` ids identical, step 19 included, at `41020`.

## The uncomfortable part: the table

The project's headline benchmark table was measured before this fix. It is a committed run on a real binary, and its timings are real, but the tokens that binary produced on the chunked path came from a model with forty dead heads. The table is therefore evidence about speed and not about behaviour, and it was published as both.

Re-measuring after the fix moved timings by **0.2% to 0.9%**: 100K TTFT went from 107.777 s to 108.00 s, 8K from 6.541 s to 6.598 s. That is inside the band two consecutive runs of one binary already show on this box (107.56 to 108.08 s at 100K). The explanation is simple — the projection is a few dozen rows, and even its first fixed fp16 version cost 0.82 s of a 108-second prefill — but that reasoning would have been a guess before the fix. The table was replaced because its provenance was wrong, not because its milliseconds were.

## What generalizes

Every test that missed this bug shared an implementation assumption with the code under test. The parity harness called the same launch code; the shape table never named the projection; the acceptance prompt was too short to reach the chunk. A test built from the same parts as the system can confirm that the parts agree with each other. It cannot tell you that one of the parts is lying.

The check that worked was cheap in exactly the dimension the others were rich: a different implementation of the whole model, compared on observable output. It did not care how the grid was computed or what shape the case table contained. If you keep one lesson: for every fast path, keep one end-to-end comparison against a slow path that shares no code with it, run it at a depth that reaches the fast path, and make it fail on a number, not on a vibe.

## Where the numbers come from

- 9.74 nats (rms 1.77), 0.0001 on the token loop, the hypothesis table, the post-fix drift table and 20/20 ids: [`../notes/wp42-forward-parity.md`](../notes/wp42-forward-parity.md), sections "What it found" and "Drift after the fix".
- 40 of 48 rows zero, the launch arithmetic and the grid test: [`../../crates/hipml-kernels/src/launch.rs`](../../crates/hipml-kernels/src/launch.rs) and [`../../crates/hipml-kernels/src/launch/device/prefill_launch.rs`](../../crates/hipml-kernels/src/launch/device/prefill_launch.rs).
- The 0.05-nat tolerance, the 20-step comparison and the near-tie rule: [`../../crates/hipml-cli/src/oracle_gate.rs`](../../crates/hipml-cli/src/oracle_gate.rs).
- "The run predates the correctness fix; it moved timings by 0.2–0.9%": [`../benchmarks.md`](../benchmarks.md), status paragraph.
- Pre-fix and post-fix timings: [`../notes/wp42-forward-parity.md`](../notes/wp42-forward-parity.md), "Cost of the projection", against [`../benchmarks.md`](../benchmarks.md).
- The 169-token prompt: [`../../bench/fixtures/p2_math_169.ids`](../../bench/fixtures/p2_math_169.ids).
