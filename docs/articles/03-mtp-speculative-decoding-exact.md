# Speculative decoding that is exact: 1.53× to 2.00× from the model's own draft head

*The MTP head drafts at most three tokens, the target verifies them in one weight pass, and every accepted stream in the release run matches plain greedy token for token.*

On a 364-token tool prompt the engine emits 65 tokens at 58.60 tok/s. Plain greedy on the same ids runs at 29.35 tok/s. The 65 tokens are identical to plain greedy, in order, and the run stops at EOS rather than at the token limit. The other four release conversations follow: 1.53× to 1.91×, and 384/384, 384/384, 174/174 and 100/100 tokens identical to plain decode.

## Draft, verify, accept

The model carries a multi-token-prediction head: one extra transformer block (`blk.64`) plus two glue tensors, `nextn.eh_proj` and `shared_head_norm`, in a small sidecar. The head shares the target's embedding and output head, runs its own f32 KV cache, and drafts recursively: token 1 from the last target hidden state, token 2 from the state after token 1, up to three.

The target then runs all `1 + drafts` columns through all 64 layers in one weight pass:

```
draft:  d1, d2, .. dD              D <= 3, recursive
verify: [t, d1, .. dD]             one weight pass, R = 1 + D columns
accept: ids[c] == d_{c+1}          longest matching prefix a
emit:   d1..da, ids[a]             a accepted drafts + the correction
```

The verify is layer-major, so the weights are streamed once for the whole step. Every projection uses the batched GEMV, which keeps the single-sequence accumulation order per row; the GDN chain, the attention front end and the KV append are one launch over all columns; causal attention is one split-K launch over all columns. Each GDN layer writes its recurrent state after every column but the last, so a rejected draft restores the state at the accepted prefix and the KV rows past that position are overwritten by the next append.

Acceptance is an id comparison, not a probability test. If every verify column is bit-identical to a plain decode, the accepted prefix plus the correction is the plain greedy stream. If a column differs by one ulp, the comparison can accept a token plain decode would not have produced. Exactness is therefore not a property of the acceptance rule; it is a property of the kernels.

## The release matrix

| conversation | prompt tok | plain tok/s | drafts | MTP tok/s | speedup | accept | exact |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 01_single_turn_code | 91 | 29.34 | 2 | 49.49 | 1.69× | 0.874, 0.758 | 384/384 |
| 02_multi_turn | 134 | 29.29 | 2 | 44.79 | 1.53× | 0.754, 0.714 | 384/384 |
| 03_system_tool | 364 | 29.35 | 3 | 58.60 | 2.00× | 0.944, 0.941, 0.875 | 65/65 |
| 04_longdoc_qa | 8,192 | 25.14 | 3 | 46.01 | 1.83× | 0.885, 0.870, 0.900 | 174/174 |
| 05_copy | 89 | 29.43 | 3 | 56.33 | 1.91× | 0.897, 0.923, 0.875 | 100/100 |

Drafts is the count with the highest measured rate, not a fixed setting; the sweep covers 1 to 3. Acceptance is per draft position, `P(draft k | 1..k-1 correct)`, and ranges from 0.714 to 0.944. The two conversations that reach EOS are also token-identical to llama.cpp's greedy stream (65 and 174 tokens). The other three diverge from llama.cpp at near-ties of 0.2064, 0.0059 and 0.0260 nats, the KV-layout difference documented in the release run; that is a separate comparison from speculation.

The exactness premise is checked in two places. The stock run checks it end to end: every cell in the table above is exact, including 384 tokens on the two conversations that hit the length limit. The gate run, on an earlier Qwen3.8-27B quantisation, checks the mechanism. `gemv-check` compares each batched projection with the row-tile decode entry at R=1, 2 and 4: 12/12, zero differing values, max |d| 0.000e0. `column-check` compares the whole pass — residual stream, post-norm hidden row and all 248,320 logits of every column — against that many plain decodes, bit-identical at R=1, 2 and 4. The acceptance rule then does the rest.

## The verify pass is the cost

A step that accepts nothing still costs a full step. In the gate run on an earlier Qwen3.8-27B quantisation, a plain step is 34.33 ms. With one draft the step is 43.53 ms, 1.27× plain, of which 2.64 ms is drafting and 40.72 ms is verify. With three drafts it is 65.00 ms, 1.89×, of which 56.70 ms is verify. The floor rises with the draft count because the verify pass, not the draft, is most of the step.

Column scaling is sublinear but not free. On the same earlier quantisation, two columns cost 1.11× a one-column pass, three cost 1.29× and four cost 1.54×. The fixed part is the weight stream, about 32 ms; the variable part is the GDN recurrence, which is sequential in token order and cannot be batched across columns. Attention is batched and costs about 1.2 ms for all columns.

Speculation pays only when acceptance beats that floor. For `copy` the per-draft acceptance is 0.897/0.923/0.875 and three drafts return 56.33 tok/s against 29.43. For `prose` in the earlier sweep, acceptance fell to 0.736/0.590/0.565 and the third draft was slower than the second. The release run picks two drafts for the two lowest-acceptance conversations and three for the rest.

## The server flag that broke it

The drafter and the verify are exact in the CLI. In the server they were not, for a while. `serve` sets `HIPML_BATCH_SKINNY=1` for the whole process so batched decode at four or more slots can use a WMMA tile that is faster than the exact batched GEMV. An MTP verify with three drafts is a four-column `gemv_batched`. At R≥4 the same flag selected the WMMA tile, which is not bit-identical to the one-column launch. The verify columns drifted from plain decode, and the server's drafts=3 output diverged at token 20, before the fix: `41020, 11, 1179` where plain decode gave `41020, 321, 1067`.

The CLI was exact throughout because it never set the server's flag; the bug was server-only. The fix is a thread-local exact scope: `forward_columns` holds it unless the skinny path was requested explicitly, and the kernel dispatch honours it. A process-wide performance switch no longer reaches an operation whose contract is bit-identity.

After the fix, the release run's five cells are exact (the table above). A 343-token run that fills the context to its last row is also exact, and drafts=2 and drafts=3 both match plain decode over 128 tokens.

## What this does not prove

- The release MTP cells use f32 KV. MTP at q8 KV on the stock model is implemented but not measured, so these rates do not transfer to the q8 serving matrix.
- Five fixtures, 384 tokens each, or 65 to 174 when EOS stops early. Exactness over 384 tokens is not a proof over 100,000.
- The drafter is the model's own head, not a separately trained draft model. No comparison against a dedicated drafter was run.
- Acceptance is workload-dependent. An earlier Qwen3.8-27B quantisation at 8K with q8 KV fell to about 0.55 acceptance, and three drafts were slower than plain. The release long-doc cell is f32 KV and 1.83×; the two are not the same experiment.
- The verify multipliers of 1.11/1.29/1.54 come from an earlier quantisation, not the release file.
- The speedup is single-stream. In the server, prefix-cache hits skip MTP because the drafter's KV cannot cover a restored prefix; the 8- and 16-slot cells are batched decode, not speculation. See [article 05](05-serving-8-agents-on-one-card.md).

## Where the numbers come from

- The five cells, accept rates, exact counts, chat-divergence margins and the f32-KV caveat: [`../release-run.md`](../release-run.md).
- Raw per-conversation MTP and comparison output: [`../../bench/release/mtp/mtp_summary.txt`](../../bench/release/mtp/mtp_summary.txt) and [`../../bench/release/mtp/summary.json`](../../bench/release/mtp/summary.json).
- Step decomposition, no-accept floors and 128/128 exact: [`../../bench/release/gates/gateD1.out`](../../bench/release/gates/gateD1.out) (gate run, earlier Qwen3.8-27B quantisation).
- The drafter, verify layout, column scaling and the earlier 8K result: [`../notes/mtp.md`](../notes/mtp.md).
- The server divergence at token 20, the process-wide flag and the thread-local fix: [`../notes/review4-server-audit.md`](../notes/review4-server-audit.md).
- The fix itself: [`../../crates/hipml-model/src/gpu/columns.rs`](../../crates/hipml-model/src/gpu/columns.rs) and [`../../crates/hipml-kernels/src/lib.rs`](../../crates/hipml-kernels/src/lib.rs).
