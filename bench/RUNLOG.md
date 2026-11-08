# Run log: why the first table had `ERROR: missing`

The first committed `RESULTS.md` was rendered from **partial smoke runs**, not
from a completed matrix. `bench/aggregate.py` walks every cell in
`fixtures/prompts.json`, so any cell that had no raw job file was printed as
`ERROR: missing`. That label meant "never attempted", not "launched and
failed". The smoke invocations only produced:

- `hipml prose_explanation@0` (1 rep)
- `llama prose_explanation@0` (1 rep)

Everything else in that table was unrun. This is a harness bug: the aggregator
should not present unrun cells as failures. The new table is filtered to the
planned cells, and `## Completeness` names any cell that is genuinely missing.

## Actual job failures

| when | job | what happened | rerun |
| --- | --- | --- | --- |
| 15:27 | `llama prose_explanation@0` | GPU lock held by a stale `wp22` lock; `acquire` timed out after 30 min, job never started | reran at 15:41, OK |
| 16:25 | `hipml prose_explanation@8192` | I terminated the process (`taskkill /PID`) to stop the incomplete 40-cell run and switch the head-to-head to q8 KV; log shows `hipml exit 1` | reran in the final matrix |

The remaining cells in the first table were never attempted. The incomplete
40-cell run had completed, before it was stopped:

- hipml depth 0: all five prompts, 5 reps
- llama depth 0: all five prompts, 5 reps
- hipml 8192: `code_edit` and `json_tool_call`, 5 reps

## The f32 100K row

The `llama 6.36 tok/s @ 100000` row was a **same-KV-dtype experiment**: both
engines were forced to f32 KV. That is not llama.cpp's path at depth (f32 KV is
13 GiB) and it is not the head-to-head. It is preserved under
`bench/sensitivity/llama_f32_prose_explanation_100000.json` and was replaced by
q8 KV on both sides:

- hipml `--kv q8`
- llama `-ctk q8_0 -ctv q8_0`

The project's earlier 97,932-token q8 measurement was 22.32 tok/s decode; the
new q8 run should land near that, not near 6.36.

## Final plan

| depth | prompts | reps |
| --- | --- | ---: |
| 0 | all five | 3 |
| 8192 | all five | 3 |
| 32768 | `code_edit` | 1 |
| 100000 | `code_edit` | 1 |

Every cell is labelled with its rep count, and the 100K row reports KV type,
peak VRAM, and whether llama.cpp spilled.
