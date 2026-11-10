# Release run: what supersedes what

This directory is the raw output of the **release run** on the stock model
`Qwen3.8-27B-Q4_K_M.gguf`. `docs/release-run.md` is the prose summary; every
number quoted in the README, the articles and the release notes comes from one
of the files here.

## What this supersedes

- **The pre-fix tables are gone.** `bench/results.json` and `bench/RESULTS.md`
  described pre-fix measurements with binary `5713f143…` at commit `cef7657`,
  *before* the WP42 projection-path correctness fix. The fix moved timings by
  0.2–0.9% and, more importantly, changed the model every number was measured
  on. Both files have been removed; nothing here is derived from them.
- **`docs/benchmarks.md`** now points at `docs/release-run.md` for the
  headline matrix and per-feature cells. Its old narrative table is marked as
  superseded; the file itself is not rewritten because the method section and
  the caveats still apply.
- Older packaging notes that quote pre-fix figures are listed in
  `docs/release-run.md` under "Still citing pre-fix numbers" (a note for P6).

## What did not change

The harness, the fixture format, the depth definition, the q8-class KV choice
and the decode window are the same as `bench/README.md` describes. Both engines
still receive identical token ids; neither retokenizes at run time.

## Layout

```
bench/release/
  README.md          this file
  matrix/            head-to-head cells, one directory per (engine, prompt, depth)
  mtp/               MTP decode on the five chat prompts (EOS stop)
  server/            serving cells 1 / 8 / 16 slots
  batch/             batch decode B=8 and B=16, both engines
  cache/             exact-repeat and shared-prefix TTFT
  tokenizer/         multilingual strings and chat-template diffs
  gates/             the merged-tree gate logs (cargo, parity, oracle, spec, rows)
  matrix/meta.json   GPU, driver, ROCm, llama build, model size and SHA-256
  matrix/results.json  aggregated matrix, including peak VRAM per cell
  matrix/jobs/       per-cell raw jobs and llama-server logs
```

Raw files are copied from the run host unchanged. Timings are as recorded; the
tables in `docs/release-run.md` are medians (ranges in parentheses) exactly as
the harness writes them.
