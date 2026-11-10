#!/bin/sh
# Run one benchmark cell on the GPU host.
#   sh bench/serve/run.sh hipml a
#   sh bench/serve/run.sh llama b
#
# HIPML_GPU_LOCK is optional: when set it is a script that takes
# `acquire <label>` and `release <label>`. The lock is released on exit even
# when the run fails; when the variable is unset there is no locking at all.
set -e
ENGINE=${1:?engine}
CELL=${2:?cell}
LOCK="${HIPML_GPU_LOCK:-}"
LABEL="${HIPML_LOCK_LABEL:-serve}"

if [ -n "$LOCK" ]; then
  powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" acquire "$LABEL"
  trap 'powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" release "$LABEL"' EXIT
fi

python "$(dirname "$0")/serve_bench.py" --engine "$ENGINE" --cell "$CELL"
