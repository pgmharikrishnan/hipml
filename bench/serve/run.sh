#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Run one benchmark cell on the GPU host.
#   sh bench/serve/run.sh hipml a
#   sh bench/serve/run.sh llama b
#
# HIPML_GPU_LOCK is optional: when set it is a script that takes
# `acquire <label>` and `release <label>`. The lock is released on exit even
# when the run fails; when the variable is unset there is no locking at all.
set -e

usage() {
  cat <<'EOF'
usage: bench/serve/run.sh ENGINE CELL

Run one serve-benchmark cell. ENGINE is hipml or llama; CELL is the fixture
cell name (a, b, c, ...).

Environment:
  HIPML_GPU_LOCK     optional lock script: acquire <label> / release <label>
  HIPML_LOCK_LABEL   label passed to the lock (default: serve)
  HIPML_GGUF         path to the model
  HIPML_EXE          path to the hipml binary (hipml runs)
  HIPML_MTP_GGUF     path to the MTP model (MTP cells)
  LLAMA_SERVER       path to llama-server (llama runs)
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

ENGINE=${1:?engine}
CELL=${2:?cell}
LOCK="${HIPML_GPU_LOCK:-}"
LABEL="${HIPML_LOCK_LABEL:-serve}"

if [ -n "$LOCK" ]; then
  powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" acquire "$LABEL"
  trap 'powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" release "$LABEL"' EXIT
fi

python "$(dirname "$0")/serve_bench.py" --engine "$ENGINE" --cell "$CELL"
