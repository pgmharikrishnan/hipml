#!/bin/sh
# Run one benchmark cell on the GPU host under the wp45 lock.
#   sh bench/serve/run.sh hipml a
#   sh bench/serve/run.sh llama b
# The lock is released on exit even when the run fails.
set -e
ENGINE=${1:?engine}
CELL=${2:?cell}
LOCK='C:\models\gpu-lock.ps1'
powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" acquire wp47
trap 'powershell -NoProfile -ExecutionPolicy Bypass -File "$LOCK" release wp47' EXIT
python "$(dirname "$0")/serve_bench.py" --engine "$ENGINE" --cell "$CELL"
