#!/usr/bin/env bash
# One command for the like-for-like bench.
#
#   bench/run.sh                         # all prompts, all depths, both engines
#   bench/run.sh --depth 8192,100000     # only those depths
#   bench/run.sh --engine llama          # only llama.cpp
#   bench/run.sh --force                 # rerun cells that already have a result
#
# It syncs HEAD to rtx, builds hipml, then runs one (engine, prompt, depth) job
# per GPU-lock acquisition. That chunking keeps every hold short: the 100K
# cells default to one rep and never share a lock window with anything else.
# Results are pulled back and rendered locally by bench/aggregate.py.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
HOST="${RTX_HOST:-rtx}"
REMOTE_DIR='C:\hipml-bench'
MODEL="${HIPML_BENCH_MODEL:-C:\\models\\Qwen3.8-27B-Q4_K_M.gguf}"
LLAMA_SERVER="${HIPML_BENCH_LLAMA_SERVER:-C:\\llama.cpp\\rocm-gfx120x\\llama-server.exe}"
DEPTHS="0,8192,32768,100000"
ENGINES="hipml,llama"
REPS=3
REPS_100K=1
REPS_DEEP=1
FORCE=0
SYNC=1
BUILD=1
REFRESH_FIXTURE=0
STAGE=1
SINGLE_HOLD=0
VALIDATE=0
LLAMA_KV="${HIPML_BENCH_LLAMA_KV:-q8_0}"
HIPML_KV="${HIPML_BENCH_HIPML_KV:-q8}"
PROMPTS=""

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --depth LIST        comma-separated depths (default: 0,8192,32768,100000)
  --engine LIST       comma-separated engines: hipml,llama (default: both)
  --prompt LIST       comma-separated prompt labels (default: all five)
  --reps N            cold reps for depths 0 and 8192 (default: 3)
  --reps-deep N       cold reps for depths 32768 and 100000 (default: 1)
  --reps-100k N       alias for the 100000 rep count (default: 1)
  --force             rerun cells even if a result file exists
  --no-sync           use the tree already at C:\hipml-bench
  --skip-build        do not run cargo build on the host
  --refresh-fixture   regenerate bench/fixtures/prompts.json from the host
  --no-stage          skip the hipml prefill stage breakdown at depth 0
  --single-hold       acquire the GPU lock once for this invocation and hold
                      it across every selected cell (keep the selection under
                      20 minutes; the 100K pair is about 14)
  --validate          fail the run if results.json is missing a planned cell,
                      has the wrong rep count, or has the wrong KV type
  --llama-kv TYPE     f32, f16 or q8_0 (default: q8_0)
  --hipml-kv TYPE     f32, q8 or q4 (default: q8; same class as llama q8_0)
  --host NAME         ssh host (default: rtx, or $RTX_HOST)
  -h, --help          this text
EOF
}

die() { echo "run.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --depth) DEPTHS="$2"; shift 2 ;;
    --engine) ENGINES="$2"; shift 2 ;;
    --prompt) PROMPTS="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --reps-deep) REPS_DEEP="$2"; shift 2 ;;
    --reps-100k) REPS_100K="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --no-sync) SYNC=0; shift ;;
    --skip-build) BUILD=0; shift ;;
    --refresh-fixture) REFRESH_FIXTURE=1; shift ;;
    --no-stage) STAGE=0; shift ;;
    --single-hold) SINGLE_HOLD=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --llama-kv) LLAMA_KV="$2"; shift 2 ;;
    --hipml-kv) HIPML_KV="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1 (try --help)" ;;
  esac
done

cd "$ROOT" || die "cannot cd to $ROOT"

# ---------------------------------------------------------------- sync + build
if [ "$SYNC" = 1 ]; then
  echo "== sync HEAD to $HOST"
  git archive HEAD -o /tmp/hipml-bench.tar || die "git archive failed"
  scp -q /tmp/hipml-bench.tar "$HOST":x.tar || die "scp failed"
  ssh "$HOST" 'cmd /c "if exist C:\hipml-bench rmdir /s /q C:\hipml-bench"' || true
  ssh "$HOST" 'cmd /c "mkdir C:\hipml-bench"' || die "remote mkdir failed"
  ssh "$HOST" 'cmd /c "tar -xf %USERPROFILE%\x.tar -C C:\hipml-bench"' || die "remote extract failed"
fi

if [ "$BUILD" = 1 ]; then
  echo "== build hipml on $HOST"
  ssh "$HOST" "$REMOTE_DIR\\bench\\remote\\with-env.cmd cargo build --release -p hipml-cli --features hip" \
    || die "cargo build failed"
fi

if [ "$REFRESH_FIXTURE" = 1 ]; then
  echo "== regenerate token fixture on $HOST"
  ssh "$HOST" "$REMOTE_DIR\\bench\\remote\\with-env.cmd python bench\\prepare_fixture.py --hipml target\\release\\hipml.exe --gguf $MODEL --out bench\\fixtures\\prompts.json" \
    || die "fixture generation failed"
  mkdir -p bench/fixtures
  scp -q "$HOST":'C:/hipml-bench/bench/fixtures/prompts.json' bench/fixtures/prompts.json \
    || die "fixture pull failed"
fi

[ -f bench/fixtures/prompts.json ] || die "bench/fixtures/prompts.json missing (use --refresh-fixture)"

# ------------------------------------------------------------------- lock helpers
lock_status() {
  ssh "$HOST" 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 status' 2>/dev/null
}
LOCK_LABEL="${HIPML_LOCK_LABEL:-wp24}"
acquire_lock() {
  local st
  st="$(lock_status)"
  case "$st" in
    *"HELD by $LOCK_LABEL"*) LOCKED=1; return 0 ;;
    *FREE*) ssh "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\models\\gpu-lock.ps1 acquire $LOCK_LABEL" || return 1 ;;
    *)
      echo "== waiting for GPU lock ($st)"
      # acquire blocks for up to 30 minutes by itself.
      ssh "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\models\\gpu-lock.ps1 acquire $LOCK_LABEL" || return 1
      ;;
  esac
  LOCKED=1
}
release_lock() {
  ssh "$HOST" "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\models\\gpu-lock.ps1 release $LOCK_LABEL" >/dev/null 2>&1 || true
}
LOCKED=0
cleanup() { if [ "$LOCKED" = 1 ]; then release_lock; fi; }
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------ jobs
mkdir -p bench/out/jobs
JOBS_FAILED=0
JOBS_RAN=0

trim() { echo "$1" | tr -d ' '; }

IFS=',' read -r -a DEPTH_ARR <<< "$(trim "$DEPTHS")"
IFS=',' read -r -a ENGINE_ARR <<< "$(trim "$ENGINES")"

if [ -z "$PROMPTS" ]; then
  PROMPT_ARR=(code_edit json_tool_call prose_explanation longdoc_qa copy_200_words)
else
  IFS=',' read -r -a PROMPT_ARR <<< "$(trim "$PROMPTS")"
fi

# Group cells so one lock acquisition covers several short runs, while every
# group stays comfortably under 20 minutes. 100K is always one cell per hold.
# With --single-hold the whole invocation is one hold; the selection must be
# chosen to fit the 20-minute budget.
if [ "$SINGLE_HOLD" = 1 ]; then
  echo "== single hold: acquiring once for ${DEPTHS} / ${ENGINES}"
  if ! acquire_lock; then die "could not acquire GPU lock"; fi
  single_start=$(date +%s)
fi
for depth in "${DEPTH_ARR[@]}"; do
  if [ "$depth" -ge 32768 ] && [ -z "$PROMPTS" ]; then
    DEPTH_PROMPTS=(code_edit)
  else
    DEPTH_PROMPTS=("${PROMPT_ARR[@]}")
  fi
  case "$depth" in
    32768|100000) group_size=1 ;;
    *) group_size=${#DEPTH_PROMPTS[@]} ;;
  esac
  for engine in "${ENGINE_ARR[@]}"; do
    case "$engine" in hipml|llama) ;; *) die "unknown engine $engine" ;; esac
    i=0
    while [ "$i" -lt "${#DEPTH_PROMPTS[@]}" ]; do
      group=("${DEPTH_PROMPTS[@]:i:group_size}")
      need=0
      for prompt in "${group[@]}"; do
        out="bench/out/jobs/${engine}_${prompt}_${depth}/result.json"
        if [ ! -f "$out" ] || [ "$FORCE" = 1 ]; then
          need=$((need + 1))
        fi
      done
      if [ "$need" -eq 0 ]; then
        echo "== skip $engine ${group[*]}@$depth (results exist)"
        i=$((i + group_size))
        continue
      fi

      echo "== $engine ${group[*]}@$depth"
      if [ "$SINGLE_HOLD" = 0 ]; then
        if ! acquire_lock; then
          echo "   could not acquire GPU lock; skipping group" >&2
          JOBS_FAILED=$((JOBS_FAILED + 1))
          LOCKED=0
          i=$((i + group_size))
          continue
        fi
      fi
      batch_start=$(date +%s)
      for prompt in "${group[@]}"; do
        out="bench/out/jobs/${engine}_${prompt}_${depth}/result.json"
        if [ -f "$out" ] && [ "$FORCE" = 0 ]; then
          echo "== skip $engine $prompt@$depth (result exists)"
          continue
        fi
        reps="$REPS"
        [ "$depth" -ge 32768 ] && reps="$REPS_DEEP"
        [ "$depth" -ge 100000 ] && reps="$REPS_100K"
        extra=""
        if [ "$engine" = hipml ] && [ "$STAGE" = 1 ] && [ "$depth" = 0 ]; then
          extra="--stage-report"
        fi

        echo "== run $engine $prompt@$depth reps=$reps"
        start=$(date +%s)
        ssh "$HOST" "$REMOTE_DIR\\bench\\remote\\with-env.cmd python bench\\job.py --engine $engine --prompt $prompt --depth $depth --reps $reps --model $MODEL --llama-server $LLAMA_SERVER --llama-kv $LLAMA_KV --hipml-kv $HIPML_KV --fixture bench\\fixtures\\prompts.json --out-dir bench\\out\\jobs --tmp bench\\out\\tmp $extra"
        rc=$?
        elapsed=$(( $(date +%s) - start ))
        echo "   ${elapsed}s, exit $rc"
        if [ "$rc" -ne 0 ]; then
          JOBS_FAILED=$((JOBS_FAILED + 1))
        fi
        JOBS_RAN=$((JOBS_RAN + 1))
        if [ "$elapsed" -ge 1080 ]; then
          echo "   WARNING: this cell took ${elapsed}s; keep 100K at one rep" >&2
        fi
      done
      held=$(( $(date +%s) - batch_start ))
      if [ "$SINGLE_HOLD" = 0 ]; then
        echo "   lock hold ${held}s"
        release_lock
        LOCKED=0
      else
        echo "   group ${held}s (single hold continues)"
      fi
      i=$((i + group_size))
    done
  done
done
if [ "$SINGLE_HOLD" = 1 ]; then
  held=$(( $(date +%s) - single_start ))
  echo "   single lock hold ${held}s"
  release_lock
  LOCKED=0
fi

# ------------------------------------------------------------------------ meta
echo "== collect metadata"
HIPML_COMMIT="$(git rev-parse HEAD)"
HIPML_DIRTY="false"
if [ -n "$(git status --porcelain)" ]; then HIPML_DIRTY="true"; fi
HIPML_SHA="$(ssh "$HOST" 'powershell -NoProfile -Command "(Get-FileHash C:\hipml-bench\target\release\hipml.exe -Algorithm SHA256).Hash.ToLower()"' 2>/dev/null | tr -d '\r')"
LLAMA_VERSION="$(ssh "$HOST" "\"$LLAMA_SERVER\" --version" 2>&1 | head -1 | tr -d '\r')"
LLAMA_BUILD="$(echo "$LLAMA_VERSION" | sed -n 's/.*build \([0-9][0-9]*\).*/\1/p')"
LLAMA_COMMIT="$(echo "$LLAMA_VERSION" | sed -n 's/.*commit \([0-9a-f][0-9a-f]*\).*/\1/p')"
GPU_INFO="$(ssh "$HOST" 'powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Format-Table -HideTableHeaders Name,DriverVersion"' 2>/dev/null | grep -i 'R9700' | head -1 | tr -d '\r')"
[ -n "$GPU_INFO" ] || GPU_INFO="$(ssh "$HOST" 'powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Format-Table -HideTableHeaders Name,DriverVersion"' 2>/dev/null | grep -i radeon | tail -1 | tr -d '\r')"
OS_INFO="$(ssh "$HOST" 'powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption"' 2>/dev/null | tr -d '\r')"

mkdir -p bench/out
HIPML_COMMIT="$HIPML_COMMIT" HIPML_DIRTY="$HIPML_DIRTY" HIPML_SHA="$HIPML_SHA" \
LLAMA_VERSION="$LLAMA_VERSION" LLAMA_BUILD="$LLAMA_BUILD" LLAMA_COMMIT="$LLAMA_COMMIT" \
GPU_INFO="$GPU_INFO" OS_INFO="$OS_INFO" HOST_NAME="$HOST" LLAMA_KV="$LLAMA_KV" \
HIPML_KV="$HIPML_KV" MODEL_PATH="$MODEL" LLAMA_SERVER_PATH="$LLAMA_SERVER" \
DEPTHS="$DEPTHS" REPS="$REPS" REPS_DEEP="$REPS_DEEP" REPS_100K="$REPS_100K" \
python3 - <<'PY'
import json, os
meta = {
    "host": os.environ.get("HOST_NAME"),
    "os": os.environ.get("OS_INFO"),
    "gpu": os.environ.get("GPU_INFO"),
    "hipml_commit": os.environ.get("HIPML_COMMIT"),
    "hipml_dirty": os.environ.get("HIPML_DIRTY") == "true",
    "hipml_binary_sha256": os.environ.get("HIPML_SHA"),
    "llama_version": os.environ.get("LLAMA_VERSION"),
    "llama_build": os.environ.get("LLAMA_BUILD"),
    "llama_commit": os.environ.get("LLAMA_COMMIT"),
    "llama_kv": os.environ.get("LLAMA_KV"),
    "hipml_kv": os.environ.get("HIPML_KV"),
    "model_path": os.environ.get("MODEL_PATH"),
    "llama_server_path": os.environ.get("LLAMA_SERVER_PATH"),
    "depths": os.environ.get("DEPTHS"),
    "reps": os.environ.get("REPS"),
    "reps_deep": os.environ.get("REPS_DEEP"),
    "reps_100k": os.environ.get("REPS_100K"),
}
with open("bench/out/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")
print(json.dumps(meta, indent=2))
PY

# ---------------------------------------------------------------- pull + render
echo "== pull raw results"
mkdir -p bench/out
rm -rf bench/out/jobs
ssh "$HOST" 'cmd /c "cd /d C:\hipml-bench\bench\out && tar -cf jobs.tar jobs"' \
  || die "failed to archive job results"
scp -q "$HOST":'C:/hipml-bench/bench/out/jobs.tar' /tmp/hipml-bench-jobs.tar \
  || die "failed to pull job results"
tar -xf /tmp/hipml-bench-jobs.tar -C bench/out \
  || die "failed to extract job results"

python3 bench/aggregate.py \
  --fixture bench/fixtures/prompts.json \
  --jobs bench/out/jobs \
  --meta bench/out/meta.json \
  --results bench/results.json \
  --markdown bench/RESULTS.md \
  || die "aggregate failed"

echo
echo "== done: $JOBS_RAN jobs, $JOBS_FAILED failed"
echo "   bench/results.json"
echo "   bench/RESULTS.md"
[ "$JOBS_FAILED" = 0 ] || exit 1
if [ "$VALIDATE" = 1 ]; then
  python3 bench/validate_results.py || die "results validation failed"
fi
