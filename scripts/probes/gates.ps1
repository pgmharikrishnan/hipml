# Every invariant the card requires, on one tree, under one lock hold.
#
#   powershell -NoProfile -File gates.ps1 -Tag after [-Tree <dir>] [-Models <dir>]
#
# `-Models` holds the GGUFs and `gpu-lock.ps1` (default `$env:HIPML_MODELS`); the
# stock file is passed literally as the second oracle case. The ambient
# HIPML_GGUF on a box that has one points at a different file, which is exactly
# why nothing here relies on the environment.
param(
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Models = $env:HIPML_MODELS,
  [string]$Tag = "after",
  [string]$Log = ""
)
$ErrorActionPreference = "Continue"
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
if (-not $Log) { $Log = Join-Path $Models "ablate-gates.log" }
$env:HIPML_MTP_GGUF = ""
$exe = Join-Path $Tree "target\release\hipml.exe"
$ud = Join-Path $Models "Qwen3.8-27B-UD-Q4_K_M.gguf"
$stock = Join-Path $Models "Qwen3.8-27B-Q4_K_M.gguf"
$mtp = Join-Path $Models "mtp-Qwen3.8-27B-Q4_0.gguf"
$lock = Join-Path $Models "gpu-lock.ps1"
$ids = Join-Path $Tree "bench\fixtures\p2_math_169.ids"

"==== $Tag $Tree $(Get-Date -Format o)" | Out-File -Append $Log
"ambient HIPML_GGUF=$env:HIPML_GGUF (ignored, paths passed literally)" | Out-File -Append $Log
$acq = & $lock acquire gates
$acq | Out-File -Append $Log
if ($acq -notmatch "ACQUIRED") { "no lock, exiting" | Out-File -Append $Log; exit 9 }
try {
  "---- greedy 12 ids, UD" | Out-File -Append $Log
  & $exe generate --gguf $ud --ids 760,6511,314,9338,369 --tokens 12 --device --chunk 2048 2>&1 | Out-File -Append $Log

  "---- 200-token stream, depth 0" | Out-File -Append $Log
  & $exe gpu-bench --gguf $ud --depth 0 --tokens 200 --kv q8 --ids-out (Join-Path $Tree "ids-d0-$Tag.txt") 2>&1 | Out-File -Append $Log
  "---- 200-token stream, depth 8192" | Out-File -Append $Log
  & $exe gpu-bench --gguf $ud --depth 8192 --tokens 200 --kv q8 --ids-out (Join-Path $Tree "ids-d8192-$Tag.txt") 2>&1 | Out-File -Append $Log

  "---- gemv-check" | Out-File -Append $Log
  foreach ($t in "blk.0.ffn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ffn_down.weight", "output.weight") {
    foreach ($r in 1, 2, 4) {
      & $exe gemv-check --gguf $ud --tensor $t --rows $r 2>&1 | Out-File -Append $Log
    }
  }

  "---- column-bench --check q8 cols 2/4, f32 cols 4" | Out-File -Append $Log
  & $exe column-bench --gguf $ud --ids-file $ids --cols 2 --tokens 32 --kv q8 --check 2>&1 | Out-File -Append $Log
  & $exe column-bench --gguf $ud --ids-file $ids --cols 4 --tokens 32 --kv q8 --check 2>&1 | Out-File -Append $Log
  & $exe column-bench --gguf $ud --ids-file $ids --cols 4 --tokens 32 --kv f32 --check 2>&1 | Out-File -Append $Log

  "---- spec-bench drafts 0-3" | Out-File -Append $Log
  foreach ($d in 0, 1, 2, 3) {
    & $exe spec-bench --gguf $ud --mtp $mtp --ids-file $ids --tokens 128 --drafts $d --kv q8 --stop-eos 2>&1 | Out-File -Append $Log
  }

  "---- batch-bench --slots 16 --check" | Out-File -Append $Log
  & $exe batch-bench --gguf $ud --slots 16 --tokens 16 --check 2>&1 | Out-File -Append $Log

  "---- oracle-gate UD" | Out-File -Append $Log
  & $exe oracle-gate --gguf $ud --ids-file $ids --steps 20 --chunk 2048 --kv q8 2>&1 | Out-File -Append $Log
  "---- oracle-gate stock" | Out-File -Append $Log
  & $exe oracle-gate --gguf $stock --ids-file $ids --steps 20 --chunk 2048 --kv q8 2>&1 | Out-File -Append $Log

  "---- parity --deep" | Out-File -Append $Log
  & $exe parity --deep 2>&1 | Select-Object -Last 6 | Out-File -Append $Log
} finally {
  & $lock release gates | Out-File -Append $Log
  "GATES DONE" | Out-File -Append (Join-Path $Models "ablate-gates.done")
}
