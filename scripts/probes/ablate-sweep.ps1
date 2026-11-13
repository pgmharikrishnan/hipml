# The whole decode-ablation sweep under one lock hold.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ablate-sweep.ps1 [-Models <dir>] [-Tree <dir>]
#
# `-Models` is the directory that holds the GGUFs and `gpu-lock.ps1`, and
# defaults to `$env:HIPML_MODELS`; `-Tree` defaults to the checkout this script
# lives in. One acquire for every run, one release in a finally block, so a
# failing cell cannot leave the card held. Each run appends to its own log; the
# model sha is checked once, before the lock, and the ambient
# HIPML_GGUF/HIPML_MTP_GGUF are printed and cleared because a stale ambient value
# silently wins otherwise.
param(
  [string]$Models = $env:HIPML_MODELS,
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Gguf = "Qwen3.8-27B-UD-Q4_K_M.gguf"
)
$ErrorActionPreference = "Continue"
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
$env:HIPML_MTP_GGUF = ""
$exe = Join-Path $Tree "target\release\hipml.exe"
$gguf = Join-Path $Models $Gguf
$lock = Join-Path $Models "gpu-lock.ps1"
$log0 = Join-Path $Models "ablate-d0.log"
$log8 = Join-Path $Models "ablate-d8192.log"
$logG = Join-Path $Models "ablate-graph.log"
$logC = Join-Path $Models "ablate-ctx.log"
$logN = Join-Path $Models "norm-probe.log"
$wantLen = 16564259744
$wantSha = "1BCE504DFECD84153666928A41D8A66706F0158D3A317BC5196B72819D7DC529"

$len = (Get-Item $gguf).Length
if ($len -ne $wantLen) { "MODEL LENGTH MISMATCH $len" | Out-File -Append $log0; exit 8 }
$sha = (Get-FileHash $gguf -Algorithm SHA256).Hash
if ($sha -ne $wantSha) { "MODEL SHA MISMATCH $sha" | Out-File -Append $log0; exit 8 }

$acq = & $lock acquire decode-ablate
$acq | Out-File -Append $log0
if ($acq -notmatch "ACQUIRED") { "no lock, exiting" | Out-File -Append $log0; exit 9 }
try {
  "model sha256 $sha $len B OK" | Out-File -Append $log0
  "ambient HIPML_GGUF=$env:HIPML_GGUF (ignored, path passed literally)" | Out-File -Append $log0
  "ambient HIPML_MTP_GGUF='$env:HIPML_MTP_GGUF' (cleared)" | Out-File -Append $log0

  "---- d0" | Out-File -Append $log0
  & $exe decode-ablate --gguf $gguf --depth 0 --tokens 200 --reps 3 --kv q8 2>&1 | Out-File -Append $log0

  "---- d8192" | Out-File -Append $log8
  & $exe decode-ablate --gguf $gguf --depth 8192 --tokens 200 --reps 3 --kv q8 2>&1 | Out-File -Append $log8

  # The same cells through one captured launch per token. If the isolated GEMV
  # rate does not move, host-side launch spacing is not what separates the
  # in-situ stream from the standalone bench.
  "---- graph d0" | Out-File -Append $logG
  & $exe decode-ablate --gguf $gguf --depth 0 --tokens 200 --reps 3 --graph --cells full,gemv_only 2>&1 | Out-File -Append $logG

  # The cache footprint alone: same work, same depth, only the allocation size
  # changes. Separates "the KV bytes are read" from "the KV allocation is there".
  "---- ctx 208" | Out-File -Append $logC
  & $exe decode-ablate --gguf $gguf --depth 0 --ctx 208 --tokens 200 --reps 3 --cells full,gemv_only 2>&1 | Out-File -Append $logC
  "---- ctx 8448" | Out-File -Append $logC
  & $exe decode-ablate --gguf $gguf --depth 0 --ctx 8448 --tokens 200 --reps 3 --cells full,gemv_only 2>&1 | Out-File -Append $logC

  # The norm on its own, from the tree's own kernel source.
  "---- norm probe" | Out-File -Append $logN
  $env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
  $env:HIPML_ARCH = "gfx1201"
  $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;$env:PATH"
  Set-Location (Join-Path $Tree "scripts\probes")
  hipcc -O3 --offload-arch=$env:HIPML_ARCH probe_norm.cpp -o (Join-Path $Tree "probe_norm.exe") 2>&1 |
    Select-Object -Last 4 | Out-File -Append $logN
  & (Join-Path $Tree "probe_norm.exe") 5120 129 20 2>&1 | Out-File -Append $logN
} finally {
  & $lock release decode-ablate | Out-File -Append $log0
  "SWEEP DONE" | Out-File -Append (Join-Path $Models "ablate-sweep.done")
}
