# The final interleaved A/B: the base tree against the slice's tree, in one lock
# hold, 3 reps x 2 processes, plus the id streams and the norm probe.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ab-final.ps1 -Base <dir> [-Tree <dir>] [-Models <dir>]
#
# `-Base` is the checkout of the parent commit and `-Tree` the checkout under
# test (default: this one). `-Models` holds the GGUFs and `gpu-lock.ps1`
# (default `$env:HIPML_MODELS`). Both binaries alternate inside the same window,
# so a clock ramp cannot favour either. Each run is a fresh process (the engine
# is rebuilt and the model re-uploaded), which is what "2 processes" means here:
# the spread across processes is the session-to-session noise the delta has to
# beat.
param(
  [string]$Base = "",
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Models = $env:HIPML_MODELS,
  [string]$Gguf = "Qwen3.8-27B-UD-Q4_K_M.gguf"
)
$ErrorActionPreference = "Continue"
if (-not $Base) { Write-Error "pass -Base <checkout of the parent commit>"; exit 2 }
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
$env:HIPML_MTP_GGUF = ""
$gguf = Join-Path $Models $Gguf
$lock = Join-Path $Models "gpu-lock.ps1"
$log = Join-Path $Models "ab-final.log"
$wantLen = 16564259744
$wantSha = "1BCE504DFECD84153666928A41D8A66706F0158D3A317BC5196B72819D7DC529"

"==== ab-final $(Get-Date -Format o)" | Out-File -Append $log
"ambient HIPML_GGUF=$env:HIPML_GGUF (ignored, path passed literally)" | Out-File -Append $log
$len = (Get-Item $gguf).Length
$sha = (Get-FileHash $gguf -Algorithm SHA256).Hash
if ($len -ne $wantLen -or $sha -ne $wantSha) { "MODEL CHECK FAILED $len $sha" | Out-File -Append $log; exit 8 }
"model sha256 $sha $len B OK" | Out-File -Append $log

$acq = & $lock acquire decode-ablate
$acq | Out-File -Append $log
if ($acq -notmatch "ACQUIRED") { "no lock, exiting" | Out-File -Append $log; exit 9 }
try {
  $env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
  $env:HIPML_ARCH = "gfx1201"
  $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;$env:PATH"

  # A binary built without `--features hip` answers every GPU command with
  # "build with --features hip" and exits 0, which would silently produce empty
  # cells. Check both trees actually have the device before timing anything.
  $trees = @{ base = $Base; new = $Tree }
  foreach ($tag in "base", "new") {
    $p = & (Join-Path $trees[$tag] "target\release\hipml.exe") probe 2>&1 | Out-String
    if ($p -notmatch "built with hip feature: true") {
      "BUILD CHECK FAILED ($tag): $p" | Out-File -Append $log
      exit 7
    }
  }

  # The norm kernel on its own, from each tree's own source.
  "---- norm probe base" | Out-File -Append $log
  Set-Location (Join-Path $Base "scripts\probes")
  hipcc -O3 --offload-arch=$env:HIPML_ARCH -I (Join-Path $Base "crates\hipml-kernels\kernels") probe_norm.cpp -o (Join-Path $Base "probe_norm.exe") 2>&1 | Select-Object -Last 3 | Out-File -Append $log
  & (Join-Path $Base "probe_norm.exe") 5120 129 20 2>&1 | Out-File -Append $log
  "---- norm probe new" | Out-File -Append $log
  Set-Location (Join-Path $Tree "scripts\probes")
  hipcc -O3 --offload-arch=$env:HIPML_ARCH -I (Join-Path $Tree "crates\hipml-kernels\kernels") probe_norm.cpp -o (Join-Path $Tree "probe_norm.exe") 2>&1 | Select-Object -Last 3 | Out-File -Append $log
  & (Join-Path $Tree "probe_norm.exe") 5120 129 20 2>&1 | Out-File -Append $log

  # Greedy ids and a 200-token stream, both trees, both depths.
  foreach ($tag in "base", "new") {
    $tree = $trees[$tag]; $exe = Join-Path $tree "target\release\hipml.exe"
    "---- $tag greedy 12 ids" | Out-File -Append $log
    & $exe generate --gguf $gguf --ids 760,6511,314,9338,369 --tokens 12 --device --chunk 2048 2>&1 | Out-File -Append $log
    foreach ($d in 0, 8192) {
      "---- $tag ids depth $d" | Out-File -Append $log
      & $exe gpu-bench --gguf $gguf --depth $d --tokens 200 --kv q8 --ids-out (Join-Path $tree "ids-$tag-$d.txt") 2>&1 | Out-File -Append $log
    }
  }

  # The interleaved cells: 2 processes x 3 reps x {base, new} x {0, 8192}.
  foreach ($proc in 1, 2) {
    foreach ($rep in 1, 2, 3) {
      foreach ($d in 0, 8192) {
        foreach ($tag in "base", "new") {
          $tree = $trees[$tag]
          $exe = Join-Path $tree "target\release\hipml.exe"
          $line = "p$proc r$rep d$d $tag " + (& $exe gpu-bench --gguf $gguf --depth $d --tokens 200 --kv q8 2>&1 | Select-String -Pattern "ms/token" | Select-Object -First 1)
          $line | Out-File -Append $log
        }
      }
    }
  }

  # The attribution on the new tree only, so the note's table and the shipped
  # binary are the same build.
  foreach ($d in 0, 8192) {
    "---- decode-ablate d$d (new)" | Out-File -Append $log
    & (Join-Path $Tree "target\release\hipml.exe") decode-ablate --gguf $gguf --depth $d --tokens 200 --reps 3 --kv q8 --cells full,gemv_only,gemv+norm,-norm 2>&1 | Out-File -Append $log
  }
} finally {
  & $lock release decode-ablate | Out-File -Append $log
  "AB DONE" | Out-File -Append (Join-Path $Models "ab-final.done")
}
