# Decode ablation runner. Acquires the GPU lock around every run and releases it
# in a finally block, so a failing probe cannot leave the card held.
#
#   powershell -NoProfile -File ablate-run.ps1 -Tag smoke -Cmd "decode-ablate --cells full --tokens 40 --reps 1"
#
# `-Cmd` is the hipml argument line; the executable is `-Exe`, which defaults to
# the release build of the checkout this script lives in. `-Models` is the
# directory holding the GGUF and `gpu-lock.ps1` (default `$env:HIPML_MODELS`).
# The GGUF path is passed literally in `-Cmd` because a stale ambient
# `HIPML_GGUF` would otherwise win, and `HIPML_MTP_GGUF` is cleared here so no
# subcommand can pick it up.
param(
  [string]$Tag = "run",
  [string]$Models = $env:HIPML_MODELS,
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Exe = "",
  [string]$Log = "",
  [string]$Label = "decode-ablate",
  [string]$Gguf = "Qwen3.8-27B-UD-Q4_K_M.gguf",
  [string]$Cmd = ""
)
$ErrorActionPreference = "Continue"
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
if (-not $Exe) { $Exe = Join-Path $Tree "target\release\hipml.exe" }
if (-not $Log) { $Log = Join-Path $Models "ablate.log" }
$env:HIPML_MTP_GGUF = ""
$gguf = Join-Path $Models $Gguf
$lock = Join-Path $Models "gpu-lock.ps1"
$wantLen = 16564259744
$wantSha = "1BCE504DFECD84153666928A41D8A66706F0158D3A317BC5196B72819D7DC529"

"==== $Tag start $(Get-Date -Format o)" | Out-File -Append $Log
"exe: $Exe" | Out-File -Append $Log
"cmd: $Cmd" | Out-File -Append $Log
"ambient HIPML_GGUF=$env:HIPML_GGUF (ignored: the path is passed literally)" | Out-File -Append $Log
"ambient HIPML_MTP_GGUF='$env:HIPML_MTP_GGUF' (cleared)" | Out-File -Append $Log
"ambient HIPML_ABLATE='$env:HIPML_ABLATE'" | Out-File -Append $Log
$len = (Get-Item $gguf).Length
if ($len -ne $wantLen) { "MODEL LENGTH MISMATCH $len != $wantLen" | Out-File -Append $Log; exit 8 }
$sha = (Get-FileHash $gguf -Algorithm SHA256).Hash
if ($sha -ne $wantSha) { "MODEL SHA MISMATCH $sha" | Out-File -Append $Log; exit 8 }
"model sha256 $sha $len B OK" | Out-File -Append $Log

$acq = & $lock acquire $Label
$acq | Out-File -Append $Log
if ($acq -notmatch "ACQUIRED") { "no lock, exiting" | Out-File -Append $Log; exit 9 }
try {
  $argv = $Cmd.Split(' ') | Where-Object { $_ -ne "" }
  & $Exe @argv 2>&1 | Out-File -Append $Log
  "exit $LASTEXITCODE" | Out-File -Append $Log
} finally {
  & $lock release $Label | Out-File -Append $Log
  "==== $Tag end $(Get-Date -Format o)" | Out-File -Append $Log
}
