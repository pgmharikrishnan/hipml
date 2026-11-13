# Build and run the standalone norm probe, under the GPU lock.
#
#   powershell -NoProfile -File norm-probe.ps1 [-Tree <dir>] [-Tag base]
#
# The probe compiles the tree's own `rmsnorm_f32.hip`, so pointing `-Tree` at two
# trees is the A/B between two versions of the kernel. `-Models` is the directory
# holding `gpu-lock.ps1` and the log (default `$env:HIPML_MODELS`).
param(
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
  [string]$Models = $env:HIPML_MODELS,
  [string]$Tag = "base",
  [string]$Log = ""
)
$ErrorActionPreference = "Continue"
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
if (-not $Log) { $Log = Join-Path $Models "norm-probe.log" }
$lock = Join-Path $Models "gpu-lock.ps1"
$env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
$env:HIPML_ARCH = "gfx1201"
$env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;$env:PATH"

"==== $Tag $Tree $(Get-Date -Format o)" | Out-File -Append $Log
$acq = & $lock acquire norm-probe
$acq | Out-File -Append $Log
if ($acq -notmatch "ACQUIRED") { "no lock, exiting" | Out-File -Append $Log; exit 9 }
try {
  Set-Location (Join-Path $Tree "scripts\probes")
  hipcc -O3 --offload-arch=$env:HIPML_ARCH probe_norm.cpp -o (Join-Path $Tree "probe_norm.exe") 2>&1 |
    Select-Object -Last 5 | Out-File -Append $Log
  & (Join-Path $Tree "probe_norm.exe") 5120 129 20 2>&1 | Out-File -Append $Log
} finally {
  & $lock release norm-probe | Out-File -Append $Log
}
