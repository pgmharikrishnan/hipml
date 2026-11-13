# Detach the ablation sweep so an ssh drop cannot kill it.
#
#   powershell -NoProfile -File ablate-launch.ps1 [-Models <dir>]
param(
  [string]$Models = $env:HIPML_MODELS,
  [string]$Tree = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)
$ErrorActionPreference = "Continue"
if (-not $Models) { Write-Error "pass -Models or set HIPML_MODELS"; exit 2 }
Start-Process powershell -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', (Join-Path $Tree "scripts\probes\ablate-sweep.ps1"),
  '-Models', $Models,
  '-Tree', $Tree
) -WindowStyle Hidden
Start-Sleep 2
& (Join-Path $Models "gpu-lock.ps1") status
