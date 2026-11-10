# SPDX-License-Identifier: Apache-2.0
# Run one serve_bench cell under the optional GPU lock, releasing it even on
# failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_bench.ps1 -Cell b
#
# Results land in bench/serve/results/wp51/hipml_<cell>.json so the earlier
# baselines in bench/serve/results stay untouched.
#
# Environment: HIPML_GPU_LOCK (optional lock script), HIPML_GGUF, HIPML_EXE,
# HIP_PATH. Nothing here is machine-specific.
param(
    [string]$Cell = "b",
    [string]$Engine = "hipml",
    [string]$OutDir = ""
)

$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot 'bench\serve\results\wp51' }
$lock = $env:HIPML_GPU_LOCK
$label = if ($env:HIPML_LOCK_LABEL) { $env:HIPML_LOCK_LABEL } else { "wp51" }

if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire $label }
$code = 1
try {
    & python (Join-Path $RepoRoot 'bench\serve\serve_bench.py') --engine $Engine `
        --cell $Cell --out-dir $OutDir
    $code = $LASTEXITCODE
}
finally {
    if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release $label }
}
exit $code
