# Run one WP51 probe plan under the optional GPU lock, releasing it even on
# failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_run.ps1 isolation
#
# `acquire` blocks until the lock is free (30 min cap) and steals only a lock
# that is more than 45 min stale. The release runs from `finally`, so a failed
# probe or a Ctrl+C does not leave the GPU held.
#
# Environment: HIPML_EXE, HIPML_GGUF, HIPML_MTP_GGUF, HIPML_GPU_LOCK (optional),
# HIPML_LOCK_LABEL (optional).
param(
    [Parameter(Mandatory = $true)][string]$Plan,
    [string]$Tag = "",
    [string]$Exe = $env:HIPML_EXE,
    [string]$Out = ""
)

$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
if (-not $Exe) { $Exe = Join-Path $RepoRoot 'target\release\hipml.exe' }
if (-not $Out) { $Out = Join-Path $RepoRoot 'bench\serve\results\wp51.jsonl' }
if (-not $Tag) { $Tag = $Plan }

$lock = $env:HIPML_GPU_LOCK
$label = if ($env:HIPML_LOCK_LABEL) { $env:HIPML_LOCK_LABEL } else { "wp51" }

if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire $label }
$code = 1
try {
    & python (Join-Path $RepoRoot 'bench\serve\wp51_probe.py') --hipml $Exe --plan $Plan --out $Out --tag $Tag
    $code = $LASTEXITCODE
}
finally {
    if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release $label }
}
exit $code
