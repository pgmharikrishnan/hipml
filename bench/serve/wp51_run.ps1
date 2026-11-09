# Run one WP51 probe plan under the GPU lock, releasing it even on failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_run.ps1 isolation
#
# `acquire` blocks until the lock is free (30 min cap) and steals only a lock
# that is more than 45 min stale. The release runs from `finally`, so a failed
# probe or a Ctrl+C does not leave the GPU held.
param(
    [Parameter(Mandatory = $true)][string]$Plan,
    [string]$Tag = "",
    [string]$Exe = "C:\hipml-release\target\release\hipml.exe",
    [string]$Out = "C:\hipml-release\bench\serve\results\wp51.jsonl"
)

$lock = "C:\models\gpu-lock.ps1"
if (-not $Tag) { $Tag = $Plan }

& powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire wp51
$code = 1
try {
    & python "C:\hipml-release\bench\serve\wp51_probe.py" --hipml $Exe --plan $Plan --out $Out --tag $Tag
    $code = $LASTEXITCODE
}
finally {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release wp51
}
exit $code
