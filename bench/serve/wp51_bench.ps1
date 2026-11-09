# Run one serve_bench cell under the GPU lock, releasing it even on failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_bench.ps1 -Cell b
#
# Results land in bench/serve/results/wp51/hipml_<cell>.json so the earlier
# baselines in bench/serve/results stay untouched.
param(
    [string]$Cell = "b",
    [string]$Engine = "hipml",
    [string]$OutDir = "C:\hipml-release\bench\serve\results\wp51"
)

$lock = "C:\models\gpu-lock.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire wp51
$code = 1
try {
    & python "C:\hipml-release\bench\serve\serve_bench.py" --engine $Engine `
        --cell $Cell --out-dir $OutDir
    $code = $LASTEXITCODE
}
finally {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release wp51
}
exit $code
