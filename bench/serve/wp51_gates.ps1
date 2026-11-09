# Run the WP51 regression gates under the GPU lock, releasing it even on
# failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_gates.ps1
#
# oracle-gate 20/20, parity --deep, the tokenizer fixture, and review 4's crash
# plan against the new binary (the malformed-input cases must still be a 400 and
# the server must stay up).
param(
    [string]$Which = "all",
    [string]$Exe = "C:\hipml-release\target\release\hipml.exe",
    [string]$Gguf = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Res = "C:\hipml-release\bench\serve\results\wp51"
)

$lock = "C:\models\gpu-lock.ps1"
$env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
$env:HIPML_ARCH = "gfx1201"
if (-not ($env:PATH -split ';' -contains "C:\rocm-sdk\_rocm_sdk_core\bin")) {
    $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;" + $env:PATH
}
New-Item -ItemType Directory -Force -Path $Res | Out-Null

& powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire wp51
try {
    if ($Which -in @("all", "oracle")) {
        Write-Host "=== oracle-gate ==="
        & $Exe oracle-gate --gguf $Gguf --ids-file "C:\hipml-release\bench\fixtures\p2_math_169.ids" --chunk 2048
        Write-Host "oracle exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "parity")) {
        Write-Host "=== parity --deep ==="
        & $Exe parity --deep
        Write-Host "parity exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "headers")) {
        Write-Host "=== exactness headers ==="
        & python "C:\hipml-release\bench\serve\wp51_probe.py" --hipml $Exe --plan headers `
            --out "$Res\wp51.jsonl" --tag headers
        Write-Host "headers exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "supervisor")) {
        Write-Host "=== supervisor restart ==="
        & powershell -NoProfile -ExecutionPolicy Bypass -File "C:\hipml-release\bench\serve\wp51_supervisor_test.ps1"
        Write-Host "supervisor exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "crash")) {
        Write-Host "=== review4 crash plan ==="
        & python "C:\hipml-release\bench\serve\review4_probe.py" --hipml $Exe --plan crash `
            --ctx 4096 --slots 2 --drafts 1 --out "$Res\review4_crash.jsonl" --tag wp51
        Write-Host "crash exit $LASTEXITCODE"
    }
}
finally {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release wp51
}

if ($Which -in @("all", "tok")) {
    Write-Host "=== tokenizer fixture (no GPU) ==="
    Push-Location "C:\hipml-release"
    & cargo test -p hipml-core --test tokenizer_fixture
    Write-Host "tokenizer exit $LASTEXITCODE"
    Pop-Location
}
