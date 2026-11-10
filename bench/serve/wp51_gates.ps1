# SPDX-License-Identifier: Apache-2.0
# Run the WP51 regression gates under the optional GPU lock, releasing it even
# on failure.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_gates.ps1
#
# oracle-gate 20/20, parity --deep, the tokenizer fixture, and review 4's crash
# plan against the new binary (the malformed-input cases must still be a 400 and
# the server must stay up).
#
# Environment: HIPML_EXE, HIPML_GGUF, HIP_PATH (required), HIPML_GPU_LOCK
# (optional), HIPML_LOCK_LABEL (optional).
param(
    [string]$Which = "all",
    [string]$Exe = $env:HIPML_EXE,
    [string]$Gguf = $env:HIPML_GGUF,
    [string]$Res = ""
)

$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
if (-not $Exe) { $Exe = Join-Path $RepoRoot 'target\release\hipml.exe' }
if (-not $Gguf) { throw "set HIPML_GGUF (or pass -Gguf) to the target GGUF" }
if (-not $env:HIP_PATH) { throw "set HIP_PATH to your ROCm SDK root" }
if (-not $Res) { $Res = Join-Path $RepoRoot 'bench\serve\results\wp51' }

if (-not $env:HIPML_ARCH) { $env:HIPML_ARCH = "gfx1201" }
$hipBin = Join-Path $env:HIP_PATH "bin"
if (-not (($env:PATH -split ';') -contains $hipBin)) {
    $env:PATH = "$hipBin;$env:PATH"
}
New-Item -ItemType Directory -Force -Path $Res | Out-Null

$lock = $env:HIPML_GPU_LOCK
$label = if ($env:HIPML_LOCK_LABEL) { $env:HIPML_LOCK_LABEL } else { "wp51" }
$fixture = Join-Path $RepoRoot 'bench\fixtures\p2_math_169.ids'
$probe = Join-Path $RepoRoot 'bench\serve\wp51_probe.py'

if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock acquire $label }
try {
    if ($Which -in @("all", "oracle")) {
        Write-Host "=== oracle-gate ==="
        & $Exe oracle-gate --gguf $Gguf --ids-file $fixture --chunk 2048
        Write-Host "oracle exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "parity")) {
        Write-Host "=== parity --deep ==="
        & $Exe parity --deep
        Write-Host "parity exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "headers")) {
        Write-Host "=== exactness headers ==="
        & python $probe --hipml $Exe --plan headers `
            --out (Join-Path $Res 'wp51.jsonl') --tag headers
        Write-Host "headers exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "supervisor")) {
        Write-Host "=== supervisor restart ==="
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'bench\serve\wp51_supervisor_test.ps1')
        Write-Host "supervisor exit $LASTEXITCODE"
    }
    if ($Which -in @("all", "crash")) {
        Write-Host "=== review4 crash plan ==="
        & python (Join-Path $RepoRoot 'bench\serve\review4_probe.py') --hipml $Exe --plan crash `
            --ctx 4096 --slots 2 --drafts 1 --out (Join-Path $Res 'review4_crash.jsonl') --tag wp51
        Write-Host "crash exit $LASTEXITCODE"
    }
}
finally {
    if ($lock) { & powershell -NoProfile -ExecutionPolicy Bypass -File $lock release $label }
}

if ($Which -in @("all", "tok")) {
    Write-Host "=== tokenizer fixture (no GPU) ==="
    Push-Location $RepoRoot
    & cargo test -p hipml-core --test tokenizer_fixture
    Write-Host "tokenizer exit $LASTEXITCODE"
    Pop-Location
}
