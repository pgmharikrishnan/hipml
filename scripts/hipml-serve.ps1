<#
.SYNOPSIS
  Keep `hipml serve` running: relaunch it whenever it exits.

.DESCRIPTION
  A context loss, a wedged device or an out-of-memory kill costs one restart
  instead of an outage. The loop is deliberately dumb:

    * start the server, log to a timestamped file;
    * wait for it to exit;
    * back off 2 s, 4 s, 8 s ... 60 s while it keeps dying young, and reset the
      backoff once a run has been up for -StableS seconds (so a crash after an
      hour does not start at 60 s);
    * stop the child and exit when this script is interrupted.

  Every relaunch is a cold process: weights are re-read, the warm-up runs, and
  the prefix cache starts empty. That is the point - it is the same state a
  manual start has, so a restart cannot leave anything half-initialised.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1

.EXAMPLE
  # 8 slots on 16K, log to D:\logs
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1 `
      -Slots 8 -Ctx 16384 -LogDir D:\logs
#>
param(
    [string]$Exe = "C:\hipml-release\target\release\hipml.exe",
    [string]$Gguf = "C:\models\Qwen3.8-27B-Q4_K_M.gguf",
    [string]$Mtp = "C:\models\mtp-Qwen3.8-27B-Q4_0.gguf",
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8093,
    [int]$Slots = 16,
    [int]$Ctx = 32768,
    [int]$Chunk = 256,
    [int]$Drafts = 1,
    [int]$MaxQueue = -1,
    [string]$LogDir = "C:\hipml-release\logs",
    [int]$ReadyTimeoutS = 300,
    [int]$BackoffStartS = 2,
    [int]$BackoffMaxS = 60,
    [int]$StableS = 60
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
$env:HIPML_ARCH = "gfx1201"
if (-not ($env:PATH -split ';' -contains "C:\rocm-sdk\_rocm_sdk_core\bin")) {
    $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;" + $env:PATH
}

$serverArgs = @(
    "serve", "--gguf", $Gguf, "--host", $HostAddr, "--port", "$Port",
    "--ctx", "$Ctx", "--slots", "$Slots", "--chunk", "$Chunk", "--drafts", "$Drafts"
)
if ($Drafts -gt 0 -and (Test-Path $Mtp)) { $serverArgs += @("--mtp", $Mtp) }
if ($MaxQueue -ge 0) { $serverArgs += @("--max-queue", "$MaxQueue") }

$child = $null
$stop = $false
$backoff = $BackoffStartS

function Stop-Child {
    if ($null -ne $script:child -and -not $script:child.HasExited) {
        Write-Host "[supervisor] stopping pid $($script:child.Id)"
        try { $script:child.Kill() } catch { }
        try { $script:child.WaitForExit(30000) | Out-Null } catch { }
    }
}

# Ctrl+C / window close: take the server down with us.
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-Child }

Write-Host "[supervisor] $Exe -> $HostAddr`:$Port, $Slots slots x $Ctx ctx, drafts $Drafts"
Write-Host "[supervisor] logs in $LogDir"

while (-not $stop) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $out = Join-Path $LogDir "serve-$stamp.log"
    $err = Join-Path $LogDir "serve-$stamp.err.log"
    $started = Get-Date
    Write-Host "[supervisor] start $stamp -> $out"
    try {
        $child = Start-Process -FilePath $Exe -ArgumentList $serverArgs -PassThru `
            -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
    } catch {
        Write-Host "[supervisor] could not start: $_"
        Start-Sleep -Seconds $backoff
        $backoff = [Math]::Min($backoff * 2, $BackoffMaxS)
        continue
    }

    # Wait for /health, but do not block the exit wait on it: a server that
    # dies during load must be relaunched, not waited on for 5 minutes.
    $ready = $false
    $deadline = (Get-Date).AddSeconds($ReadyTimeoutS)
    while (-not $ready -and (Get-Date) -lt $deadline -and -not $child.HasExited) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 `
                -Uri "http://$HostAddr`:$Port/health"
            if ($r.StatusCode -eq 200) { $ready = $true }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if ($ready) {
        Write-Host "[supervisor] ready in $([int]((Get-Date) - $started).TotalSeconds)s (pid $($child.Id))"
    } else {
        Write-Host "[supervisor] not healthy within $ReadyTimeoutS s (or exited)"
    }

    $child.WaitForExit()
    $up = ((Get-Date) - $started).TotalSeconds
    $code = $child.ExitCode
    Write-Host "[supervisor] exit code $code after $([int]$up)s"
    if ($up -ge $StableS) { $backoff = $BackoffStartS }
    Write-Host "[supervisor] relaunching in $backoff s"
    Start-Sleep -Seconds $backoff
    $backoff = [Math]::Min($backoff * 2, $BackoffMaxS)
}

Stop-Child
