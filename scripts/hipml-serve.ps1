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
  # 8 slots on 16K, logs under <repo>\logs
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1 `
      -Slots 8 -Ctx 16384

.NOTES
  Paths come from the environment, not from this file:

    HIPML_EXE       the hipml binary (default: <repo>\target\release\hipml.exe)
    HIPML_GGUF      the target GGUF (required)
    HIPML_MTP_GGUF  the MTP drafter (optional; --mtp is omitted when unset)
    HIP_PATH        your ROCm SDK root (required; <root>\bin goes on PATH)
    HIPML_ARCH      offload arch (default: gfx1201)

  Logs go to <repo>\logs unless -LogDir says otherwise.
#>
param(
    [string]$Exe = $(if ($env:HIPML_EXE) { $env:HIPML_EXE } else { Join-Path (Split-Path -Parent $PSScriptRoot) 'target\release\hipml.exe' }),
    [string]$Gguf = $env:HIPML_GGUF,
    [string]$Mtp = $env:HIPML_MTP_GGUF,
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8093,
    [int]$Slots = 16,
    [int]$Ctx = 32768,
    [int]$Chunk = 256,
    [int]$Drafts = 1,
    [int]$MaxQueue = -1,
    [string]$LogDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'logs'),
    [int]$ReadyTimeoutS = 300,
    [int]$BackoffStartS = 2,
    [int]$BackoffMaxS = 60,
    [int]$StableS = 60
)

$ErrorActionPreference = "Stop"

if (-not $Gguf) {
    throw "set HIPML_GGUF (or pass -Gguf) to the target GGUF"
}
if (-not $env:HIP_PATH) {
    throw "set HIP_PATH to your ROCm SDK root (the directory that holds bin\)"
}
if ($Drafts -gt 0 -and $Mtp -and -not (Test-Path $Mtp)) {
    throw "HIPML_MTP_GGUF points at a missing file: $Mtp"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not $env:HIPML_ARCH) { $env:HIPML_ARCH = "gfx1201" }
$hipBin = Join-Path $env:HIP_PATH "bin"
if (-not (($env:PATH -split ';') -contains $hipBin)) {
    $env:PATH = "$hipBin;$env:PATH"
}

$serverArgs = @(
    "serve", "--gguf", $Gguf, "--host", $HostAddr, "--port", "$Port",
    "--ctx", "$Ctx", "--slots", "$Slots", "--chunk", "$Chunk", "--drafts", "$Drafts"
)
if ($Drafts -gt 0 -and $Mtp) { $serverArgs += @("--mtp", $Mtp) }
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
