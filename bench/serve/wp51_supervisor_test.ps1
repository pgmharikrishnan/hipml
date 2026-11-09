# Prove the supervisor restarts a dead server.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File wp51_supervisor_test.ps1
#
# Starts scripts/hipml-serve.ps1 on port 8093, waits for /health, kills the
# child (which is what a context loss or an OOM kill looks like from outside),
# and checks that a *new* pid answers /health within the backoff window. Then it
# stops the supervisor. The caller holds the GPU lock.
param(
    [int]$Port = 8093,
    [string]$LogDir = "C:\hipml-release\logs\supervisor-test",
    [int]$TimeoutS = 300
)

$ErrorActionPreference = "Continue"
$sup = $null
$fail = 0

function Wait-Health([int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://127.0.0.1:$Port/health"
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    return $false
}

function Child-Pid {
    (Get-Process hipml -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -like "*hipml-release*" } |
        Select-Object -First 1).Id
}

try {
    if (Wait-Health 1) { Write-Host "port $Port already serving; refusing to run"; exit 2 }
    $sup = Start-Process powershell -PassThru -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", "C:\hipml-release\scripts\hipml-serve.ps1",
        "-LogDir", $LogDir, "-Port", "$Port", "-Slots", "4", "-Ctx", "4096", "-Drafts", "0"
    )
    if (-not (Wait-Health $TimeoutS)) { Write-Host "FAIL: never became healthy"; exit 1 }
    $first = Child-Pid
    Write-Host "healthy, child pid $first"

    # A context loss: the server process disappears.
    Stop-Process -Id $first -Force
    Write-Host "killed child $first; waiting for a relaunch"
    Start-Sleep -Seconds 1

    $deadline = (Get-Date).AddSeconds($TimeoutS)
    $second = $null
    while ((Get-Date) -lt $deadline) {
        $p = Child-Pid
        if ($p -and $p -ne $first -and (Wait-Health 2)) { $second = $p; break }
        Start-Sleep -Seconds 1
    }
    if (-not $second) {
        Write-Host "FAIL: no relaunch"
        $fail = 1
    } else {
        Write-Host "relaunched as pid $second and serving"
        # And it is a working server, not just a listening socket.
        $body = '{"prompt":[1,2,3,4],"n_predict":2,"ignore_eos":true}'
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 60 -Method POST `
            -Uri "http://127.0.0.1:$Port/completion" -Body $body -ContentType "application/json"
        Write-Host "completion after restart: HTTP $($r.StatusCode)"
        if ($r.StatusCode -ne 200) { $fail = 1 }
    }
}
finally {
    if ($sup) { Stop-Process -Id $sup.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
    $p = Child-Pid
    if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }
    Write-Host "supervisor stopped"
}
exit $fail
