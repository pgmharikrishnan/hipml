# SPDX-License-Identifier: Apache-2.0
# Sample one process's dedicated GPU memory until it exits; write the peak
# byte count to -Out. WDDM exposes this per process, so it works for both the
# hipml child and llama-server without touching either program's memory.
param(
  [Parameter(Mandatory=$true)][int]$TargetPid,
  [Parameter(Mandatory=$true)][string]$Out,
  [int]$IntervalMs = 250
)

$max = [int64]0
$path = "\GPU Process Memory(pid_${TargetPid}_*)\Dedicated Usage"
while (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
  try {
    $samples = (Get-Counter -Counter $path -ErrorAction Stop).CounterSamples
    $sum = ($samples | Measure-Object -Property CookedValue -Sum).Sum
    if ($sum -gt $max) { $max = [int64]$sum }
  } catch {
    # The process may not have a GPU allocation yet; keep polling.
  }
  Start-Sleep -Milliseconds $IntervalMs
}
Set-Content -Path $Out -Value $max
