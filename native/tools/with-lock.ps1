<#
  Runs any heavy command under the same machine-wide lock as build.ps1 (e.g. torch / onnx2tf / TensorFlow jobs),
  so a Python export never overlaps a Gradle build on this RAM-limited PC.
  Usage:  powershell -File native\tools\with-lock.ps1 [-Timeout 60] -- D:\Projects\Kaizen_Eye\.venv-convert\Scripts\python.exe tools\dinov2\export.py --out x
  No param() block on purpose: with one, PowerShell tries to bind arguments such as "--out" or "-o" of the WRAPPED command
  to this script's parameters ("parameter name '' is ambiguous"). Everything is read verbatim from $args instead.
#>
$ErrorActionPreference = 'Stop'
$all = @($args)
$timeout = 60
$i = 0
if ($all.Count -ge 2 -and $all[0] -ieq '-Timeout') { $timeout = [int]$all[1]; $i = 2 }
if ($i -lt $all.Count -and $all[$i] -eq '--') { $i++ }
if ($i -ge $all.Count) { throw "no command given" }
$Command = @($all[$i..($all.Count - 1)])
$mutex = New-Object System.Threading.Mutex($false, 'Local\KaizenEyeBuildLock')
Write-Host "[with-lock] waiting for build lock (max $timeout min)..."
$got = $false
try { $got = $mutex.WaitOne([TimeSpan]::FromMinutes($timeout)) } catch [System.Threading.AbandonedMutexException] { $got = $true }
if (-not $got) { throw "Timed out waiting for the Kaizen build lock" }
$code = 1
try {
    Write-Host "[with-lock] running: $($Command -join ' ')"
    $rest = if ($Command.Count -gt 1) { @($Command[1..($Command.Count - 1)]) } else { @() }
    & $Command[0] @rest
    $code = $LASTEXITCODE
} finally {
    $mutex.ReleaseMutex()
}
exit $code
