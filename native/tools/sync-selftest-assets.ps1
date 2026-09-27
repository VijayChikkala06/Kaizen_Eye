<#
  Copies the self-test data from <repo>\testdata into the app's assets (bundled into the APK for the on-device self-test):
    testdata\golden_core.json, golden_app.json, golden_twin.json  -> app\src\main\assets\selftest\golden\
    testdata\replay_synth\{frames\,timestamps.json,script.json,expected.json} -> app\src\main\assets\selftest\replay_synth\
  Run it after regenerating the goldens (tools\lab\make_golden_twin.py) or the synthetic clip (tools\lab\make_replay_synth.py).
  Usage:  powershell -File native\tools\sync-selftest-assets.ps1
#>
$ErrorActionPreference = 'Stop'
$native = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$repo = Split-Path -Parent $native
$td = Join-Path $repo 'testdata'
$assets = Join-Path $native 'app\src\main\assets\selftest'
$golden = Join-Path $assets 'golden'
$replay = Join-Path $assets 'replay_synth'
New-Item -ItemType Directory -Force -Path $golden | Out-Null
foreach ($f in 'golden_core.json', 'golden_app.json', 'golden_twin.json') { Copy-Item (Join-Path $td $f) (Join-Path $golden $f) -Force }
if (Test-Path $replay) { Remove-Item $replay -Recurse -Force }
New-Item -ItemType Directory -Force -Path $replay | Out-Null
Copy-Item (Join-Path $td 'replay_synth\frames') (Join-Path $replay 'frames') -Recurse -Force
foreach ($f in 'timestamps.json', 'script.json', 'expected.json') { Copy-Item (Join-Path $td "replay_synth\$f") (Join-Path $replay $f) -Force }
$n = (Get-ChildItem (Join-Path $replay 'frames') -Filter *.jpg).Count
Write-Host "synced: 3 golden files, replay_synth ($n frames) -> $assets"
