<#
  Kaizen Eye 2 APK audit - run at every gate.
  Checks: arm64-only native libs, 16 KB ELF alignment, expected-lib allow-list, merged-manifest permissions
  (no INTERNET / ACCESS_NETWORK_STATE), required <uses-native-library> entries, extractNativeLibs, zipalign -P 16, APK size.
  Usage:  powershell -File native\tools\apk-audit.ps1 -Apk native\app\build\outputs\apk\debug\app-debug.apk
  Exit code 0 = PASS, 1 = at least one FAIL.
#>
param(
    [Parameter(Mandatory = $true)] [string]$Apk,
    [string]$ExpectedLibs = '',
    [switch]$AllowInternet,
    [int]$MaxApkMB = 250
)
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ExpectedLibs) { $ExpectedLibs = Join-Path $scriptDir 'expected-libs.txt' }
$bt = 'D:\AndroidSdk\build-tools\36.1.0'
$Apk = (Resolve-Path $Apk).Path
Add-Type -AssemblyName System.IO.Compression
$fails = New-Object System.Collections.Generic.List[string]
$warns = New-Object System.Collections.Generic.List[string]

function Get-Elf16k([byte[]]$b) {
    if ($b.Length -lt 64 -or $b[0] -ne 0x7F -or $b[1] -ne 0x45) { return 'notELF' }
    if ($b[4] -ne 2) { return 'elf32' }   # Hexagon DSP skel libraries are 32-bit ELF; loaded by the DSP, not the Android linker
    $phoff = [BitConverter]::ToUInt64($b, 0x20); $phentsize = [BitConverter]::ToUInt16($b, 0x36); $phnum = [BitConverter]::ToUInt16($b, 0x38)
    $minAlign = [UInt64]::MaxValue
    for ($i = 0; $i -lt $phnum; $i++) {
        $o = [int]($phoff + $i * $phentsize)
        if ([BitConverter]::ToUInt32($b, $o) -eq 1) { $al = [BitConverter]::ToUInt64($b, $o + 48); if ($al -lt $minAlign) { $minAlign = $al } }
    }
    if ($minAlign -ge 16384) { return '16K-ok' } else { return "4K-ONLY($minAlign)" }
}

$apkFile = Get-Item $Apk
$apkMB = $apkFile.Length / 1MB
Write-Host ("APK: {0}  ({1:N1} MB)" -f $apkFile.Name, $apkMB)
if ($apkMB -gt $MaxApkMB) { $fails.Add(("APK size {0:N0} MB exceeds {1} MB" -f $apkMB, $MaxApkMB)) }

# ---- native libs -------------------------------------------------------------------------------------------
$bytes = [IO.File]::ReadAllBytes($Apk)
$zip = New-Object IO.Compression.ZipArchive((New-Object IO.MemoryStream(, $bytes)))
$abis = @{}
$arm64 = @()
foreach ($e in $zip.Entries) {
    if ($e.FullName -match '^lib/([^/]+)/([^/]+\.so)$') {
        $abi = $Matches[1]; $name = $Matches[2]
        if (-not $abis.ContainsKey($abi)) { $abis[$abi] = 0 }
        $abis[$abi] += $e.Length
        if ($abi -eq 'arm64-v8a') {
            $s = $e.Open(); $m = New-Object IO.MemoryStream; $s.CopyTo($m); $s.Close()
            $arm64 += [pscustomobject]@{ Name = $name; Size = $e.Length; Elf = (Get-Elf16k $m.ToArray()) }
        }
    }
}
Write-Host "ABIs with native libs: $(($abis.GetEnumerator() | ForEach-Object { '{0}={1:N1}MB' -f $_.Key, ($_.Value / 1MB) }) -join ', ')"
foreach ($abi in $abis.Keys) { if ($abi -ne 'arm64-v8a') { $fails.Add("unexpected ABI in APK: $abi") } }
Write-Host ""
Write-Host "arm64-v8a libs:"
foreach ($l in ($arm64 | Sort-Object Name)) {
    Write-Host ("  {0,-40} {1,12:N0} B  {2}" -f $l.Name, $l.Size, $l.Elf)
    if ($l.Elf -like '4K-ONLY*') { $fails.Add("$($l.Name) is not 16 KB aligned ($($l.Elf))") }
}
$zip.Dispose()

if (Test-Path $ExpectedLibs) {
    $expected = Get-Content $ExpectedLibs | Where-Object { $_ -and -not $_.StartsWith('#') } | ForEach-Object { $_.Trim() }
    $have = $arm64 | ForEach-Object { $_.Name }
    foreach ($n in $have) { if ($expected -notcontains $n) { $fails.Add("unexpected native lib (not in expected-libs.txt): $n") } }
    foreach ($n in $expected) { if ($have -notcontains $n) { $warns.Add("expected native lib missing: $n") } }
} else {
    $prop = Join-Path $scriptDir 'expected-libs.proposed.txt'
    ($arm64 | Sort-Object Name | ForEach-Object { $_.Name }) | Set-Content $prop -Encoding UTF8
    $warns.Add("no expected-libs.txt yet; wrote a proposal to $prop (review it, then rename to expected-libs.txt)")
}

# ---- manifest ----------------------------------------------------------------------------------------------
$perms = & "$bt\aapt2.exe" dump permissions $Apk 2>&1 | Where-Object { $_ -match "^uses-permission" } | ForEach-Object { ($_ -replace ".*name='([^']+)'.*", '$1') }
Write-Host ""
Write-Host "permissions: $($perms -join ', ')"
# Explicit allow-list: anything else in the merged manifest is a FAIL (the offline / cannot-phone-home guarantee).
$allowed = @('android.permission.CAMERA', 'android.permission.VIBRATE')
if ($AllowInternet) { $allowed += @('android.permission.INTERNET', 'android.permission.ACCESS_NETWORK_STATE') }
foreach ($p in $perms) {
    if ($allowed -contains $p) { continue }
    if ($p -like '*.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION') { continue }   # AndroidX core's own signature permission
    $fails.Add("unexpected permission in merged manifest: $p")
}

$tree = (& "$bt\aapt2.exe" dump xmltree --file AndroidManifest.xml $Apk 2>&1) -join "`n"
$nativeLibs = [regex]::Matches($tree, 'E: uses-native-library[^\n]*\n\s*(?:[^\n]*\n)*?\s*A: http://schemas.android.com/apk/res/android:name\([^)]*\)="([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
if (-not $nativeLibs) {
    $nativeLibs = [regex]::Matches($tree, 'uses-native-library[\s\S]{0,200}?name\([^)]*\)="([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
}
Write-Host "uses-native-library: $((($nativeLibs | Select-Object -Unique) -join ', '))"
foreach ($need in 'libcdsprpc.so', 'libOpenCL.so') { if (($nativeLibs | Select-Object -Unique) -notcontains $need) { $fails.Add("merged manifest lacks <uses-native-library> $need") } }
$ext = [regex]::Match($tree, 'extractNativeLibs[^\n]*\)=(?:\(type \w+\))?(\w+)').Groups[1].Value
Write-Host "extractNativeLibs: $ext"
if ($ext -and $ext -notmatch '^(true|0xffffffff|-1)$') { $fails.Add("extractNativeLibs is '$ext' (Hexagon skel libs need real files on disk)") }

# ---- zipalign 16 KB ----------------------------------------------------------------------------------------
$za = & "$bt\zipalign.exe" -c -P 16 -v 4 $Apk 2>&1
if ($LASTEXITCODE -ne 0) { $fails.Add("zipalign -c -P 16 failed (exit $LASTEXITCODE)") } else { Write-Host "zipalign -P 16 check: OK" }

# ---- verdict -----------------------------------------------------------------------------------------------
Write-Host ""
foreach ($w in $warns) { Write-Host "WARN  $w" -ForegroundColor Yellow }
foreach ($f in $fails) { Write-Host "FAIL  $f" -ForegroundColor Red }
if ($fails.Count -eq 0) { Write-Host "AUDIT PASS" -ForegroundColor Green; exit 0 } else { Write-Host "AUDIT FAIL ($($fails.Count))" -ForegroundColor Red; exit 1 }
