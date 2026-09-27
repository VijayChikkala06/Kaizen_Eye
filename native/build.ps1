<#
  Kaizen Eye 2 build wrapper.
  - Sets JAVA_HOME to Android Studio's JBR 21 (JDK 25 is what's on PATH; the Gradle daemon-JVM criteria want 21).
  - Takes a machine-wide mutex so only ONE Gradle build / heavy job runs at a time (this PC has ~15.6 GB RAM, often ~4 GB free).
  Usage:  powershell -File native\build.ps1 :app:assembleDebug :core:test
          powershell -File native\build.ps1 -Timeout 45 :app:assembleRelease
          powershell -File native\build.ps1 -ProjectDir D:\kz-tmp\h1a -LogName h1a.log :core:test     (helper scratch copies)
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Timeout = 40,                       # minutes to wait for the lock (use -Timeout 60)
    [string]$ProjectDir = '',                 # Gradle project to build (default: this native/ folder)
    [string]$LogName = 'last-build.log',      # file name under native\dist\logs (one per helper, so logs never collide)
    [Parameter(ValueFromRemainingArguments = $true)] [string[]]$GradleArgs
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ProjectDir) { $ProjectDir = $here }
$ProjectDir = (Resolve-Path $ProjectDir).Path
if (-not (Test-Path (Join-Path $ProjectDir 'gradlew.bat'))) { throw "no gradlew.bat in $ProjectDir" }

$jbr = 'C:\Program Files\Android\Android Studio\jbr'
if (-not (Test-Path "$jbr\bin\java.exe")) { throw "Android Studio JBR not found at $jbr" }
$env:JAVA_HOME = $jbr
$env:ANDROID_HOME = 'D:\AndroidSdk'
$env:ANDROID_SDK_ROOT = 'D:\AndroidSdk'
$env:Path = "$jbr\bin;$env:Path"

# Java 21 NIO Selector on Windows opens an AF_UNIX socket file in java.io.tmpdir; a long TEMP path (> ~100 chars, e.g. inside a
# tool sandbox) makes it fail with "Unable to establish loopback connection". Use a short temp dir for every Gradle/Kotlin JVM.
$shortTmp = 'D:\kz-tmp'
New-Item -ItemType Directory -Force -Path $shortTmp | Out-Null
$env:TEMP = $shortTmp
$env:TMP = $shortTmp

$mutex = New-Object System.Threading.Mutex($false, 'Local\KaizenEyeBuildLock')
Write-Host "[build.ps1] waiting for build lock (max $Timeout min)..."
$got = $false
try { $got = $mutex.WaitOne([TimeSpan]::FromMinutes($Timeout)) } catch [System.Threading.AbandonedMutexException] { $got = $true }
if (-not $got) { throw "Timed out waiting for the Kaizen build lock" }
$code = 1
$logDir = Join-Path $here 'dist\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir $LogName
try {
    Push-Location $ProjectDir
    Write-Host "[build.ps1] JAVA_HOME=$env:JAVA_HOME  project=$ProjectDir  gradlew $($GradleArgs -join ' ')"
    # Output goes to a FILE, not a pipe: the Gradle daemon inherits the client's stdout handle and would otherwise keep a
    # piped caller waiting until the daemon exits.
    $quoted = ($GradleArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    & cmd.exe /c "`"$ProjectDir\gradlew.bat`" $quoted --console=plain > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    # Print the tail while still holding the lock, so a following build cannot truncate the log first.
    Write-Host "[build.ps1] exit=$code  full log: $log"
    Get-Content $log -Tail $(if ($code -eq 0) { 25 } else { 80 })
} finally {
    Pop-Location
    $mutex.ReleaseMutex()
}
exit $code
