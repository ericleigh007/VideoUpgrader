param(
    [switch]$ReuseExisting
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $repoRoot "artifacts/runtime"
$stdoutLog = Join-Path $runtimeDir "desktop-test-tauri.stdout.log"
$stderrLog = Join-Path $runtimeDir "desktop-test-tauri.stderr.log"
$webviewUrl = "http://127.0.0.1:9223/json/version"
$appUrl = "http://127.0.0.1:1420"
$desktopModes = @("afterUpscale", "interpolateOnly")
$blindComparisonSourcePath = if ($env:DESKTOP_BLIND_COMPARISON_SOURCE_PATH) {
    $env:DESKTOP_BLIND_COMPARISON_SOURCE_PATH
}
else {
    Join-Path $repoRoot "public/fixtures/gui-progress-sample.mp4"
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            return
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for $Url"
}

function Wait-PortListening {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $connections = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
        if ($connections) {
            return
        }

        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for local port $Port to start listening"
}

function Wait-DesktopHarnessReady {
    param(
        [switch]$IncludeAppPort
    )

    if ($IncludeAppPort) {
        Wait-PortListening -Port 1420 -TimeoutSeconds 120
    }

    Wait-HttpReady -Url $webviewUrl -TimeoutSeconds 120
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

$python = Get-UpscalerPython
if (-not (Test-PythonModuleAvailable -Python $python -ModuleName "torch")) {
    Write-Host "Selected worker Python is missing torch. Running bootstrap first..."
    Invoke-CheckedCommand -Command { & "$PSScriptRoot/bootstrap.ps1" } -FailureMessage "Bootstrap failed."
}

$startedProcess = $null
try {
    if (-not $ReuseExisting) {
        & "$PSScriptRoot/stop.ps1" -IncludeRepoWorkers | Out-Null
    }

    Clear-DevPort -Port 1420
    Remove-Item $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

    if (-not $ReuseExisting) {
        $startedProcess = Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/c", "set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9223 && npm run tauri:dev" `
            -WorkingDirectory $repoRoot `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -PassThru
    }

    Wait-DesktopHarnessReady -IncludeAppPort

    $originalRealSourcePath = $env:REAL_SOURCE_PATH
    $originalRealPreviewPath = $env:REAL_PREVIEW_PATH
    $originalDesktopPipelineMode = $env:DESKTOP_PIPELINE_MODE
    try {
        $env:REAL_SOURCE_PATH = ""
        $env:REAL_PREVIEW_PATH = ""
        $env:DESKTOP_PIPELINE_MODE = 'afterUpscale'
        Wait-DesktopHarnessReady
        Invoke-CheckedCommand -Command { node "$repoRoot/scripts/desktop_webview_playback_smoke.mjs" } -FailureMessage "Desktop WebView smoke failed for mode 'afterUpscale'."
        Wait-DesktopHarnessReady
        Invoke-CheckedCommand -Command { node "$repoRoot/scripts/desktop_jobs_window_recovery_smoke.mjs" } -FailureMessage "Desktop Jobs window recovery smoke failed."

        $env:REAL_SOURCE_PATH = $blindComparisonSourcePath
        Wait-DesktopHarnessReady
        Invoke-CheckedCommand -Command { node "$repoRoot/scripts/desktop_blind_comparison_smoke.mjs" } -FailureMessage "Desktop blind comparison smoke failed."

        $env:REAL_SOURCE_PATH = ""
        $env:DESKTOP_PIPELINE_MODE = 'interpolateOnly'
        Wait-DesktopHarnessReady
        Invoke-CheckedCommand -Command { node "$repoRoot/scripts/desktop_webview_playback_smoke.mjs" } -FailureMessage "Desktop WebView smoke failed for mode 'interpolateOnly'."
    }
    finally {
        $env:REAL_SOURCE_PATH = $originalRealSourcePath
        $env:REAL_PREVIEW_PATH = $originalRealPreviewPath
        $env:DESKTOP_PIPELINE_MODE = $originalDesktopPipelineMode
    }
}
finally {
    if ($startedProcess -ne $null) {
        Start-Sleep -Seconds 1
        & "$PSScriptRoot/stop.ps1" -IncludeRepoWorkers | Out-Null
    }
}