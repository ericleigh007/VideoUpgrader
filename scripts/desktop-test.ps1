param(
    [switch]$ReuseExisting,
    [int]$HarnessTimeoutSeconds = 90,
    [int]$StandardSmokeTimeoutSeconds = 180,
    [int]$BlindComparisonSmokeTimeoutSeconds = 420
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeDir = Join-Path $repoRoot "artifacts/runtime"
$stdoutLog = Join-Path $runtimeDir "desktop-test-tauri.stdout.log"
$stderrLog = Join-Path $runtimeDir "desktop-test-tauri.stderr.log"
$webviewUrl = "http://127.0.0.1:9223/json/version"
$blindComparisonScenarios = @(
    @{ Name = "upscale"; PreviewDurationSeconds = 3; ExerciseBlindJobControls = $true; CancelBlindComparison = $false },
    @{ Name = "colorizeOnly"; PreviewDurationSeconds = 3; ExerciseBlindJobControls = $false; CancelBlindComparison = $false },
    @{ Name = "colorizeBeforeUpscale"; PreviewDurationSeconds = 3; ExerciseBlindJobControls = $false; CancelBlindComparison = $false },
    @{ Name = "interpolateAfterUpscale"; PreviewDurationSeconds = 3; ExerciseBlindJobControls = $false; CancelBlindComparison = $false },
    @{ Name = "upscale"; PreviewDurationSeconds = 3; ExerciseBlindJobControls = $false; CancelBlindComparison = $true }
)
$blindComparisonSourcePath = if ($env:DESKTOP_BLIND_COMPARISON_SOURCE_PATH) {
    $env:DESKTOP_BLIND_COMPARISON_SOURCE_PATH
}
else {
    Join-Path $repoRoot "public/fixtures/gui-progress-sample.mp4"
}
$desktopColorContextPath = Join-Path $runtimeDir "desktop-color-context-smoke.png"

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
        [switch]$IncludeAppPort,
        [System.Diagnostics.Process]$StartedProcess
    )

    if ($StartedProcess -and $StartedProcess.HasExited) {
        $stdoutTail = if (Test-Path $stdoutLog) {
            (Get-Content $stdoutLog -Tail 40 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }
        $stderrTail = if (Test-Path $stderrLog) {
            (Get-Content $stderrLog -Tail 40 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }

        throw "Desktop harness exited before becoming ready. Stdout:`n$stdoutTail`nStderr:`n$stderrTail"
    }

    if ($IncludeAppPort) {
        Wait-PortListening -Port 1420 -TimeoutSeconds $HarnessTimeoutSeconds
    }

    Wait-HttpReady -Url $webviewUrl -TimeoutSeconds $HarnessTimeoutSeconds
}

function Invoke-NodeSmoke {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ScriptPath,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds,
        [string]$Label = "smoke"
    )

    $safeLabel = ($Label -replace "[^A-Za-z0-9_.-]", "-").ToLowerInvariant()
    $commandStdoutLog = Join-Path $runtimeDir "$safeLabel.stdout.log"
    $commandStderrLog = Join-Path $runtimeDir "$safeLabel.stderr.log"
    Remove-Item $commandStdoutLog, $commandStderrLog -ErrorAction SilentlyContinue

    Write-Host "Running $Label (timeout ${TimeoutSeconds}s)..."
    $process = Start-Process -FilePath "node" `
        -ArgumentList $ScriptPath `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $commandStdoutLog `
        -RedirectStandardError $commandStderrLog `
        -PassThru

    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        $stdoutTail = if (Test-Path $commandStdoutLog) {
            (Get-Content $commandStdoutLog -Tail 80 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }
        $stderrTail = if (Test-Path $commandStderrLog) {
            (Get-Content $commandStderrLog -Tail 80 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }

        throw "$FailureMessage Timed out after $TimeoutSeconds seconds. Stdout:`n$stdoutTail`nStderr:`n$stderrTail"
    }

    if ($process.ExitCode -ne 0) {
        $stdoutTail = if (Test-Path $commandStdoutLog) {
            (Get-Content $commandStdoutLog -Tail 80 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }
        $stderrTail = if (Test-Path $commandStderrLog) {
            (Get-Content $commandStderrLog -Tail 80 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        }
        else {
            ""
        }

        throw "$FailureMessage Exit code: $($process.ExitCode). Stdout:`n$stdoutTail`nStderr:`n$stderrTail"
    }
}

function Start-DesktopHarness {
    if ($ReuseExisting) {
        Write-Host "Waiting for existing desktop harness..."
        Wait-DesktopHarnessReady -IncludeAppPort -StartedProcess $script:startedProcess
        return
    }

    Clear-DevPort -Port 1420
    Remove-Item $stdoutLog, $stderrLog -ErrorAction SilentlyContinue
    $script:startedProcess = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9223 && npm run tauri:dev" `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    Write-Host "Waiting for desktop harness..."
    Wait-DesktopHarnessReady -IncludeAppPort -StartedProcess $script:startedProcess
}

function Restart-DesktopHarness {
    if ($ReuseExisting) {
        Wait-DesktopHarnessReady -IncludeAppPort -StartedProcess $script:startedProcess
        return
    }

    & "$PSScriptRoot/stop.ps1" -IncludeRepoWorkers | Out-Null
    $script:startedProcess = $null
    Start-DesktopHarness
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

$python = Get-UpscalerPython
if (-not (Test-PythonModuleAvailable -Python $python -ModuleName "torch")) {
    Write-Host "Selected worker Python is missing torch. Running bootstrap first..."
    Invoke-CheckedCommand -Command { & "$PSScriptRoot/bootstrap.ps1" } -FailureMessage "Bootstrap failed."
}

$script:startedProcess = $null
try {
    if (-not $ReuseExisting) {
        & "$PSScriptRoot/stop.ps1" -IncludeRepoWorkers | Out-Null
    }

    Start-DesktopHarness

    $originalRealSourcePath = $env:REAL_SOURCE_PATH
    $originalRealPreviewPath = $env:REAL_PREVIEW_PATH
    $originalRealContextPath = $env:REAL_CONTEXT_PATH
    $originalDesktopPipelineMode = $env:DESKTOP_PIPELINE_MODE
    $originalBlindComparisonScenario = $env:BLIND_COMPARISON_SCENARIO
    $originalBlindPreviewDurationSeconds = $env:BLIND_PREVIEW_DURATION_SECONDS
    $originalExerciseBlindJobControls = $env:EXERCISE_BLIND_JOB_CONTROLS
    $originalCancelBlindComparison = $env:CANCEL_BLIND_COMPARISON
    try {
        $env:REAL_SOURCE_PATH = ""
        $env:REAL_PREVIEW_PATH = ""
        $env:DESKTOP_PIPELINE_MODE = 'afterUpscale'
        Wait-DesktopHarnessReady -StartedProcess $script:startedProcess
        $env:CDP_CONNECT_TIMEOUT_MS = [string]($HarnessTimeoutSeconds * 1000)
        $env:RUN_TIMEOUT_MS = [string]($StandardSmokeTimeoutSeconds * 1000)
        Invoke-NodeSmoke -ScriptPath "$repoRoot/scripts/desktop_webview_playback_smoke.mjs" -FailureMessage "Desktop WebView smoke failed for mode 'afterUpscale'." -TimeoutSeconds $StandardSmokeTimeoutSeconds -Label "desktop-webview-afterupscale"
        Restart-DesktopHarness
        Invoke-NodeSmoke -ScriptPath "$repoRoot/scripts/desktop_jobs_window_recovery_smoke.mjs" -FailureMessage "Desktop Jobs window recovery smoke failed." -TimeoutSeconds $StandardSmokeTimeoutSeconds -Label "desktop-jobs-recovery"
        Copy-Item (Join-Path $repoRoot "docs/images/Blind-test-box.png") $desktopColorContextPath -Force
        Restart-DesktopHarness
        $env:REAL_SOURCE_PATH = $blindComparisonSourcePath
        $env:REAL_CONTEXT_PATH = $desktopColorContextPath
        Wait-DesktopHarnessReady -StartedProcess $script:startedProcess
        Invoke-NodeSmoke -ScriptPath "$repoRoot/scripts/desktop_color_context_smoke.mjs" -FailureMessage "Desktop color context smoke failed." -TimeoutSeconds $StandardSmokeTimeoutSeconds -Label "desktop-color-context"

        $env:REAL_SOURCE_PATH = $blindComparisonSourcePath
        foreach ($blindScenario in $blindComparisonScenarios) {
            Restart-DesktopHarness
            $env:BLIND_COMPARISON_SCENARIO = [string]$blindScenario.Name
            $env:BLIND_PREVIEW_DURATION_SECONDS = [string]$blindScenario.PreviewDurationSeconds
            if ($blindScenario.ExerciseBlindJobControls) {
                $env:EXERCISE_BLIND_JOB_CONTROLS = '1'
            }
            else {
                $env:EXERCISE_BLIND_JOB_CONTROLS = ''
            }

            if ($blindScenario.CancelBlindComparison) {
                $env:CANCEL_BLIND_COMPARISON = '1'
            }
            else {
                $env:CANCEL_BLIND_COMPARISON = ''
            }

            Wait-DesktopHarnessReady -StartedProcess $script:startedProcess
            $scenarioLabel = [string]$blindScenario.Name
            if ($blindScenario.CancelBlindComparison) {
                $scenarioLabel = "$scenarioLabel-cancel"
            }
            $env:RUN_TIMEOUT_MS = [string]($BlindComparisonSmokeTimeoutSeconds * 1000)
            Invoke-NodeSmoke -ScriptPath "$repoRoot/scripts/desktop_blind_comparison_smoke.mjs" -FailureMessage "Desktop blind comparison smoke failed for scenario '$scenarioLabel'." -TimeoutSeconds $BlindComparisonSmokeTimeoutSeconds -Label "desktop-blind-$scenarioLabel"
        }

        $env:REAL_SOURCE_PATH = ""
        $env:DESKTOP_PIPELINE_MODE = 'interpolateOnly'
    Restart-DesktopHarness
    Wait-DesktopHarnessReady -StartedProcess $script:startedProcess
        $env:RUN_TIMEOUT_MS = [string]($StandardSmokeTimeoutSeconds * 1000)
        Invoke-NodeSmoke -ScriptPath "$repoRoot/scripts/desktop_webview_playback_smoke.mjs" -FailureMessage "Desktop WebView smoke failed for mode 'interpolateOnly'." -TimeoutSeconds $StandardSmokeTimeoutSeconds -Label "desktop-webview-interpolateonly"
    }
    finally {
        $env:REAL_SOURCE_PATH = $originalRealSourcePath
        $env:REAL_PREVIEW_PATH = $originalRealPreviewPath
        $env:REAL_CONTEXT_PATH = $originalRealContextPath
        $env:DESKTOP_PIPELINE_MODE = $originalDesktopPipelineMode
        $env:BLIND_COMPARISON_SCENARIO = $originalBlindComparisonScenario
        $env:BLIND_PREVIEW_DURATION_SECONDS = $originalBlindPreviewDurationSeconds
        $env:EXERCISE_BLIND_JOB_CONTROLS = $originalExerciseBlindJobControls
        $env:CANCEL_BLIND_COMPARISON = $originalCancelBlindComparison
        $env:CDP_CONNECT_TIMEOUT_MS = $null
        $env:RUN_TIMEOUT_MS = $null
    }
}
finally {
    if ($script:startedProcess -ne $null) {
        & "$PSScriptRoot/stop.ps1" -IncludeRepoWorkers | Out-Null
    }
}