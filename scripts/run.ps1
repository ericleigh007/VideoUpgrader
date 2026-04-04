param(
	[switch]$RestartExisting
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$existingDevProcesses = @(Get-UpscalerDevProcesses -RepoRoot $repoRoot)
if ($existingDevProcesses.Count -gt 0 -and -not $RestartExisting) {
	Write-Host "Upscaler dev is already running. Reusing the existing session. Use -RestartExisting to force a fresh launch."
	return
}

if ($RestartExisting) {
	Stop-UpscalerDevProcesses -RepoRoot $repoRoot
}

Clear-DevPort -Port 1420
Wait-PortAvailable -Port 1420 -TimeoutSeconds 10

$python = Get-UpscalerPython
if (-not (Test-PythonModuleAvailable -Python $python -ModuleName "torch")) {
	Write-Host "Selected worker Python is missing torch. Running bootstrap first..."
	Invoke-CheckedCommand -Command { & "$PSScriptRoot/bootstrap.ps1" } -FailureMessage "Bootstrap failed."
}

Invoke-CheckedCommand -Command { npm run tauri:dev } -FailureMessage "Tauri dev launch failed."
