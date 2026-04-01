param()

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites
Clear-DevPort -Port 1420

Invoke-CheckedCommand -Command { npm run tauri:dev } -FailureMessage "Tauri dev launch failed."
