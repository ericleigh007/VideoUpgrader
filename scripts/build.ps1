param(
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

if ($RunTests) {
    & "$PSScriptRoot/test.ps1"
}

Invoke-CheckedCommand -Command { npm run build:web } -FailureMessage "Frontend build failed."
Invoke-CheckedCommand -Command { cargo check --manifest-path "$PSScriptRoot/../src-tauri/Cargo.toml" } -FailureMessage "Cargo check failed."

Write-Host "Build checks completed."
