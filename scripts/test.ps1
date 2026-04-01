param()

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

$python = Get-UpscalerPython
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$pythonPath = Join-Path $repoRoot "python"

Invoke-CheckedCommand -Command { npm run test:web } -FailureMessage "Frontend tests failed."
$env:PYTHONPATH = $pythonPath
Invoke-CheckedCommand -Command { & $python -m unittest discover -s "$pythonPath/tests" -p "test_*.py" } -FailureMessage "Python tests failed."

Write-Host "All tests passed."
