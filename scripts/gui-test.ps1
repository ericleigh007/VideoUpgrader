param()

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

Invoke-CheckedCommand -Command { npm run test:gui } -FailureMessage "GUI tests failed."
