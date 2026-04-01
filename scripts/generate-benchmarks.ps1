param(
    [string]$OutputDir = "artifacts/benchmarks",
    [string]$Name = "baseline_fixture",
    [int]$Frames = 12,
    [int]$Width = 3840,
    [int]$Height = 2160,
    [int]$DownscaleWidth = 1280,
    [int]$DownscaleHeight = 720
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

$python = Get-UpscalerPython
$repoRoot = Resolve-Path "$PSScriptRoot/.."
$env:PYTHONPATH = Join-Path $repoRoot "python"

Invoke-CheckedCommand -Command {
    & $python -m upscaler_worker.cli generate-benchmark `
        --output-dir (Join-Path $repoRoot $OutputDir) `
        --name $Name `
        --frames $Frames `
        --width $Width `
        --height $Height `
        --downscale-width $DownscaleWidth `
        --downscale-height $DownscaleHeight
} -FailureMessage "Synthetic benchmark generation failed."
