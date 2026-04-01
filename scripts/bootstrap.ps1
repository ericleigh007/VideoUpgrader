param()

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

Assert-Prerequisites

$python = Get-UpscalerPython

Invoke-CheckedCommand -Command { npm install } -FailureMessage "npm install failed."
Invoke-CheckedCommand -Command { & $python -m pip install --upgrade pip } -FailureMessage "pip upgrade failed."
Invoke-CheckedCommand -Command { & $python -m pip install -r "$PSScriptRoot/../python/requirements.txt" } -FailureMessage "Python dependency installation failed."

$torchIndexUrl = if ($env:UPSCALER_TORCH_INDEX_URL) {
	$env:UPSCALER_TORCH_INDEX_URL
} else {
	"https://download.pytorch.org/whl/cu128"
}

$torchVersion = if ($env:UPSCALER_TORCH_VERSION) {
	$env:UPSCALER_TORCH_VERSION
} else {
	"2.11.0"
}

$torchVisionVersion = if ($env:UPSCALER_TORCHVISION_VERSION) {
	$env:UPSCALER_TORCHVISION_VERSION
} else {
	"0.26.0"
}

Invoke-CheckedCommand -Command {
	& $python -m pip install --upgrade --force-reinstall --index-url $torchIndexUrl "torch==$torchVersion" "torchvision==$torchVisionVersion"
} -FailureMessage "PyTorch GPU runtime installation failed."

Write-Host "Bootstrap complete."
