param(
	[switch]$RunTests
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"
Ensure-UpscalerSystemDependencies

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
	$bootstrapPythonSeed = Resolve-BootstrapPythonSeed
	Invoke-CheckedCommand -Command {
		Invoke-PythonSeedCommand -CommandSpec $bootstrapPythonSeed -Arguments @("-m", "venv", $venvRoot)
	} -FailureMessage "Creating the repo Python virtual environment failed."
}

$env:UPSCALER_PYTHON = $venvPython

$python = Get-UpscalerPython
$requirementsPath = Join-Path $repoRoot "python\requirements.txt"
$ddcolorRequirement = "git+https://github.com/piddnad/DDColor.git@master"

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = (Join-Path $repoRoot "python")

Push-Location $repoRoot
try {
	Invoke-CheckedCommand -Command { npm install } -FailureMessage "npm install failed."
	Invoke-CheckedCommand -Command { & $python -m pip install --upgrade pip } -FailureMessage "pip upgrade failed."

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

	$tempRequirementsPath = [System.IO.Path]::GetTempFileName()
	try {
		(Get-Content $requirementsPath) | Where-Object { $_.Trim() -ne $ddcolorRequirement } | Set-Content $tempRequirementsPath
		Invoke-CheckedCommand -Command { & $python -m pip install --no-build-isolation -r $tempRequirementsPath } -FailureMessage "Python dependency installation failed."
		Invoke-CheckedCommand -Command { & $python -m pip install --no-build-isolation --no-deps $ddcolorRequirement } -FailureMessage "DDColor dependency installation failed."
	}
	finally {
		Remove-Item $tempRequirementsPath -ErrorAction SilentlyContinue
	}

 	Invoke-CheckedCommand -Command {
		& $python -m upscaler_worker.cli prefetch-app-assets
	} -FailureMessage "Runtime and model asset prefetch failed."

	$buildArguments = @()
	if ($RunTests) {
		$buildArguments += "-RunTests"
	}

	Invoke-CheckedCommand -Command {
		& "$PSScriptRoot/build.ps1" @buildArguments
	} -FailureMessage "Build step failed during bootstrap."
}
finally {
	Pop-Location
	$env:PYTHONPATH = $previousPythonPath
}

Write-Host "Bootstrap complete. Dependencies, builds, runtimes, and model weights are ready."
