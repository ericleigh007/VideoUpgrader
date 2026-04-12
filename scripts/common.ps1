function Get-UpscalerPython {
    if ($env:UPSCALER_PYTHON) {
        return $env:UPSCALER_PYTHON
    }

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $repoLocalPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $repoLocalPython) {
        return $repoLocalPython
    }

    return "python"
}

function Get-UpscalerPythonSeedCandidates {
    $candidates = @()

    if ($env:UPSCALER_PYTHON -and (Test-Path $env:UPSCALER_PYTHON)) {
        $candidates += ,@($env:UPSCALER_PYTHON)
    }

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $repoLocalPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $repoLocalPython) {
        $candidates += ,@($repoLocalPython)
    }

    $python310 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
    if (Test-Path $python310) {
        $candidates += ,@($python310)
    }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $candidates += ,@($pyCommand.Source, "-3.10")
        $candidates += ,@($pyCommand.Source, "-3.11")
        $candidates += ,@($pyCommand.Source, "-3.12")
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $candidates += ,@($pythonCommand.Source)
    }

    return $candidates
}

function Invoke-PythonSeedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandSpec,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $executable = $CommandSpec[0]
    $prefixArgs = @()
    if ($CommandSpec.Count -gt 1) {
        $prefixArgs = $CommandSpec[1..($CommandSpec.Count - 1)]
    }

    & $executable @prefixArgs @Arguments
}

function Test-PythonSeedUsable {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandSpec
    )

    try {
        Invoke-PythonSeedCommand -CommandSpec $CommandSpec -Arguments @("-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)") | Out-Null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-BootstrapPythonSeed {
    foreach ($candidate in Get-UpscalerPythonSeedCandidates) {
        if (Test-PythonSeedUsable -CommandSpec $candidate) {
            return $candidate
        }
    }

    throw "Python 3.10+ is required. Run scripts/bootstrap.ps1 so it can install Python, or set UPSCALER_PYTHON to a valid interpreter."
}

function Refresh-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathSegments = @($machinePath, $userPath) | Where-Object { $_ -and $_.Trim() }
    if ($pathSegments.Count -gt 0) {
        $env:Path = ($pathSegments -join ";")
    }
}

function Get-WingetCommand {
    $wingetCommand = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCommand) {
        return $wingetCommand.Source
    }

    $windowsAppsWinget = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\winget.exe"
    if (Test-Path $windowsAppsWinget) {
        return $windowsAppsWinget
    }

    return $null
}

function Assert-WingetAvailable {
    $winget = Get-WingetCommand
    if (-not $winget) {
        throw "winget is required for one-click workstation bootstrap on Windows. Install App Installer from the Microsoft Store and rerun scripts/bootstrap.ps1."
    }

    return $winget
}

function Get-VisualStudioBuildToolsInstallPath {
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswherePath)) {
        return $null
    }

    $installationPath = & $vswherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -latest -property installationPath
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $resolved = ($installationPath | Select-Object -First 1).Trim()
    if (-not $resolved) {
        return $null
    }

    return $resolved
}

function Test-WebView2RuntimeInstalled {
    $clientKey = "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    $clientKeyWow = "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    foreach ($path in @($clientKey, $clientKeyWow)) {
        try {
            $value = (Get-ItemProperty -Path $path -Name "pv" -ErrorAction Stop).pv
            if ($value) {
                return $true
            }
        }
        catch {
        }
    }

    return $false
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Id,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,
        [string[]]$AdditionalArguments = @()
    )

    $winget = Assert-WingetAvailable
    Write-Host "Installing $DisplayName via winget..."
    & $winget install --id $Id --exact --accept-source-agreements --accept-package-agreements --silent --disable-interactivity @AdditionalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Installing $DisplayName failed. Exit code: $LASTEXITCODE"
    }
    Refresh-ProcessPath
}

function Ensure-UpscalerSystemDependencies {
    Refresh-ProcessPath

    if (-not (Get-Command node -ErrorAction SilentlyContinue) -or -not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Install-WingetPackage -Id "OpenJS.NodeJS.LTS" -DisplayName "Node.js LTS"
    }

    try {
        $null = Resolve-BootstrapPythonSeed
    }
    catch {
        Install-WingetPackage -Id "Python.Python.3.10" -DisplayName "Python 3.10"
    }

    if (-not (Test-Path (Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe")) -and -not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Install-WingetPackage -Id "Rustlang.Rustup" -DisplayName "Rust toolchain"
    }

    if (-not (Get-VisualStudioBuildToolsInstallPath)) {
        Install-WingetPackage -Id "Microsoft.VisualStudio.2022.BuildTools" -DisplayName "Microsoft C++ Build Tools" -AdditionalArguments @(
            "--override",
            "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        )
    }

    if (-not (Test-WebView2RuntimeInstalled)) {
        Install-WingetPackage -Id "Microsoft.EdgeWebView2Runtime" -DisplayName "Microsoft Edge WebView2 Runtime"
    }

    Refresh-ProcessPath
    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if ((Test-Path $cargoBin) -and -not (($env:Path -split ";") -contains $cargoBin)) {
        $env:Path = "$cargoBin;$env:Path"
    }
}

function Assert-CommandAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string]$HelpText
    )

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$Command is required. $HelpText"
    }
}

function Assert-Prerequisites {
    Assert-CommandAvailable -Command "npm" -HelpText "Install Node.js and npm before running repository scripts, or run scripts/bootstrap.ps1 to provision them automatically."
    Assert-CommandAvailable -Command "cargo" -HelpText "Install the Rust toolchain before running repository scripts, or run scripts/bootstrap.ps1 to provision it automatically."
    Assert-CommandAvailable -Command (Get-UpscalerPython) -HelpText "Provide Python 3.10+, keep the repo .venv available, or run scripts/bootstrap.ps1 to provision it automatically."
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage Exit code: $LASTEXITCODE"
    }
}

function Get-ProcessCommandLine {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId
    )

    try {
        return (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop).CommandLine
    }
    catch {
        return $null
    }
}

function Get-UpscalerDevProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $normalizedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).ToLowerInvariant()
    $candidateNames = @("node", "cargo", "upscaler")
    $candidates = Get-Process -Name $candidateNames -ErrorAction SilentlyContinue
    $matches = @()

    foreach ($process in $candidates) {
        $isMatch = $false

        if ($process.ProcessName -eq "upscaler") {
            $isMatch = $true
        }
        else {
            $commandLine = Get-ProcessCommandLine -ProcessId $process.Id
            if ($commandLine) {
                $normalizedCommandLine = $commandLine.ToLowerInvariant()
                if ($normalizedCommandLine.Contains($normalizedRepoRoot) -or $normalizedCommandLine.Contains("tauri dev") -or $normalizedCommandLine.Contains("vite")) {
                    $isMatch = $true
                }
            }
        }

        if ($isMatch) {
            $matches += $process
        }
    }

    return $matches
}

function Test-PythonModuleAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" | Out-Null
    return $LASTEXITCODE -eq 0
}

function Clear-DevPort {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    if (-not $connections) {
        return
    }

    foreach ($processId in $connections) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if (-not $process) {
            continue
        }

        if ($process.ProcessName -ne "node") {
            throw "Port $Port is already in use by process '$($process.ProcessName)' (PID $processId). Stop that process or change the dev port before launching Upscaler."
        }

        Stop-Process -Id $processId -Force
        Start-Sleep -Milliseconds 250
    }
}

function Wait-PortAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [int]$TimeoutSeconds = 10
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
        if (-not $connections) {
            return
        }

        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)

    throw "Port $Port did not become available within $TimeoutSeconds seconds."
}

function Stop-UpscalerDevProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $processes = Get-UpscalerDevProcesses -RepoRoot $RepoRoot |
        Sort-Object @{ Expression = {
            if ($_.ProcessName -eq "node") {
                $commandLine = Get-ProcessCommandLine -ProcessId $_.Id
                if ($commandLine -and $commandLine.ToLowerInvariant().Contains("tauri dev")) {
                    return 0
                }
                return 2
            }
            if ($_.ProcessName -eq "cargo") {
                return 1
            }
            return 3
        } }

    foreach ($process in $processes) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }

    return $processes
}

function Get-UpscalerRepoProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $normalizedRepoRoot = [System.IO.Path]::GetFullPath($RepoRoot).ToLowerInvariant()
    $candidateNames = @("node", "cargo", "upscaler", "python", "python3", "ffmpeg", "pwsh", "powershell", "cmd")
    $candidates = Get-Process -Name $candidateNames -ErrorAction SilentlyContinue
    $matches = @()

    foreach ($process in $candidates) {
        $commandLine = Get-ProcessCommandLine -ProcessId $process.Id
        if (-not $commandLine) {
            continue
        }

        if ($commandLine.ToLowerInvariant().Contains($normalizedRepoRoot)) {
            $matches += $process
        }
    }

    return $matches
}

function Stop-UpscalerRepoProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $processes = Get-UpscalerRepoProcesses -RepoRoot $RepoRoot | Sort-Object ProcessName, Id -Unique
    foreach ($process in $processes) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }

    return $processes
}

