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
    Assert-CommandAvailable -Command "npm" -HelpText "Install Node.js and npm before running repository scripts."
    Assert-CommandAvailable -Command "cargo" -HelpText "Install the Rust toolchain before running repository scripts."
    Assert-CommandAvailable -Command (Get-UpscalerPython) -HelpText "Provide Python 3.10+, keep the repo .venv available, or set UPSCALER_PYTHON explicitly."
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

