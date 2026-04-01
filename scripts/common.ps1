function Get-UpscalerPython {
    if ($env:UPSCALER_PYTHON) {
        return $env:UPSCALER_PYTHON
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
    Assert-CommandAvailable -Command (Get-UpscalerPython) -HelpText "Provide Python 3.10+ or set UPSCALER_PYTHON."
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
