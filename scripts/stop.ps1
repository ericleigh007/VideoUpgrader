param(
    [switch]$IncludeRepoWorkers
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/common.ps1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stopped = @()

$stopped += @(Stop-UpscalerDevProcesses -RepoRoot $repoRoot)

if ($IncludeRepoWorkers) {
    $stopped += @(Stop-UpscalerRepoProcesses -RepoRoot $repoRoot)
}

$remaining = if ($IncludeRepoWorkers) {
    @(Get-UpscalerRepoProcesses -RepoRoot $repoRoot)
} else {
    @(Get-UpscalerDevProcesses -RepoRoot $repoRoot)
}

[pscustomobject]@{
    Stopped = @($stopped | Select-Object Id, ProcessName -Unique)
    Remaining = @($remaining | Select-Object Id, ProcessName -Unique)
} | ConvertTo-Json -Depth 4