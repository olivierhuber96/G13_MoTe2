[CmdletBinding()]
param(
    [string]$CondaExe = "C:\ProgramData\miniforge3\Scripts\conda.exe",
    [string]$EnvName = "hydra_env",
    [string]$OutputDir = "environments"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoPath = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$outputPath = Join-Path $repoPath $OutputDir

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "conda.exe not found: $CondaExe"
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$environmentYml = Join-Path $outputPath "$EnvName.environment.yml"
$explicitSpec = Join-Path $outputPath "$EnvName.conda-explicit-win-64.txt"
$pipRequirements = Join-Path $outputPath "$EnvName.requirements-pip.txt"

& $CondaExe env export -n $EnvName |
    Where-Object { $_ -notmatch "^prefix:" } |
    Set-Content -Path $environmentYml -Encoding utf8

& $CondaExe list -n $EnvName --explicit |
    Set-Content -Path $explicitSpec -Encoding utf8

$envPrefix = (& $CondaExe env list | Select-String -Pattern "^\s*$EnvName\s+(.+)$").Matches.Groups[1].Value.Trim()
if ([string]::IsNullOrWhiteSpace($envPrefix)) {
    throw "Could not resolve environment prefix for $EnvName"
}

$pythonExe = Join-Path $envPrefix "python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "python.exe not found in environment: $pythonExe"
}

& $pythonExe -m pip freeze --all |
    Set-Content -Path $pipRequirements -Encoding utf8

Write-Host "Wrote:"
Write-Host "  $environmentYml"
Write-Host "  $explicitSpec"
Write-Host "  $pipRequirements"
