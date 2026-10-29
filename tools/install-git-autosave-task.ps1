[CmdletBinding()]
param(
    [int]$IntervalMinutes = 60,
    [string]$TaskName = "Git autosave - Python Scripts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoPath = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$autosaveScript = Join-Path $scriptRoot "git-autosave.ps1"

if (-not (Test-Path -LiteralPath $autosaveScript)) {
    throw "Autosave script not found: $autosaveScript"
}

$actionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$autosaveScript`"",
    "-RepoPath", "`"$repoPath`"",
    "-Remote", "origin"
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs -WorkingDirectory $repoPath
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$description = "Safely snapshots this repository to an autosave branch every $IntervalMinutes minutes."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description $description `
    -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName'."
Write-Host "It will run every $IntervalMinutes minutes and push to autosave/<computer>/<branch>."
