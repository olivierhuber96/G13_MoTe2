[CmdletBinding()]
param(
    [string]$RepoPath = "",
    [string]$Remote = "origin",
    [string]$BranchPrefix = "autosave",
    [switch]$NoPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($RepoPath)) {
    $RepoPath = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
else {
    $RepoPath = (Resolve-Path $RepoPath).Path
}

function Write-Log {
    param([string]$Message)

    try {
        $logRoot = Join-Path $env:LOCALAPPDATA "GitAutosave"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $logPath = Join-Path $logRoot "PythonScripts-autosave.log"
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "$timestamp $Message" | Add-Content -Path $logPath
    }
    catch {
        # Console output is enough when the log directory is unavailable.
    }
    Write-Host $Message
}

function Invoke-Git {
    $Arguments = $args
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git -C $RepoPath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed: $($output -join [Environment]::NewLine)"
    }
    return $output
}

function Resolve-GitPath {
    param([string]$GitPathName)

    $path = (Invoke-Git rev-parse --git-path $GitPathName | Select-Object -First 1).Trim()
    if ([System.IO.Path]::IsPathRooted($path)) {
        return $path
    }
    return (Join-Path $RepoPath $path)
}

function ConvertTo-RefSegment {
    param([string]$Value)

    $safe = $Value -replace "[^A-Za-z0-9._-]+", "-"
    $safe = $safe.Trim(".-_/")
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return "unknown"
    }
    return $safe
}

try {
    Invoke-Git rev-parse --is-inside-work-tree | Out-Null

    $blockedStates = @(
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "rebase-apply",
        "rebase-merge"
    )

    foreach ($state in $blockedStates) {
        if (Test-Path -LiteralPath (Resolve-GitPath $state)) {
            Write-Log "Skipped: repository has active Git state '$state'."
            exit 0
        }
    }

    $head = (Invoke-Git rev-parse --verify HEAD | Select-Object -First 1).Trim()
    $status = @(Invoke-Git status --porcelain=v1 --untracked-files=all)
    if ($status.Count -eq 0) {
        Write-Log "Skipped: no working tree changes."
        exit 0
    }

    $branch = (Invoke-Git branch --show-current | Select-Object -First 1).Trim()
    if ([string]::IsNullOrWhiteSpace($branch)) {
        $shortHead = (Invoke-Git rev-parse --short HEAD | Select-Object -First 1).Trim()
        $branch = "detached-$shortHead"
    }

    $computer = ConvertTo-RefSegment $env:COMPUTERNAME
    $branchSegment = ConvertTo-RefSegment $branch
    $localRef = "refs/heads/$BranchPrefix/$computer/$branchSegment"
    $remoteRef = "refs/heads/$BranchPrefix/$computer/$branchSegment"

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "git-autosave"
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    $tempIndex = Join-Path $tempRoot ("index-" + [guid]::NewGuid().ToString("N"))

    $oldGitIndexFile = $env:GIT_INDEX_FILE
    try {
        $env:GIT_INDEX_FILE = $tempIndex
        Invoke-Git read-tree HEAD | Out-Null
        Invoke-Git add -A -- . | Out-Null
        $tree = (Invoke-Git write-tree | Select-Object -First 1).Trim()
    }
    finally {
        if ($null -eq $oldGitIndexFile) {
            Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue
        }
        else {
            $env:GIT_INDEX_FILE = $oldGitIndexFile
        }
        Remove-Item -LiteralPath $tempIndex -Force -ErrorAction SilentlyContinue
    }

    $parent = $head
    $existingAutosave = & git -C $RepoPath rev-parse --verify --quiet $localRef 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($existingAutosave)) {
        $parent = ($existingAutosave | Select-Object -First 1).Trim()
    }

    $parentTree = (Invoke-Git show -s --format=%T $parent | Select-Object -First 1).Trim()
    if ($tree -eq $parentTree) {
        if (-not $NoPush) {
            Invoke-Git push $Remote "${localRef}:${remoteRef}" | Out-Null
            Write-Log "Pushed existing autosave branch to $remoteRef."
        }
        else {
            Write-Log "Skipped: autosave branch already matches working tree."
        }
        exit 0
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
    $subject = "autosave: $branch at $timestamp"
    $body = "Source branch: $branch`nSource HEAD: $head`nComputer: $env:COMPUTERNAME"
    $commit = (Invoke-Git commit-tree $tree -p $parent -m $subject -m $body | Select-Object -First 1).Trim()

    Invoke-Git update-ref $localRef $commit | Out-Null

    if (-not $NoPush) {
        Invoke-Git push $Remote "${commit}:${remoteRef}" | Out-Null
        Write-Log "Autosaved and pushed $commit to $remoteRef."
    }
    else {
        Write-Log "Autosaved $commit to $localRef without pushing."
    }
}
catch {
    Write-Log "Failed: $($_.Exception.Message)"
    exit 1
}
