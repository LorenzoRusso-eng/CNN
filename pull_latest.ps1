param(
    [string]$Branch = "CNN---cuda"
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $RepoDir

git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Errore: questa cartella non sembra una repository Git."
    exit 1
}

Write-Host "Repository: $RepoDir"
Write-Host "Remote origin:"
git remote get-url origin
Write-Host ""

git diff --quiet
$HasUnstagedChanges = $LASTEXITCODE -ne 0
git diff --cached --quiet
$HasStagedChanges = $LASTEXITCODE -ne 0

if ($HasUnstagedChanges -or $HasStagedChanges) {
    Write-Error "Errore: ci sono modifiche locali non salvate. Salvale con commit/stash prima di fare pull: git status"
    exit 1
}

Write-Host "Passo al branch $Branch..."
git checkout $Branch

Write-Host "Scarico gli aggiornamenti..."
git pull --ff-only origin $Branch

Write-Host ""
Write-Host "Pull completato."
