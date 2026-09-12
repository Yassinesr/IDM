# Run on your WINDOWS machine, from inside your local clone.
#
#   cd C:\Users\hp\Desktop\IDM
#   .\scripts\alaya\sync_from_windows.ps1 -SshHost idm-1.bj5
#
# Ships the repo as a git bundle rather than a zip. Two reasons that matters:
#   - a bundle carries real git history, so the copy on the Workshop can
#     `git pull` afterwards; a zip of the working tree cannot, and you end up
#     re-zipping for every change
#   - it is the packed history only (~25 MB here) instead of the 78 MB tree
#
# Only the first hop is special. After this, pull and push work normally.

param(
    [Parameter(Mandatory=$true)][string]$SshHost,
    [string]$RemoteDir = "/pvc/idm",
    [string]$Branch    = "claude/alaya-new-cloud-setup-4ode4d",
    [string]$OriginUrl = "https://github.com/Yassinesr/IDM.git"
)

$ErrorActionPreference = "Stop"

git rev-parse --git-dir *> $null
if ($LASTEXITCODE -ne 0) { throw "Run this from inside your local clone of the repo." }

# Fetch into a local branch: git bundle can only package local refs. This hop is
# incremental - you already have the history, so it is small even on a slow link.
Write-Host "==> fetching $Branch"
git fetch origin "${Branch}:${Branch}"
if ($LASTEXITCODE -ne 0) { Write-Host "    (already current, or branch already local)" }

$bundle = Join-Path $env:TEMP "idm.bundle"
Write-Host "==> building bundle"
$refs = @($Branch)
git show-ref --verify --quiet refs/heads/main
if ($LASTEXITCODE -eq 0) { $refs = @("main", $Branch) }
git bundle create $bundle @refs
git bundle verify $bundle *> $null
Write-Host ("    {0:N1} MB" -f ((Get-Item $bundle).Length / 1MB))

Write-Host "==> copying to ${SshHost}:$RemoteDir/"
ssh $SshHost "mkdir -p '$RemoteDir'"
scp $bundle "${SshHost}:$RemoteDir/idm.bundle"

Write-Host "==> updating the checkout on the Workshop"
$remote = @"
set -e
cd '$RemoteDir'
if [ -d IDM/.git ]; then
    cd IDM
    git fetch '$RemoteDir/idm.bundle' '${Branch}:refs/remotes/bundle/$Branch'
    git checkout -B '$Branch' 'refs/remotes/bundle/$Branch'
else
    rm -rf IDM.stale && [ -d IDM ] && mv IDM IDM.stale || true
    git clone -b '$Branch' '$RemoteDir/idm.bundle' IDM
    cd IDM
fi
git remote set-url origin '$OriginUrl'
echo "    branch: \$(git rev-parse --abbrev-ref HEAD)"
echo "    commit: \$(git log --oneline -1)"
"@
ssh $SshHost $remote

Write-Host ""
Write-Host "Done. On the Workshop:"
Write-Host "    cd $RemoteDir/IDM"
Write-Host "    export IDM_ROOT=$RemoteDir"
Write-Host "    bash scripts/alaya/00_bootstrap_workshop.sh"
