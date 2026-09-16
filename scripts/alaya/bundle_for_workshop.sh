#!/usr/bin/env bash
# Run this on your LAPTOP, not in the Workshop.
#
# Cloning from GitHub inside China is painfully slow. This ships the repo as a
# single git bundle over the SSH connection you already have. The bundle is the
# packed history only (~25 MB here), and git rebuilds the 78 MB working tree on
# the far side, so it moves a fraction of the bytes a clone would.
#
#   bash scripts/alaya/bundle_for_workshop.sh cci-evhcgiil.bj5
#   bash scripts/alaya/bundle_for_workshop.sh cci-evhcgiil.bj5 /root/idm
#
# Afterwards the checkout on the Workshop is an ordinary git repo pointing at
# GitHub, so pull and push work as normal - only this first hop is special.

set -euo pipefail

HOST="${1:-}"
REMOTE_DIR="${2:-/root/idm}"
BRANCH="${BRANCH:-claude/alaya-new-cloud-setup-4ode4d}"
ORIGIN_URL="${ORIGIN_URL:-https://github.com/Yassinesr/IDM.git}"

if [ -z "$HOST" ]; then
    echo "usage: $0 <ssh-host> [remote-dir]" >&2
    echo "  e.g. $0 cci-evhcgiil.bj5 /root/idm" >&2
    exit 1
fi

git rev-parse --git-dir >/dev/null 2>&1 || {
    echo "ERROR: run this from inside your local clone of the repo." >&2; exit 1; }

BUNDLE="$(mktemp -t idm-XXXXXX.bundle)"
trap 'rm -f "$BUNDLE"' EXIT

# Fetch straight into a local branch: `git bundle` can only package local refs,
# and a bare `git fetch origin <branch>` leaves you with FETCH_HEAD only.
# This hop is incremental and tiny - you already have the history.
echo "==> fetching $BRANCH from origin (incremental)"
git fetch origin "$BRANCH:$BRANCH" 2>&1 | sed 's/^/    /' || {
    echo "    (already up to date, or the branch is already local)"; }

REFS="$BRANCH"
git show-ref --verify --quiet refs/heads/main && REFS="main $BRANCH"

echo "==> building bundle of: $REFS"
git bundle create "$BUNDLE" $REFS
git bundle verify "$BUNDLE" >/dev/null
echo "    $(du -h "$BUNDLE" | cut -f1) -> $HOST:$REMOTE_DIR/idm.bundle"

echo "==> copying"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
scp "$BUNDLE" "$HOST:$REMOTE_DIR/idm.bundle"

echo "==> cloning on the Workshop"
ssh "$HOST" "
set -e
cd '$REMOTE_DIR'
if [ -d IDM/.git ]; then
    echo '    IDM already exists; fetching from the bundle instead'
    cd IDM
    git fetch '$REMOTE_DIR/idm.bundle' '$BRANCH:refs/remotes/bundle/$BRANCH'
    git checkout -B '$BRANCH' 'refs/remotes/bundle/$BRANCH'
else
    git clone -b '$BRANCH' '$REMOTE_DIR/idm.bundle' IDM
    cd IDM
fi
# Point at GitHub so pull/push behave normally from here on.
git remote set-url origin '$ORIGIN_URL'
echo
echo '    branch:  '\$(git rev-parse --abbrev-ref HEAD)
echo '    commit:  '\$(git log --oneline -1)
echo '    origin:  '\$(git remote get-url origin)
"

cat <<EOF

Done. On the Workshop:

    cd $REMOTE_DIR/IDM
    export IDM_ROOT=$REMOTE_DIR
    export IDM_ALLOW_EPHEMERAL=1
    bash scripts/alaya/00_bootstrap_workshop.sh

The bundle at $REMOTE_DIR/idm.bundle can be deleted once the clone works.
EOF
