#!/usr/bin/env bash
# Pull the latest code onto the server over SSH, and optionally reset the
# checkout to exactly match the remote.
#
#   bash scripts/alaya/sync_from_github.sh            # set up + fetch + show plan
#   bash scripts/alaya/sync_from_github.sh --force    # actually reset and clean
#   bash scripts/alaya/sync_from_github.sh --force --assets   # also restore the
#                                                    example photos from main
#
# HTTPS to github.com is blocked from Beijing, and so is SSH on port 22. GitHub
# also serves SSH on port 443, which normally survives - that is what this
# configures.

set -euo pipefail

REPO_URL_SSH="${REPO_URL_SSH:-git@github.com:Yassinesr/IDM.git}"
BRANCH="${BRANCH:-claude/alaya-new-cloud-setup-4ode4d}"
FORCE=0
ASSETS=0
for a in "$@"; do
    case "$a" in
        --force)  FORCE=1 ;;
        --assets) ASSETS=1 ;;
    esac
done

# Untracked things a reset must not destroy: pipeline outputs, and any images
# copied straight to the server that were never committed on this branch.
KEEP_PATTERNS=(work out results variants masks "*.png" "*.jpg" "*.jpeg")

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ----------------------------------------------------------------- ssh key ---
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "==> generating an SSH key (no passphrase, so pulls need no input)"
    ssh-keygen -t ed25519 -N "" -C "idm-server-$(hostname)" -f ~/.ssh/id_ed25519
    NEW_KEY=1
fi

# ------------------------------------------------------------- ssh config ----
# github.com over port 443: port 22 is filtered on many Chinese networks.
if ! grep -qs "Host github.com" ~/.ssh/config 2>/dev/null; then
    echo "==> routing github.com over ssh.github.com:443"
    cat >> ~/.ssh/config <<'CFG'

Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
CFG
    chmod 600 ~/.ssh/config
fi

# ------------------------------------------------------------ known_hosts ----
# Verify against GitHub's published ed25519 fingerprint rather than trusting
# whatever answers - the whole point here is a network that interferes.
GITHUB_ED25519_FP="SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"
if ! ssh-keygen -F ssh.github.com -f ~/.ssh/known_hosts >/dev/null 2>&1; then
    echo "==> fetching and verifying GitHub's host key"
    SCAN="$(ssh-keyscan -t ed25519 -p 443 ssh.github.com 2>/dev/null)" || true
    [ -n "$SCAN" ] || { echo "ERROR: could not reach ssh.github.com:443." >&2
                        echo "       Port 443 looks blocked too - see the notes below." >&2
                        exit 1; }
    GOT_FP="$(printf '%s\n' "$SCAN" | ssh-keygen -lf - | awk '{print $2}')"
    if [ "$GOT_FP" != "$GITHUB_ED25519_FP" ]; then
        echo "ERROR: host key does not match GitHub's published fingerprint." >&2
        echo "       expected $GITHUB_ED25519_FP" >&2
        echo "       got      $GOT_FP" >&2
        echo "       Refusing to continue - do not add this key." >&2
        exit 1
    fi
    printf '%s\n' "$SCAN" >> ~/.ssh/known_hosts
    echo "    fingerprint matches"
fi

if [ "${NEW_KEY:-0}" = "1" ]; then
    cat <<EOF

================ ADD THIS KEY TO GITHUB, THEN RE-RUN ================
$(cat ~/.ssh/id_ed25519.pub)

  github.com > Settings > SSH and GPG keys > New SSH key
  (or the repo's Settings > Deploy keys, if you prefer it scoped)
=====================================================================
EOF
    exit 0
fi

# -------------------------------------------------------------- remote -------
CURRENT="$(git remote get-url origin 2>/dev/null || echo none)"
if [ "$CURRENT" != "$REPO_URL_SSH" ]; then
    echo "==> switching origin: $CURRENT -> $REPO_URL_SSH"
    git remote set-url origin "$REPO_URL_SSH"
fi

echo "==> testing the connection"
ssh -o BatchMode=yes -T git@github.com 2>&1 | head -2 || true

echo "==> fetching"
git fetch origin "$BRANCH" main

# ---------------------------------------------------------------- plan -------
echo
echo "current : $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"
echo "remote  : $(git rev-parse --short "origin/$BRANCH") $(git log -1 --format=%s "origin/$BRANCH" | cut -c1-60)"

CLEAN_ARGS=(-d)
for p in "${KEEP_PATTERNS[@]}"; do CLEAN_ARGS+=(-e "$p"); done

echo
echo "would delete (untracked, excluding ${KEEP_PATTERNS[*]}):"
git clean -n "${CLEAN_ARGS[@]}" | sed 's/^/  /' || true
LOCAL_COMMITS="$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)"
[ "$LOCAL_COMMITS" != "0" ] && echo "  WARNING: $LOCAL_COMMITS local commit(s) would be discarded"

if [ "$FORCE" != "1" ]; then
    cat <<EOF

Nothing changed. To apply:
    bash scripts/alaya/sync_from_github.sh --force
EOF
    exit 0
fi

echo
echo "==> resetting to origin/$BRANCH"
git checkout -B "$BRANCH" "origin/$BRANCH"
git reset --hard "origin/$BRANCH"
git clean -f "${CLEAN_ARGS[@]}"

echo "==> now at $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"

# ---------------------------------------------------------------- assets ----
# Test photos live on main while the code lives on this branch, so a reset to
# the branch leaves the examples missing. Copy them out of origin/main WITHOUT
# staging them - git show, not git checkout - so they stay untracked here and
# the clean excludes above keep protecting them.
if [ "$ASSETS" = "1" ]; then
    echo "==> restoring gradio_demo/example/ from origin/main"
    n=0
    while read -r f; do
        [ -n "$f" ] || continue
        mkdir -p "$(dirname "$f")"
        git show "origin/main:$f" > "$f" && n=$((n+1))
    done < <(git ls-tree -r --name-only origin/main -- gradio_demo/example)
    echo "    $n file(s)"
fi
