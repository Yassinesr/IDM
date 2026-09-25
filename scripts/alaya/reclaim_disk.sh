#!/usr/bin/env bash
# Find and remove space that is pure cache, on a box with no PVC and no NAS
# where 50 GB is all there will ever be.
#
#   bash scripts/alaya/reclaim_disk.sh          # report only, deletes nothing
#   bash scripts/alaya/reclaim_disk.sh --yes    # actually delete
#
# It never touches the two expensive things: the HF cache (the model weights -
# a 31 GB re-download over a throttled link) and the virtualenvs. Those are
# reported at the end so you can see the real shape of the disk.

set -uo pipefail

APPLY=0
[ "${1:-}" = "--yes" ] && APPLY=1

IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
FREED=0

# du -s in bytes, tolerating a path that is not there.
size_of() { du -sb "$1" 2>/dev/null | cut -f1 || echo 0; }
human()   { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0}B"; }

# report <label> <bytes> <command-to-run...>
# Prints the item, and runs the command only under --yes.
reclaim() {
    local label="$1" bytes="$2"; shift 2
    [ "${bytes:-0}" -lt 1048576 ] && return 0   # under 1 MB is noise
    FREED=$((FREED + bytes))
    printf '  %-34s %8s' "$label" "$(human "$bytes")"
    if [ "$APPLY" = "1" ]; then
        if "$@" >/dev/null 2>&1; then echo "   removed"; else echo "   FAILED"; fi
    else
        echo "   (would remove)"
    fi
}

BEFORE="$(df -B1 --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "== free now: $(human "${BEFORE:-0}") on / =="
echo
echo "== safe to delete (rebuilt automatically when next needed) =="

# 1. pip's wheel cache. On an ephemeral container this is dead weight: it only
#    speeds up a reinstall that will never happen, and building a torch stack
#    leaves a couple of GB behind.
for d in "${PIP_CACHE_DIR:-}" "$HOME/.cache/pip" /root/.cache/pip; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    reclaim "pip wheel cache" "$(size_of "$d")" rm -rf "$d"
    break
done

# 2. conda's package tarballs, already unpacked into the envs.
if [ -d "$IDM_ROOT/miniconda/pkgs" ]; then
    reclaim "conda package cache" "$(size_of "$IDM_ROOT/miniconda/pkgs")" \
        "$IDM_ROOT/miniconda/bin/conda" clean -a -y
fi

# 3. Interrupted downloads and stale locks in the HF cache. NOT the weights -
#    only *.incomplete, which is what a killed snapshot_download leaves behind.
HF="${HF_HOME:-$IDM_ROOT/hf}"
if [ -d "$HF" ]; then
    INC_B=0
    while IFS= read -r f; do
        INC_B=$((INC_B + $(stat -c%s "$f" 2>/dev/null || echo 0)))
    done < <(find "$HF" \( -name '*.incomplete' -o -name '*.lock' \) -type f 2>/dev/null)
    if [ "$INC_B" -gt 0 ]; then
        reclaim "HF partial downloads / locks" "$INC_B" \
            find "$HF" \( -name '*.incomplete' -o -name '*.lock' \) -type f -delete
    fi
fi

# 4. apt's lists and .debs. Nothing here is needed after install.
for d in /var/cache/apt/archives /var/lib/apt/lists; do
    [ -d "$d" ] || continue
    reclaim "apt cache ($(basename "$d"))" "$(size_of "$d")" \
        find "$d" -mindepth 1 -delete
done

# 5. Byte-code caches in the checkout.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYC_B=0
while IFS= read -r d; do
    PYC_B=$((PYC_B + $(size_of "$d")))
done < <(find "$REPO_DIR" -name __pycache__ -type d 2>/dev/null)
if [ "$PYC_B" -gt 0 ]; then
    reclaim "__pycache__ in the checkout" "$PYC_B" \
        find "$REPO_DIR" -name __pycache__ -type d -exec rm -rf {} +
fi

# 6. Loose git objects. gc is safe and reversible - it only repacks what is
#    already reachable, and prunes objects nothing points at.
if [ -d "$REPO_DIR/.git" ]; then
    G_BEFORE="$(size_of "$REPO_DIR/.git")"
    printf '  %-34s %8s' "git gc (.git is $(human "$G_BEFORE"))" ""
    if [ "$APPLY" = "1" ]; then
        if git -C "$REPO_DIR" gc --prune=now --quiet >/dev/null 2>&1; then
            G_AFTER="$(size_of "$REPO_DIR/.git")"
            FREED=$((FREED + G_BEFORE - G_AFTER))
            echo "   now $(human "$G_AFTER")"
        else
            echo "   FAILED"
        fi
    else
        echo "   (would repack)"
    fi
fi

echo
echo "== kept (expensive to rebuild - delete only deliberately) =="
[ -d "$HF" ] && printf '  %-34s %8s   model weights, re-download\n' \
    "$HF" "$(human "$(size_of "$HF")")"
for v in "$IDM_ROOT/venv" "$IDM_ROOT/venv-qwen"; do
    [ -d "$v" ] && printf '  %-34s %8s   rebuilt from requirements.txt\n' \
        "$v" "$(human "$(size_of "$v")")"
done

echo
if [ "$APPLY" = "1" ]; then
    AFTER="$(df -B1 --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "free now: $(human "${AFTER:-0}")  (was $(human "${BEFORE:-0}"))"
else
    echo "reclaimable: about $(human "$FREED")"
    echo "would leave: about $(human $((${BEFORE:-0} + FREED)))"
    echo
    echo "Re-run with --yes to delete. Nothing above is unique - every item is"
    echo "a cache that regenerates on demand."
fi

# The reason anyone runs this script: is there room to build an environment?
# 8 GB installed was the old figure and it was the wrong one - pip unpacks each
# wheel under TMPDIR before installing it, and torch is ~2.5 GB unpacked, so
# the peak is what has to fit. One shared environment (bootstrap_all.sh
# --single-torch) peaks around 10 GB; a second, separate torch stack on top of
# an existing env peaks around 11.
NEED=$((10 * 1024 * 1024 * 1024))
if [ "$APPLY" = "1" ]; then
    PROJ="$(df -B1 --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
else
    PROJ=$(( ${BEFORE:-0} + FREED ))   # what --yes would leave
fi
echo
if [ "${PROJ:-0}" -ge "$NEED" ]; then
    echo "That is enough for one shared environment (~10 GB at peak):"
    echo "    bash scripts/alaya/bootstrap_all.sh --single-torch"
    echo
    echo "A second, separate torch stack needs ~11 GB on top of an existing"
    echo "env, which is why --single-torch exists."
else
    SHORT=$((NEED - PROJ))
    echo "Still about $(human "$SHORT") short of the ~10 GB an environment"
    echo "needs at peak."
    echo
    echo
    echo "First try sharing one torch instead of installing two, ~6 GB less:"
    echo "    bash scripts/alaya/bootstrap_all.sh --single-torch"
    echo
    echo "If even that will not fit: the stages never run together - stage 10"
    echo "writes PNGs that stages 20 and 30 read - so the two envs do not have"
    echo "to coexist. Swap them:"
    echo
    echo "    $IDM_ROOT/venv/bin/pip freeze > $IDM_ROOT/idm-venv.txt   # first!"
    echo "    rm -rf $IDM_ROOT/venv"
    echo "    bash scripts/pipeline/00_setup_qwen_env.sh"
    echo "    # ... run stage 10, then ..."
    echo "    rm -rf $IDM_ROOT/venv-qwen"
    echo "    bash scripts/alaya/00_bootstrap_workshop.sh"
    echo
    if [ -d "$HF" ]; then
        echo "The $(human "$(size_of "$HF")") of weights stay put throughout;"
    fi
    echo "only the envs move, and those rebuild from a mirror in minutes."
fi
