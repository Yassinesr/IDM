#!/usr/bin/env bash
# Take a fresh container from bare to both pipelines working.
#
#   git clone -b <branch> git@github.com:Yassinesr/IDM.git /pvc/idm/IDM
#   cd /pvc/idm/IDM && bash scripts/alaya/bootstrap_all.sh
#
# Each stage is skippable and each is safe to re-run, so a failure part-way
# means fixing that one thing and running it again, not starting over.
#
#   --skip-storage   do not verify persistent storage first
#   --skip-idm       do not build the IDM-VTON env
#   --skip-weights   do not download checkpoints
#   --skip-qwen      do not build the Qwen env
#   --slim           download the smaller weight set (~half)

set -uo pipefail

IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
export IDM_ROOT
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

SKIP_STORAGE=0 SKIP_IDM=0 SKIP_WEIGHTS=0 SKIP_QWEN=0 SLIM=""
for a in "$@"; do case "$a" in
    --skip-storage) SKIP_STORAGE=1 ;; --skip-idm) SKIP_IDM=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;; --skip-qwen) SKIP_QWEN=1 ;;
    --slim) SLIM="--slim" ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

step() { echo; echo "════════ $* ════════"; }
fail() { echo; echo "FAILED at: $*" >&2
         echo "Fix it, then re-run - completed stages are skipped automatically." >&2
         exit 1; }

echo "repo     $REPO_DIR"
echo "IDM_ROOT $IDM_ROOT"

# ---------------------------------------------------------------- storage ---
# First, because everything below is wasted if it does not persist.
if [ "$SKIP_STORAGE" = "0" ]; then
    step "1/5  storage"
    bash scripts/alaya/check_storage.sh || fail "storage check (see §2.1.1)"
    FREE_GB="$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "free on $IDM_ROOT: ${FREE_GB:-?} GB"
    if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 60 ]; then
        echo "WARNING: under 60 GB. Weights ~32 (or ~17 with --slim), IDM env ~5," >&2
        echo "         Qwen env ~10, repo ~2. It will be tight." >&2
    fi
fi

# ------------------------------------------------------------- idm-vton -----
if [ "$SKIP_IDM" = "0" ]; then
    step "2/5  IDM-VTON environment"
    bash scripts/alaya/00_bootstrap_workshop.sh || fail "IDM-VTON env"
fi

# shellcheck disable=SC1091
source scripts/alaya/env.sh || fail "sourcing env.sh"

# -------------------------------------------------------------- weights -----
if [ "$SKIP_WEIGHTS" = "0" ]; then
    step "3/5  checkpoints"
    python scripts/alaya/find_local_models.py 2>/dev/null | head -20 || true
    python scripts/alaya/01_download_checkpoints.py $SLIM || fail "checkpoint download"
fi

# ------------------------------------------------------------- preflight ----
step "4/5  preflight"
python scripts/alaya/preflight.py || fail "preflight - fix the FAIL lines above"

# ----------------------------------------------------------------- qwen -----
if [ "$SKIP_QWEN" = "0" ]; then
    step "5/5  Qwen environment"
    # A full venv, not --overlay: QwenImageEditPlusPipeline needs diffusers
    # >= 0.36, which touches torch.xpu at import and so needs torch >= 2.4.
    # The IDM-VTON env's torch 2.0.1 cannot host it.
    bash scripts/pipeline/00_setup_qwen_env.sh || fail "Qwen env"
fi

cat <<EOF

════════ done ════════

IDM-VTON (stages 20, 30, and the try-on demo):
    source scripts/alaya/env.sh
    python scripts/alaya/demo_single.py --category lower_body \\
        --human gradio_demo/example/human/model_front.jpg \\
        --garment gradio_demo/example/cloth/yoga_pants.jpg \\
        --desc "plain mauve leggings" --save-mask

Qwen (stage 10) - separate env:
    source \$IDM_ROOT/venv-qwen/bin/activate   # conda env: PATH is set by env.sh
    python scripts/pipeline/10_generate_views.py \\
        --reference gradio_demo/example/human/model_front.jpg --count 3

$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9') GB still free on $IDM_ROOT
EOF
