#!/usr/bin/env bash
# Take a fresh container from bare to both pipelines working.
#
#   git clone -b <branch> git@github.com:Yassinesr/IDM.git /pvc/idm/IDM
#   cd /pvc/idm/IDM && bash scripts/alaya/bootstrap_all.sh
#
# Each stage is skippable and each is safe to re-run, so a failure part-way
# means fixing that one thing and running it again, not starting over.
#
#   --single-torch   build the IDM-VTON env on torch 2.4.1 and give Qwen an
#                    overlay on top of it, instead of a second torch stack.
#                    ~6 GB less, which is the difference between fitting on a
#                    49 GB disk and not. See the note at the Qwen stage.
#   --qwen-only      stage 10 box: Qwen env only, no IDM-VTON, no weights
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

SKIP_STORAGE=0 SKIP_IDM=0 SKIP_WEIGHTS=0 SKIP_QWEN=0 SLIM="" SINGLE_TORCH=0
for a in "$@"; do case "$a" in
    --single-torch) SINGLE_TORCH=1 ;;
    --qwen-only)    SKIP_IDM=1; SKIP_WEIGHTS=1 ;;
    --skip-storage) SKIP_STORAGE=1 ;; --skip-idm) SKIP_IDM=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;; --skip-qwen) SKIP_QWEN=1 ;;
    --slim) SLIM="--slim" ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

# On a Qwen-only box there is no IDM-VTON env to source and nothing for
# preflight to check - it would fail on a machine that is entirely correct.
IDM_STAGES=1
[ "$SKIP_IDM" = "1" ] && [ "$SKIP_WEIGHTS" = "1" ] && IDM_STAGES=0

step() { echo; echo "════════ $* ════════"; }
fail() { echo; echo "FAILED at: $*" >&2
         echo "Fix it, then re-run - completed stages are skipped automatically." >&2
         exit 1; }

# 2.4.1 is the floor: diffusers >= 0.34 touches torch.xpu at import, and that
# arrived in torch 2.4. Nothing in requirements.txt caps torch - the default
# 2.0.1 was chosen for an index that carried it, not for a dependency.
if [ "$SINGLE_TORCH" = "1" ]; then
    TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
    export TORCH_VERSION
fi

echo "repo     $REPO_DIR"
echo "IDM_ROOT $IDM_ROOT"
[ "$SINGLE_TORCH" = "1" ] && echo "torch    $TORCH_VERSION (shared with Qwen via an overlay)"

# ---------------------------------------------------------------- storage ---
# First, because everything below is wasted if it does not persist.
if [ "$SKIP_STORAGE" = "0" ]; then
    step "1/5  storage"
    bash scripts/alaya/check_storage.sh || fail "storage check (see §2.1.1)"
    FREE_GB="$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "free on $IDM_ROOT: ${FREE_GB:-?} GB"
    NEED=60
    [ "$SINGLE_TORCH" = "1" ] && NEED=52
    if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt "$NEED" ]; then
        echo "WARNING: under $NEED GB. Weights ~31 (or ~17 with --slim), repo ~2," >&2
        if [ "$SINGLE_TORCH" = "1" ]; then
            echo "         one shared env ~7, Qwen overlay ~0.5. Tight." >&2
        else
            echo "         IDM env ~5, Qwen env ~8. Two torch stacks do not fit" >&2
            echo "         on a 49 GB disk alongside the weights - try" >&2
            echo "         --single-torch." >&2
        fi
    fi
fi

# ------------------------------------------------------------- idm-vton -----
if [ "$SKIP_IDM" = "0" ]; then
    step "2/5  IDM-VTON environment"
    bash scripts/alaya/00_bootstrap_workshop.sh || fail "IDM-VTON env"
fi

if [ "$IDM_STAGES" = "1" ]; then
    # shellcheck disable=SC1091
    source scripts/alaya/env.sh || fail "sourcing env.sh"
fi

# -------------------------------------------------------------- weights -----
if [ "$SKIP_WEIGHTS" = "0" ]; then
    step "3/5  checkpoints"
    python scripts/alaya/find_local_models.py 2>/dev/null | head -20 || true
    python scripts/alaya/01_download_checkpoints.py $SLIM || fail "checkpoint download"
fi

# ------------------------------------------------------------- preflight ----
if [ "$IDM_STAGES" = "1" ]; then
    step "4/5  preflight"
    python scripts/alaya/preflight.py || fail "preflight - fix the FAIL lines above"
fi

# ----------------------------------------------------------------- qwen -----
if [ "$SKIP_QWEN" = "0" ]; then
    step "5/5  Qwen environment"
    # QwenImageEditPlusPipeline needs diffusers >= 0.36, which touches torch.xpu
    # at import and so needs torch >= 2.4. With the default torch 2.0.1 that
    # means a second, complete torch stack. With --single-torch the IDM-VTON env
    # is already on 2.4.1, so Qwen needs only the newer libraries on top -
    # a few hundred MB against ~8 GB.
    if [ "$SINGLE_TORCH" = "1" ]; then
        bash scripts/pipeline/00_setup_qwen_env.sh --overlay || fail "Qwen overlay"
    else
        bash scripts/pipeline/00_setup_qwen_env.sh || fail "Qwen env"
    fi
fi

cat <<EOF

════════ done ════════

$([ "$IDM_STAGES" = "0" ] && echo "This box runs stage 10 only. Copy work/variants/ to the IDM-VTON box
for stages 20 and 30." || echo "IDM-VTON (stages 20, 30, and the try-on demo):")
    source scripts/alaya/env.sh
    python scripts/alaya/demo_single.py --category lower_body \\
        --human gradio_demo/example/human/model_front.jpg \\
        --garment gradio_demo/example/cloth/yoga_pants.jpg \\
        --desc "plain mauve leggings" --save-mask

$([ "$SINGLE_TORCH" = "1" ] && echo "Qwen (stages 10, 25, 35) - same env, overlay on sys.path. env.sh
exports QWEN_OVERLAY when the directory exists, so there is nothing else
to activate:" || echo "Qwen (stages 10, 25, 35) - separate env:
    source \$IDM_ROOT/venv-qwen/bin/activate   # conda env: PATH is set by env.sh")
    python scripts/pipeline/10_generate_views.py \\
        --reference gradio_demo/example/human/model_front.jpg --count 3

$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9') GB still free on $IDM_ROOT
EOF
