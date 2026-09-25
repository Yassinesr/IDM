#!/usr/bin/env bash
# Build a SECOND virtualenv for Qwen-Image-Edit.
#
#   bash scripts/pipeline/00_setup_qwen_env.sh
#
# Why separate: IDM-VTON is pinned to diffusers 0.25.0 because src/unet_hacked_*.py
# subclass its internals, and Qwen-Image needs a modern diffusers. The two cannot
# share an environment, so the pipeline stages talk through files on disk instead.

set -euo pipefail

# --overlay: install ONLY the new libraries into a directory that stage 10 puts
# first on sys.path, reusing the IDM-VTON env's torch. A few hundred MB rather
# than a second torch stack, which matters when the disk cannot hold one. The
# trade is that Qwen then runs on whatever torch that env has.
MODE=venv
[ "${1:-}" = "--overlay" ] && MODE=overlay

IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
# Overridable so this env can live on a different filesystem from IDM_ROOT.
# On a box where the container disk is full but some other mount is not, that
# is the difference between the two envs coexisting and swapping them in and
# out. Whatever mount you pick, treat it as scratch: only the weights are
# expensive, and a venv rebuilds from a script.
QWEN_VENV="${QWEN_VENV:-$IDM_ROOT/venv-qwen}"
PIP_INDEX_URL="${PIP_INDEX_URL-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/alaya/_activate.sh"

if [ "$MODE" = "overlay" ]; then
    OVERLAY="$IDM_ROOT/qwen-overlay"
    echo "==> overlay mode: $OVERLAY (reusing the IDM-VTON env's torch)"
    # shellcheck disable=SC1091
    source "$REPO_DIR/scripts/alaya/env.sh" >/dev/null 2>&1 || true
    command -v python >/dev/null 2>&1 \
        || { echo "ERROR: activate the IDM-VTON env first: source scripts/alaya/env.sh" >&2; exit 1; }
    echo "    base torch: $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
    mkdir -p "$OVERLAY"
    [ -n "$PIP_INDEX_URL" ] && export PIP_INDEX_URL
    # --no-deps deliberately: resolving normally would drag in torch and undo
    # the entire point. That means every package whose version must be NEWER
    # than the IDM-VTON env's has to be named here. huggingface_hub is one:
    # the env pins 0.25.2 for diffusers 0.25.0, and modern diffusers needs
    # DDUFEntry, added in 0.27. Overriding it is safe because the overlay is
    # only on sys.path when QWEN_OVERLAY is set - stages 20 and 30 still see
    # the pinned 0.25.2.
    PKGS="${QWEN_OVERLAY_PKGS:-diffusers>=0.35 transformers>=4.51 tokenizers huggingface_hub>=0.27 safetensors accelerate}"
    # shellcheck disable=SC2086
    pip install --target "$OVERLAY" --upgrade --no-deps $PKGS
    echo
    echo "    $(du -sh "$OVERLAY" | cut -f1) installed"
    cat <<EOF

Use it by exporting QWEN_OVERLAY - no second env to activate:

    source scripts/alaya/env.sh
    export QWEN_OVERLAY=$OVERLAY
    python scripts/pipeline/10_generate_views.py --reference ... --count 3

If an import fails naming a symbol the base env's older copy lacks, that
package needs adding to the overlay too:

    QWEN_OVERLAY_PKGS="diffusers>=0.35 transformers>=4.51 tokenizers \
        huggingface_hub>=0.27 safetensors accelerate <the-missing-one>" \
        bash scripts/pipeline/00_setup_qwen_env.sh --overlay

If Qwen refuses to run on this torch, that is the trade-off of overlay mode;
the full venv (no --overlay) is the clean answer once there is disk for it.
EOF
    exit 0
fi

echo "==> target: $QWEN_VENV"

# pip keeps every wheel it downloads. Building a torch stack that way needs the
# download AND the unpacked copy on disk at once - roughly 2.5 GB of headroom
# bought for a reinstall that will never happen on a container this size.
export PIP_NO_CACHE_DIR=1

# pip unpacks each wheel under TMPDIR before installing it, and torch is ~2.5 GB
# unpacked. TMPDIR defaults to /tmp, which on this box is the same filesystem as
# everything else - so with QWEN_VENV pointed at a roomier mount, the install
# would still fail against / unless the scratch space follows it there.
QWEN_PARENT="$(dirname "$QWEN_VENV")"
mkdir -p "$QWEN_PARENT" || { echo "ERROR: cannot create $QWEN_PARENT" >&2; exit 1; }
if [ -z "${TMPDIR:-}" ]; then
    TMPDIR="$QWEN_PARENT/tmp-qwen-build"
    mkdir -p "$TMPDIR" && export TMPDIR \
        || { echo "ERROR: cannot create $TMPDIR" >&2; exit 1; }
    # Leaving several GB of unpacked wheels behind on a full disk would be a
    # poor thanks for the space.
    trap 'rm -rf "$TMPDIR"' EXIT
fi

# Measure the filesystem the env is actually going on, which is not IDM_ROOT's
# whenever QWEN_VENV has been pointed elsewhere.
FREE_GB="$(df -BG --output=avail "$QWEN_PARENT" 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "==> free on $QWEN_PARENT: ${FREE_GB:-?} GB (pip caching disabled)"
echo "==> build scratch: $TMPDIR"
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 12 ]; then
    echo >&2
    echo "WARNING: only ${FREE_GB} GB here. The env lands at roughly 8 GB, and" >&2
    echo "         pip needs a few GB more while it unpacks torch, so 12 GB is" >&2
    echo "         the realistic floor. The Qwen weights cost nothing - they" >&2
    echo "         are read off /root/public in place - but this is not free." >&2
    echo "         If it dies with ENOSPC:" >&2
    echo "           rm -rf $QWEN_VENV          # the partial install" >&2
    echo "           bash scripts/alaya/reclaim_disk.sh" >&2
    echo "         or put it on a mount with room:" >&2
    echo "           QWEN_VENV=/other/mount/venv-qwen \\" >&2
    echo "             bash scripts/pipeline/00_setup_qwen_env.sh" >&2
    echo >&2
fi

if [ -f "$QWEN_VENV/bin/activate" ] || [ -d "$QWEN_VENV/conda-meta" ]; then
    echo "==> environment already exists"
else
    # Reuse the Miniconda the IDM bootstrap installed, if it is there.
    if [ -x "$IDM_ROOT/miniconda/bin/conda" ]; then
        echo "==> creating conda env (python 3.10)"
        "$IDM_ROOT/miniconda/bin/conda" create -p "$QWEN_VENV" python=3.10 -y \
            --override-channels -c "${CONDA_CHANNEL:-conda-forge}"
    else
        echo "==> creating venv with $(python3 -V)"
        python3 -m venv "$QWEN_VENV"
    fi
fi

IDM_VENV="$QWEN_VENV" idm_activate \
    || { echo "ERROR: could not activate $QWEN_VENV" >&2; exit 1; }

[ -n "$PIP_INDEX_URL" ] && export PIP_INDEX_URL
python -m pip install --upgrade pip wheel

# Qwen-Image is a 20B MMDiT; it wants a recent torch for its attention kernels.
python -c 'import torch' 2>/dev/null \
    || pip install torch torchvision --index-url "$TORCH_INDEX_URL"

# Deliberately unpinned: this env exists to track current diffusers, which is
# exactly what the IDM-VTON env cannot do.
pip install --upgrade "diffusers>=0.35" transformers accelerate safetensors \
    sentencepiece protobuf pillow

python - <<'PY'
import torch, diffusers, transformers
print(f"  torch        {torch.__version__}")
print(f"  diffusers    {diffusers.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  cuda         {torch.cuda.is_available()}")
PY

cat <<EOF

Done. This env is only for stage 10:

    source $QWEN_VENV/bin/activate
    python scripts/pipeline/10_generate_views.py --help

Stages 20 and 30 use the ORIGINAL env (source scripts/alaya/env.sh).
EOF
