#!/usr/bin/env bash
# Build a SECOND virtualenv for Qwen-Image-Edit.
#
#   bash scripts/pipeline/00_setup_qwen_env.sh
#
# Why separate: IDM-VTON is pinned to diffusers 0.25.0 because src/unet_hacked_*.py
# subclass its internals, and Qwen-Image needs a modern diffusers. The two cannot
# share an environment, so the pipeline stages talk through files on disk instead.

set -euo pipefail

IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
QWEN_VENV="$IDM_ROOT/venv-qwen"
PIP_INDEX_URL="${PIP_INDEX_URL-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/alaya/_activate.sh"

echo "==> target: $QWEN_VENV"
FREE_GB="$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "==> free on $IDM_ROOT: ${FREE_GB:-?} GB"
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 12 ]; then
    echo "WARNING: this venv needs roughly 10 GB (torch is most of it)." >&2
    echo "         The model weights themselves cost nothing - they are read" >&2
    echo "         straight off /root/public - but the venv is not free." >&2
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
