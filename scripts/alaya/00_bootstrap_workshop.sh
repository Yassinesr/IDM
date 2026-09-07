#!/usr/bin/env bash
# Build the IDM-VTON python environment inside an Alaya NeW Workshop.
#
# This is the "official image + your own venv" path that the platform manual
# recommends over building a custom Docker image.
#
#   export IDM_ROOT=/pvc/idm          # your PVC mount path
#   bash scripts/alaya/00_bootstrap_workshop.sh
#
# Safe to re-run: it skips work that is already done.

set -euo pipefail

IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
IDM_VENV="$IDM_ROOT/venv"
# Tsinghua mirror, as used by the platform manual. Set PIP_INDEX_URL=""
# beforehand to use PyPI directly.
PIP_INDEX_URL="${PIP_INDEX_URL-https://pypi.tuna.tsinghua.edu.cn/simple}"
# Official PyTorch wheels. If download.pytorch.org is slow from the cluster,
# use the SJTU mirror instead:
#   TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "==> repo:     $REPO_DIR"
echo "==> IDM_ROOT: $IDM_ROOT"

if ! mountpoint -q "$IDM_ROOT" 2>/dev/null && [ ! -d "$IDM_ROOT" ]; then
    echo "ERROR: $IDM_ROOT does not exist." >&2
    echo "       Set IDM_ROOT to the PVC mount path you chose in Aladdin's" >&2
    echo "       'PVC MOUNTS' field. Run 'df -h' to list the mounts." >&2
    exit 1
fi

mkdir -p "$IDM_ROOT"/{hf,torch,pipcache,cache,data}

# ---------------------------------------------------------------- python ----
# IDM-VTON targets python 3.10. 3.11+ has no bitsandbytes 0.39.0 wheel and
# onnxruntime 1.16.2 stops at 3.10 for some platforms.
PY_BIN=""
for c in python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
        if [ "$v" = "3.10" ]; then PY_BIN="$c"; break; fi
        [ -z "$PY_BIN" ] && PY_BIN="$c"
    fi
done
[ -n "$PY_BIN" ] || { echo "ERROR: no python found in the image." >&2; exit 1; }

PY_VER="$("$PY_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "==> using $PY_BIN (python $PY_VER)"
if [ "$PY_VER" != "3.10" ]; then
    echo "WARNING: python $PY_VER, not 3.10. If pip cannot resolve" >&2
    echo "         bitsandbytes/onnxruntime, pick a Workshop image with 3.10" >&2
    echo "         or create the env with conda:" >&2
    echo "           conda create -p $IDM_VENV python=3.10 -y" >&2
fi

# ------------------------------------------------------------------ venv ----
if [ ! -f "$IDM_VENV/bin/activate" ]; then
    echo "==> creating venv at $IDM_VENV"
    "$PY_BIN" -m venv "$IDM_VENV"
else
    echo "==> venv already exists at $IDM_VENV"
fi
# shellcheck disable=SC1091
source "$IDM_VENV/bin/activate"

export PIP_CACHE_DIR="$IDM_ROOT/pipcache"
[ -n "$PIP_INDEX_URL" ] && pip config set global.index-url "$PIP_INDEX_URL" >/dev/null
pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------- torch ----
if python -c 'import torch' 2>/dev/null; then
    echo "==> torch already installed: $(python -c 'import torch;print(torch.__version__)')"
else
    echo "==> installing torch 2.0.1 + cu118 from $TORCH_INDEX_URL"
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url "$TORCH_INDEX_URL"
fi

# ------------------------------------------------------------------ deps ----
# Must come after torch: basicsr imports torch in its setup.py.
echo "==> installing requirements.txt"
pip install -r "$REPO_DIR/requirements.txt"

# ----------------------------------------------------------------- check ----
echo
echo "==> verifying"
python - <<'PY'
import torch, diffusers, transformers, huggingface_hub, onnxruntime
print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    gpu {i}: {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
print(f"  diffusers        {diffusers.__version__}")
print(f"  transformers     {transformers.__version__}")
print(f"  huggingface_hub  {huggingface_hub.__version__}")
print(f"  onnxruntime      {onnxruntime.__version__}")
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # trips the cached_download bug
print("  diffusers imports cleanly")
PY

cat <<EOF

Done. From now on, at the start of every Workshop session:

    export IDM_ROOT=$IDM_ROOT
    source $REPO_DIR/scripts/alaya/env.sh

Next: download the checkpoints
    python scripts/alaya/01_download_checkpoints.py
EOF
