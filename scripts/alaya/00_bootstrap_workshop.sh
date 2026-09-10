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
# IDM-VTON needs python 3.10. This is not a preference: torch 2.0.1 publishes no
# cp311/cp312 wheels at all, and neither do bitsandbytes 0.39.0 or
# onnxruntime 1.16.2. The Alaya base images ship 3.12, so we usually have to
# provision 3.10 ourselves via Miniconda (onto the PVC, so it survives).
find_py310() {
    for c in python3.10 python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        if [ "$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" = "3.10" ]; then
            command -v "$c"
            return 0
        fi
    done
    return 1
}

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_activate.sh"

PY_BIN="$(find_py310 || true)"
USE_CONDA=0

if [ -n "$PY_BIN" ]; then
    echo "==> image already has python 3.10: $PY_BIN"
else
    IMG_PY="$(python3 -V 2>&1 || echo 'none')"
    echo "==> image python is '$IMG_PY', which cannot run this stack"
    echo "    provisioning python 3.10 with Miniconda (one-off, ~5 min)"
    USE_CONDA=1
    CONDA_DIR="$IDM_ROOT/miniconda"
    if [ ! -x "$CONDA_DIR/bin/conda" ]; then
        MINICONDA_URL="${MINICONDA_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
        echo "    downloading $MINICONDA_URL"
        curl -fsSL "$MINICONDA_URL" -o "$IDM_ROOT/miniconda.sh" \
            || { echo "ERROR: Miniconda download failed. Set MINICONDA_URL to a reachable mirror." >&2; exit 1; }
        bash "$IDM_ROOT/miniconda.sh" -b -p "$CONDA_DIR"
        rm -f "$IDM_ROOT/miniconda.sh"
    else
        echo "    reusing $CONDA_DIR"
    fi
fi

# ------------------------------------------------------------------- env ----
if [ -f "$IDM_VENV/bin/activate" ] || [ -d "$IDM_VENV/conda-meta" ]; then
    echo "==> environment already exists at $IDM_VENV"
elif [ "$USE_CONDA" = "1" ]; then
    echo "==> creating conda env (python 3.10) at $IDM_VENV"
    "$IDM_ROOT/miniconda/bin/conda" create -p "$IDM_VENV" python=3.10 -y
else
    echo "==> creating venv at $IDM_VENV"
    "$PY_BIN" -m venv "$IDM_VENV"
fi

idm_activate || { echo "ERROR: could not activate $IDM_VENV" >&2; exit 1; }

ACTUAL_PY="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "==> env python $ACTUAL_PY at $(command -v python)"
if [ "$ACTUAL_PY" != "3.10" ]; then
    echo "ERROR: this environment is python $ACTUAL_PY, not 3.10." >&2
    echo "       It is probably left over from an earlier run. Delete and retry:" >&2
    echo "         rm -rf $IDM_VENV" >&2
    echo "         bash scripts/alaya/00_bootstrap_workshop.sh" >&2
    exit 1
fi

export PIP_CACHE_DIR="$IDM_ROOT/pipcache"
[ -n "$PIP_INDEX_URL" ] && pip config set global.index-url "$PIP_INDEX_URL" >/dev/null
pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------- torch ----
# torch and torchvision must be a matching pair; pip will not work that out on
# its own, so map them explicitly.
tv_for() {
    case "$1" in
        2.0.1) echo 0.15.2 ;;
        2.2.2) echo 0.17.2 ;;
        2.4.1) echo 0.19.1 ;;
        2.5.1) echo 0.20.1 ;;
        *)     echo "" ;;
    esac
}

# Preference order. 2.0.1 is what environment.yaml specifies, but nothing in
# this project actually requires it: the only dependency that needed
# torchvision < 0.17 was basicsr, which is never imported (see requirements.txt).
# The fallbacks exist because some mirrors do not carry 2.0.1 at all -
# mirror.sjtu.edu.cn/pytorch-wheels/cu118 starts at 2.2.0.
# Keep every candidate on cu118 or newer: an H800 is sm_90, which cu117 builds
# have no kernels for.
TORCH_CANDIDATES="${TORCH_VERSION:-2.0.1 2.2.2 2.4.1 2.5.1}"

if python -c 'import torch' 2>/dev/null; then
    echo "==> torch already installed: $(python -c 'import torch;print(torch.__version__)')"
else
    echo "==> looking for torch on $TORCH_INDEX_URL"
    AVAIL="$(pip index versions torch --index-url "$TORCH_INDEX_URL" 2>/dev/null || true)"
    PICK=""
    for cand in $TORCH_CANDIDATES; do
        # An empty AVAIL means the probe is unsupported, not that nothing is
        # there - fall through to the first candidate and let pip decide.
        if [ -z "$AVAIL" ] || printf '%s' "$AVAIL" | grep -qF "$cand"; then
            PICK="$cand"; break
        fi
    done

    if [ -z "$PICK" ]; then
        echo "ERROR: none of [$TORCH_CANDIDATES] are on $TORCH_INDEX_URL" >&2
        echo "$AVAIL" >&2
        echo "       The official index carries all of them:" >&2
        echo "         TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \\" >&2
        echo "           bash scripts/alaya/00_bootstrap_workshop.sh" >&2
        exit 1
    fi

    TV="$(tv_for "$PICK")"
    [ -n "$TV" ] || { echo "ERROR: no torchvision mapping for torch $PICK" >&2; exit 1; }
    if [ "$PICK" != "2.0.1" ]; then
        echo "    2.0.1 is not on this index; using torch $PICK + torchvision $TV"
    fi
    echo "==> installing torch $PICK + torchvision $TV"
    pip install "torch==$PICK" "torchvision==$TV" --index-url "$TORCH_INDEX_URL"
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
