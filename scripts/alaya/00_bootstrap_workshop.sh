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

case "$IDM_ROOT" in
    /*) ;;
    *)  echo "ERROR: IDM_ROOT must be an absolute path, got '$IDM_ROOT'." >&2
        echo "       A relative value is resolved against your current directory," >&2
        echo "       which is how 'pvc/idm' becomes '$PWD/pvc/idm'." >&2
        exit 1 ;;
esac

if ! mountpoint -q "$IDM_ROOT" 2>/dev/null && [ ! -d "$IDM_ROOT" ]; then
    echo "ERROR: $IDM_ROOT does not exist." >&2
    echo "" >&2
    # Naming a wrong path is far more common than having no storage, so show
    # what is actually here rather than only saying to go and look.
    _root_dev="$(stat -c %d / 2>/dev/null)"
    _found=0
    while read -r _src _mnt _fstype _opts _; do
        [ -d "$_mnt" ] || continue
        [ "$_mnt" = "/" ] && continue
        case "$_fstype" in tmpfs|devtmpfs|proc|sysfs|cgroup*|devpts|squashfs|overlay) continue ;; esac
        case ",$_opts," in *,ro,*) continue ;; esac
        [ "$(stat -c %d "$_mnt" 2>/dev/null)" = "$_root_dev" ] && continue
        [ "$_found" = "0" ] && echo "       Writable non-root mounts that look like candidates:" >&2
        _found=1
        printf '         %-20s (%s, %s free)\n' "$_mnt" "$_fstype" \
            "$(df -h --output=avail "$_mnt" 2>/dev/null | tail -1 | tr -d ' ')" >&2
    done < /proc/mounts
    if [ "$_found" = "1" ]; then
        echo "" >&2
        echo "       Did you mean one of those? e.g. export IDM_ROOT=/pvc/idm" >&2
    else
        echo "       No writable non-root mounts found - this container has no" >&2
        echo "       storage attached. Set Storage > Container Path when creating" >&2
        echo "       the Workshop." >&2
    fi
    exit 1
fi

# A directory existing is not proof it is a PVC: `mkdir -p /pvc/idm` on the
# container's own root filesystem looks identical and silently works, right up
# until the Workshop is released and tens of GB of weights vanish with it. The
# platform is explicit about this - 系统盘为临时工作空间，变更内容在容器实例
# 释放后消失. Compare device numbers: same device as / means no PVC.
if [ "$(stat -c %d "$IDM_ROOT" 2>/dev/null)" = "$(stat -c %d / 2>/dev/null)" ]; then
    _free_gb="$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
    echo "ERROR: $IDM_ROOT is on the container's root filesystem, not a PVC." >&2
    echo "       It has ${_free_gb:-?} GB free and is ephemeral: released with the" >&2
    echo "       container. A graceful shutdown (关机) saves an image and keeps it;" >&2
    echo "       a release (释放) does not." >&2
    echo "" >&2
    echo "       Filesystems actually mounted here:" >&2
    df -h | grep -v "^tmpfs" | sed 's/^/         /' >&2
    echo "" >&2
    echo "       Fix: recreate the Workshop with Storage > Container Path set" >&2
    echo "       (see docs/alaya-new-setup.md section 2.1.1), then point" >&2
    echo "       IDM_ROOT at that mount." >&2
    # Print the real path, not $0: pasted into an interactive shell, $0 expands
    # to the shell itself and gives "cannot execute binary file".
    echo "       To accept the loss and continue anyway:" >&2
    echo "         IDM_ALLOW_EPHEMERAL=1 bash ${BASH_SOURCE[0]}" >&2
    [ "${IDM_ALLOW_EPHEMERAL:-0}" = "1" ] || exit 1
    echo "WARNING: IDM_ALLOW_EPHEMERAL=1 - continuing on ephemeral disk." >&2
fi

mkdir -p "$IDM_ROOT"/{hf,torch,pipcache,cache,data}
# HF_HOME may point outside IDM_ROOT when the cache is shared.
[ -n "${IDM_SHARED_HF:-}" ] && mkdir -p "$IDM_SHARED_HF"

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
    # --override-channels -c conda-forge, deliberately, rather than conda's
    # defaults. Two reasons: repo.anaconda.com now refuses non-interactive use
    # until its Terms of Service are accepted (CondaToSNonInteractiveError), and
    # those same defaults require a paid licence for larger organisations.
    # conda-forge has neither constraint and carries python 3.10 fine.
    _conda="$IDM_ROOT/miniconda/bin/conda"
    _tuna_forge="https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
    if ! "$_conda" create -p "$IDM_VENV" python=3.10 -y \
            --override-channels -c "${CONDA_CHANNEL:-conda-forge}"; then
        echo "    conda-forge failed; retrying via the Tsinghua mirror" >&2
        rm -rf "$IDM_VENV"
        "$_conda" create -p "$IDM_VENV" python=3.10 -y \
            --override-channels -c "$_tuna_forge"
    fi
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

# The pip cache earns its keep on a PVC (a rebuilt Workshop reuses it) but is
# dead weight on the ephemeral disk, where a rebuild means a new container
# anyway - and there it costs several GB of a ~28 GB budget.
if [ "${IDM_ALLOW_EPHEMERAL:-0}" = "1" ]; then
    export PIP_NO_CACHE_DIR=1
    echo "==> ephemeral disk: pip cache disabled to save space"
else
    export PIP_CACHE_DIR="$IDM_ROOT/pipcache"
fi
# Deliberately an env var, NOT `pip config set`: that writes to the user-level
# ~/.config/pip/pip.conf, which on a shared Workshop changes pip's index for
# everyone else using that home directory. This stays scoped to this process.
[ -n "$PIP_INDEX_URL" ] && export PIP_INDEX_URL
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
#
# In practice the fallback is now the normal path, not an edge case: as of 2026
# neither download.pytorch.org/whl/cu118 nor mirror.sjtu.edu.cn carries 2.0.1
# any more - both start at 2.2.0 - so this usually settles on 2.2.2. 2.0.1 stays
# first so that an index which does still have it is preferred.
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
