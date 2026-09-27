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
    # A venv used purely as an installer. --system-site-packages lets pip SEE
    # the IDM-VTON env's torch as already satisfied, so it resolves everything
    # else normally instead of downloading a second 6 GB torch.
    #
    # An earlier version used `pip install --target ... --no-deps`, which does
    # keep torch out but makes every transitive dependency something you have
    # to name by hand - and each one only shows up as an ImportError after the
    # install reports success. DDUFEntry, then httpx2, then the next one.
    # Real resolution ends that.
    #
    # What gets used afterwards is the venv's site-packages directory, put
    # first on sys.path by the Qwen stages themselves. Not the venv: activating
    # it would shadow diffusers 0.25.0 for stages 20 and 30 as well, and
    # run_all.sh runs all of them in one process.
    QWEN_ENV="$IDM_ROOT/qwen-env"
    OVERLAY="$IDM_ROOT/qwen-overlay"
    echo "==> overlay mode: $QWEN_ENV (reusing the IDM-VTON env's torch)"

    # shellcheck disable=SC1091
    source "$REPO_DIR/scripts/alaya/env.sh" >/dev/null 2>&1 || true
    command -v python >/dev/null 2>&1 \
        || { echo "ERROR: activate the IDM-VTON env first: source scripts/alaya/env.sh" >&2; exit 1; }
    BASE_TORCH="$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || true)"
    [ -n "$BASE_TORCH" ] || { echo "ERROR: the active env has no torch - nothing to reuse." >&2; exit 1; }
    echo "    base torch: $BASE_TORCH"
    case "$BASE_TORCH" in
        1.*|2.0.*|2.1.*|2.2.*|2.3.*)
            echo >&2
            echo "ERROR: diffusers >= 0.34 touches torch.xpu at import, which arrived" >&2
            echo "       in torch 2.4. This env has $BASE_TORCH, so an overlay on it" >&2
            echo "       cannot work. Rebuild the base env on 2.4.1:" >&2
            echo "         IDM_ALLOW_EPHEMERAL=1 bash scripts/alaya/bootstrap_all.sh --single-torch" >&2
            exit 1 ;;
    esac

    [ -n "$PIP_INDEX_URL" ] && export PIP_INDEX_URL
    export PIP_NO_CACHE_DIR=1

    if [ ! -f "$QWEN_ENV/pyvenv.cfg" ]; then
        rm -rf "$QWEN_ENV"
        echo "==> creating $QWEN_ENV (--system-site-packages)"
        python -m venv --system-site-packages "$QWEN_ENV" \
            || { echo "ERROR: could not create $QWEN_ENV" >&2; exit 1; }
    fi

    SITE="$("$QWEN_ENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
    [ -d "$SITE" ] || { echo "ERROR: cannot locate site-packages in $QWEN_ENV" >&2; exit 1; }

    # Import from the base interpreter with SITE first on sys.path - the exact
    # arrangement the stages use, so a pass here means they will work.
    # Goes through _qwen.py, so this tests the arrangement the stages actually
    # use - including the compatibility shims it installs - rather than a
    # simplified version of it.
    overlay_imports() {
        QWEN_OVERLAY="$SITE" REPO_DIR="$REPO_DIR" python - <<'PYCHECK'
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO_DIR"], "scripts", "pipeline"))
import _qwen

imported = _qwen.import_diffusers()
if imported is None:
    sys.exit("diffusers did not import")
diffusers, _ = imported

import torch, transformers
if tuple(int(x) for x in diffusers.__version__.split(".")[:2]) < (0, 36):
    sys.exit("diffusers < 0.36 has no QwenImageEditPlusPipeline")
from transformers.utils import is_torch_available
if not is_torch_available():
    sys.exit(f"transformers {transformers.__version__} disabled torch "
             f"{torch.__version__} - it wants a newer one")
from diffusers import QwenImageEditPlusPipeline  # noqa: F401

# Importing is not running. The attention backend is where a diffusers built
# for a newer torch actually breaks - enable_gqa= landed in 2.5 - and that
# failure otherwise appears in the first denoising step, after the 54 GB load.
# Four tiny tensors settle it in milliseconds.
from diffusers.models.attention_dispatch import dispatch_attention_fn
dev = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.randn(1, 2, 8, 16, device=dev, dtype=torch.float32)
out = dispatch_attention_fn(q, q.clone(), q.clone())
if out.shape != q.shape:
    sys.exit(f"attention returned {tuple(out.shape)}, expected {tuple(q.shape)}")

print(f"    torch        {torch.__version__}  (from the base env)")
print(f"    diffusers    {diffusers.__version__}")
print(f"    transformers {transformers.__version__}")
print(f"    attention    runs on {dev}")
print("    QwenImageEditPlusPipeline available")
PYCHECK
    }

    # Newer is not better here. diffusers 0.40 registers a custom op whose
    # annotations torch 2.4's infer_schema cannot parse, and transformers 5.x
    # disables torch outright below 2.5 - both are "works on a newer torch"
    # problems, and the base torch is fixed by IDM-VTON. So walk UP from the
    # oldest release that has QwenImageEditPlusPipeline and stop at the first
    # that imports, rather than guessing which pairing is good.
    if [ -n "${QWEN_OVERLAY_PKGS:-}" ]; then
        echo "==> installing (QWEN_OVERLAY_PKGS override, no probing)"
        # shellcheck disable=SC2086
        "$QWEN_ENV/bin/pip" install --upgrade $QWEN_OVERLAY_PKGS \
            || { echo "ERROR: install failed" >&2; exit 1; }
        overlay_imports || { echo "ERROR: the override does not import." >&2; exit 1; }
    else
        TF_SPEC="${QWEN_TRANSFORMERS_SPEC:-transformers>=4.51,<5}"
        MINORS="${QWEN_DIFFUSERS_MINORS:-36 37 38 39 40}"
        OK=0
        for m in $MINORS; do
            SPEC="diffusers>=0.$m,<0.$((m + 1))"
            echo
            echo "==> trying $SPEC with $TF_SPEC"
            PIPLOG="$(mktemp)"
            if ! "$QWEN_ENV/bin/pip" install --upgrade \
                    "$SPEC" "$TF_SPEC" accelerate safetensors >"$PIPLOG" 2>&1; then
                # Quiet by design - most failures here are just "no such
                # release" while probing. But a real one (no disk, no network)
                # would otherwise be indistinguishable, so show it.
                echo "    install failed:"
                tail -3 "$PIPLOG" | sed 's/^/      /'
                rm -f "$PIPLOG"
                continue
            fi
            rm -f "$PIPLOG"
            if overlay_imports; then OK=1; break; fi
            echo "    that pairing does not import on torch $BASE_TORCH"
        done
        if [ "$OK" != "1" ]; then
            cat >&2 <<'WHY'

ERROR: no diffusers release in the probed range imports on this torch.

  Every candidate either predates QwenImageEditPlusPipeline or needs a torch
  newer than the base environment's. The base torch is set by IDM-VTON, which
  is pinned to diffusers 0.25.0, so raising it is not free.

  Options, in order:

    1. Widen the probe, if a newer diffusers might work:
         QWEN_DIFFUSERS_MINORS="36 37 38 39 40 41" \
             bash scripts/pipeline/00_setup_qwen_env.sh --overlay

    2. Name an exact pairing you know works:
         QWEN_OVERLAY_PKGS="diffusers==0.36.0 transformers==4.51.3 \
             accelerate safetensors" \
             bash scripts/pipeline/00_setup_qwen_env.sh --overlay

    3. Give Qwen its own torch - a full second environment, ~8 GB:
         bash scripts/pipeline/00_setup_qwen_env.sh
WHY
            exit 1
        fi
    fi

    # If pip put torch in here, --system-site-packages did not take effect and
    # ~6 GB just landed on a disk that has none to spare.
    if [ -d "$SITE/torch" ]; then
        echo >&2
        echo "ERROR: torch was installed into the overlay - the base env's copy" >&2
        echo "       was not visible, so this is a full second stack, not an" >&2
        echo "       overlay. Remove it:  rm -rf $QWEN_ENV" >&2
        exit 1
    fi

    ln -sfn "$SITE" "$OVERLAY"
    echo
    echo "    $(du -sh "$SITE" | cut -f1) installed"
    echo "    $OVERLAY -> $SITE"

    cat <<EOF

Done. Stages 10, 25 and 35 use this; stages 20 and 30 use the base env's
pinned diffusers 0.25.0, untouched. env.sh exports QWEN_OVERLAY when the
symlink exists, so there is nothing to activate:

    source scripts/alaya/env.sh
    python scripts/pipeline/10_generate_views.py --reference ... --count 3

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
pip install --upgrade "diffusers>=0.36" transformers accelerate safetensors \
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
