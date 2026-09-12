#!/usr/bin/env bash
# Source this at the start of every Workshop session:
#
#   source scripts/alaya/env.sh
#
# A Workshop container's root filesystem is wiped when the Workshop is deleted
# or rescheduled. Only the PVC mount survives, so the venv, the Hugging Face
# cache, the pip cache and the datasets all live under $IDM_ROOT.

# ---- Who you are, on a shared PVC ----------------------------------------
# Everyone inside a Workshop container is root, so this cannot be detected -
# set it yourself if other people share this storage:
#   export IDM_USER=yassine
# It keeps your env, repo and outputs in their own subtree, so two people do not
# end up fighting over one venv or deleting each other's work.
IDM_USER="${IDM_USER:-}"

# ---- Point this at YOUR PVC mount path -----------------------------------
# It must match the "PVC MOUNTS" mount path you chose in the Aladdin extension
# when creating the Workshop. Check it with `df -h` inside the Workshop.
if [ -n "$IDM_USER" ]; then
    export IDM_ROOT="${IDM_ROOT:-/pvc/users/$IDM_USER/idm}"
else
    export IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
fi
# --------------------------------------------------------------------------

export IDM_VENV="$IDM_ROOT/venv"
export IDM_REPO="${IDM_REPO:-$IDM_ROOT/IDM-VTON}"

# Keep every cache on the PVC so a rebuilt Workshop does not re-download 10s of GB.
#
# The model weights are tens of GB and byte-identical for everyone, so on a
# shared PVC it is worth pointing all users at one copy:
#   export IDM_SHARED_HF=/pvc/shared/hf
# The hub client takes a lock per file, so concurrent readers are fine and two
# people downloading the same repo at once will not corrupt it. Leave it unset
# to keep a private cache.
if [ -n "${IDM_SHARED_HF:-}" ]; then
    export HF_HOME="$IDM_SHARED_HF"
else
    export HF_HOME="$IDM_ROOT/hf"
fi
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$IDM_ROOT/torch"
export PIP_CACHE_DIR="$IDM_ROOT/pipcache"
export XDG_CACHE_HOME="$IDM_ROOT/cache"

# Alaya NeW sits behind the Great Firewall: huggingface.co is not routable from
# the cluster. hf-mirror.com is a full read-only mirror and the hub client
# honours HF_ENDPOINT. Unset it if your cluster has direct egress.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# gradio 4.x reads these. 0.0.0.0 lets the VS Code Remote port-forwarder and
# any in-cluster Service reach the demo; 127.0.0.1 only works for VS Code.
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

# The Workshop image ships a CUDA driver; nothing to set for CUDA itself.
# Uncomment to pin the run to one of the allocated GPUs.
# export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_activate.sh"
if ! idm_activate; then
    echo "[env.sh] No environment at $IDM_VENV - run scripts/alaya/00_bootstrap_workshop.sh first." >&2
fi

echo "[env.sh] IDM_ROOT=$IDM_ROOT  HF_HOME=$HF_HOME  python=$(command -v python)"
if [ -z "$IDM_USER" ] && [ -d /pvc/users ]; then
    echo "[env.sh] NOTE: /pvc/users exists, so this PVC is shared, but IDM_USER" >&2
    echo "         is unset - you are using the common $IDM_ROOT. Set IDM_USER to" >&2
    echo "         claim your own subtree." >&2
fi
