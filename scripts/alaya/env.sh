#!/usr/bin/env bash
# Source this at the start of every Workshop session:
#
#   source scripts/alaya/env.sh
#
# A Workshop container's root filesystem is wiped when the Workshop is deleted
# or rescheduled. Only the PVC mount survives, so the venv, the Hugging Face
# cache, the pip cache and the datasets all live under $IDM_ROOT.

# ---- Point this at YOUR PVC mount path -----------------------------------
# It must match the "PVC MOUNTS" mount path you chose in the Aladdin extension
# when creating the Workshop. Check it with `df -h` inside the Workshop.
export IDM_ROOT="${IDM_ROOT:-/pvc/idm}"
# --------------------------------------------------------------------------

export IDM_VENV="$IDM_ROOT/venv"
export IDM_REPO="${IDM_REPO:-$IDM_ROOT/IDM-VTON}"

# Keep every cache on the PVC so a rebuilt Workshop does not re-download 10s of GB.
export HF_HOME="$IDM_ROOT/hf"
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

echo "[env.sh] IDM_ROOT=$IDM_ROOT  HF_ENDPOINT=$HF_ENDPOINT  python=$(command -v python)"
