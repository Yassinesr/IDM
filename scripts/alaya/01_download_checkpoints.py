#!/usr/bin/env python
"""Fetch every checkpoint IDM-VTON needs, mirror-aware.

The ckpt/* files committed to this repo are placeholders ("put X here"), and
the main model is pulled from the Hub at run time. On Alaya NeW, huggingface.co
is not reachable, so this goes through $HF_ENDPOINT (hf-mirror.com by default,
set in scripts/alaya/env.sh).

    source scripts/alaya/env.sh
    python scripts/alaya/01_download_checkpoints.py            # inference
    python scripts/alaya/01_download_checkpoints.py --training # + IP-Adapter

Everything lands under $HF_HOME (on the PVC) and is hard-linked/copied into
ckpt/, so a rebuilt Workshop re-uses the download instead of repeating it.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# The model repo behind `from_pretrained("yisol/IDM-VTON")`: unet, unet_encoder,
# vae, both text encoders, image_encoder, tokenizers, scheduler.
MODEL_REPO = "yisol/IDM-VTON"

# Preprocessing checkpoints. The README points at the *Space*, not the model
# repo - these files only exist there.
SPACE_REPO = "yisol/IDM-VTON"
SPACE_FILES = {
    "ckpt/densepose/model_final_162be9.pkl": "ckpt/densepose/model_final_162be9.pkl",
    "ckpt/humanparsing/parsing_atr.onnx": "ckpt/humanparsing/parsing_atr.onnx",
    "ckpt/humanparsing/parsing_lip.onnx": "ckpt/humanparsing/parsing_lip.onnx",
    "ckpt/openpose/ckpts/body_pose_model.pth": "ckpt/openpose/ckpts/body_pose_model.pth",
}

# Training only (train_xl.py). Inference reads image_encoder from MODEL_REPO.
IPADAPTER_REPO = "h94/IP-Adapter"
IPADAPTER_FILES = {
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin": "ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin",
    "models/image_encoder/config.json": "ckpt/image_encoder/config.json",
    "models/image_encoder/model.safetensors": "ckpt/image_encoder/model.safetensors",
}

PLACEHOLDER_MAX = 4096  # committed placeholders are a few dozen bytes


def place(cached: str, rel_dest: str) -> None:
    dest = REPO / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > PLACEHOLDER_MAX:
        print(f"    kept {rel_dest} ({dest.stat().st_size / 2**20:.1f} MiB)")
        return
    if dest.exists():
        dest.unlink()  # drop the placeholder
    try:
        os.link(cached, dest)  # same filesystem: free
    except OSError:
        shutil.copy2(cached, dest)
    print(f"    -> {rel_dest} ({dest.stat().st_size / 2**20:.1f} MiB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true",
                    help="also fetch the IP-Adapter weights needed by train_xl.py")
    ap.add_argument("--skip-model", action="store_true",
                    help="only fetch the small preprocessing checkpoints")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is missing - run 00_bootstrap_workshop.sh first.",
              file=sys.stderr)
        return 1

    print(f"HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"HF_HOME     = {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    if not os.environ.get("HF_HOME"):
        print("WARNING: HF_HOME is unset, so the cache lands on the container's\n"
              "         ephemeral disk and is lost with the Workshop.\n"
              "         Run `source scripts/alaya/env.sh` first.", file=sys.stderr)

    if not args.skip_model:
        print(f"\n[1/2] {MODEL_REPO} (model) -> HF cache")
        print("      Tens of GB. Resumable: re-run this script if it drops.")
        # Resuming is the default in huggingface_hub 0.23+; asking for it
        # explicitly only earns a deprecation warning.
        snapshot_download(repo_id=MODEL_REPO, max_workers=4)
        print("      ok")

    print(f"\n[2/2] {SPACE_REPO} (space) -> ckpt/")
    for remote, local in SPACE_FILES.items():
        cached = hf_hub_download(repo_id=SPACE_REPO, repo_type="space",
                                 filename=remote)
        place(cached, local)

    if args.training:
        print(f"\n[extra] {IPADAPTER_REPO} -> ckpt/")
        for remote, local in IPADAPTER_FILES.items():
            cached = hf_hub_download(repo_id=IPADAPTER_REPO,
                                     filename=remote)
            place(cached, local)

    print("\nAll checkpoints in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
