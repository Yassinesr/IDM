#!/usr/bin/env python
"""Search a local model library (e.g. Alaya's read-only /root/public) for the
weights IDM-VTON needs, so you can skip downloading tens of GB.

    python scripts/alaya/find_local_models.py                 # scans /root/public
    python scripts/alaya/find_local_models.py /some/other/dir --max-depth 6

Prints what it found and the exact flag or export to use it. Nothing is copied
or modified - the mount is read-only and this only reads directory names.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# `unet_encoder` is the giveaway: stock SDXL has `unet` but no `unet_encoder`,
# so a directory holding both is an IDM-VTON checkout whatever it is called.
IDM_MARKERS = {"unet", "unet_encoder"}
SDXL_MARKERS = {"unet", "vae", "text_encoder_2"}

# Preprocessing checkpoints, by exact filename.
CKPT_FILES = {
    "model_final_162be9.pkl": "ckpt/densepose/",
    "parsing_atr.onnx": "ckpt/humanparsing/",
    "parsing_lip.onnx": "ckpt/humanparsing/",
    "body_pose_model.pth": "ckpt/openpose/ckpts/",
    "ip-adapter-plus_sdxl_vit-h.bin": "ckpt/ip_adapter/",
}

NAME_HINTS = ("idm", "vton", "viton", "ip-adapter", "ip_adapter",
              "stable-diffusion-xl", "sdxl", "clip-vit")

# Subtrees that cannot hold the model and are enormous - on Alaya, datasets/ is
# 48 TB on its own. Skipping them is the difference between a 30s scan and one
# that never finishes.
SKIP_DIRS = {"datasets", "tmp", ".git", "__pycache__", "lost+found"}


def walk(root, max_depth, deadline, skip):
    """Bounded-depth scandir. A 461 TB mount will not be walked exhaustively."""
    stack = [(root, 0)]
    seen = 0
    while stack:
        if time.time() > deadline:
            print(f"\n  (time budget reached after {seen} directories; "
                  f"raise --max-seconds or --max-depth to go further)",
                  file=sys.stderr)
            return
        path, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(os.scandir(path))
        except (PermissionError, OSError):
            continue
        seen += 1
        if seen % 500 == 0:
            print(f"\r  scanned {seen} directories...", end="", file=sys.stderr,
                  flush=True)
        dirnames = {e.name for e in entries if e.is_dir(follow_symlinks=False)}
        filenames = {e.name for e in entries if e.is_file(follow_symlinks=False)}
        yield path, dirnames, filenames
        for e in entries:
            if e.is_dir(follow_symlinks=False) and e.name not in skip:
                stack.append((e.path, depth + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=None,
                    help="default: /root/public/models if it exists, else /root/public")
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-seconds", type=int, default=180)
    ap.add_argument("--all", action="store_true",
                    help="also descend into datasets/ and tmp/ (very slow)")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        models = Path("/root/public/models")
        root = models if models.is_dir() else Path("/root/public")
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 1

    skip = set() if args.all else SKIP_DIRS
    print(f"scanning {root} (depth {args.max_depth}, {args.max_seconds}s budget"
          + (", skipping " + "/".join(sorted(skip)) if skip else "") + ")\n")
    deadline = time.time() + args.max_seconds

    idm, sdxl, ckpts, hinted = [], [], {}, []

    for path, dirnames, filenames in walk(str(root), args.max_depth, deadline, skip):
        if IDM_MARKERS <= dirnames:
            idm.append(path)
        elif SDXL_MARKERS <= dirnames:
            sdxl.append(path)
        for want in CKPT_FILES:
            if want in filenames:
                ckpts.setdefault(want, []).append(os.path.join(path, want))
        base = os.path.basename(path).lower()
        if any(h in base for h in NAME_HINTS):
            hinted.append(path)

    if idm:
        print("IDM-VTON model (has both unet/ and unet_encoder/):")
        for p in idm:
            print(f"  {p}")
        print("\n  Use it without downloading anything:")
        print(f"    export IDM_MODEL_PATH={idm[0]}")
        print("    python scripts/alaya/demo_single.py")
        print(f"    # or: accelerate launch inference.py "
              f"--pretrained_model_name_or_path {idm[0]} ...")
    else:
        print("IDM-VTON model: not found.")
        if sdxl:
            print("  (Found stock SDXL checkouts, but those lack unet_encoder and")
            print("   cannot stand in for the IDM-VTON weights:)")
            for p in sdxl[:5]:
                print(f"    {p}")

    print()
    if ckpts:
        print("Preprocessing checkpoints:")
        for name, paths in sorted(ckpts.items()):
            print(f"  {name}  ->  copy to {CKPT_FILES[name]}")
            for p in paths[:3]:
                print(f"      {p}")
    else:
        print("Preprocessing checkpoints: none found "
              "(01_download_checkpoints.py fetches these; they are small).")

    if hinted and not idm:
        print("\nDirectories whose names look related, worth a manual look:")
        for p in hinted[:25]:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
