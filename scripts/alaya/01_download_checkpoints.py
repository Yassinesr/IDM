#!/usr/bin/env python
"""Fetch every checkpoint IDM-VTON needs, mirror-aware.

The ckpt/* files committed to this repo are placeholders ("put X here"), and
the main model is pulled from the Hub at run time. On Alaya NeW, huggingface.co
is not reachable, so this goes through $HF_ENDPOINT (hf-mirror.com by default,
set in scripts/alaya/env.sh).

    source scripts/alaya/env.sh
    python scripts/alaya/01_download_checkpoints.py --dry-run  # measure first
    python scripts/alaya/01_download_checkpoints.py            # inference
    python scripts/alaya/01_download_checkpoints.py --slim     # fp16 weights only
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


def repo_sizes(repo_id, repo_type="model"):
    """{filename: bytes} for a Hub repo, without downloading anything."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    return {sib.rfilename: (sib.size or 0) for sib in info.siblings}


def slim_ignore_patterns(sizes):
    """Drop the duplicate weight formats a diffusers repo usually ships.

    Every subfolder that has a .safetensors does not also need the .bin - they
    are the same tensors. Also skip the framework ports we never load. Only the
    .bin files with a safetensors sibling are dropped, so a subfolder that has
    *only* .bin still downloads.
    """
    ignore = ["*.msgpack", "*.h5", "*.onnx"]
    safetensors_dirs = {
        os.path.dirname(f) for f in sizes if f.endswith(".safetensors")
    }
    for f in sizes:
        if f.endswith(".bin") and os.path.dirname(f) in safetensors_dirs:
            ignore.append(f)
    return ignore


def cached_bytes(repo_id, repo_type="model"):
    """Bytes already present in the local cache for this repo.

    Only real files count: everything under snapshots/ is a symlink into
    blobs/, so counting both would double every file.
    """
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hub:
        home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        hub = os.path.join(home, "hub")
    folder = Path(hub) / f"{repo_type}s--{repo_id.replace('/', '--')}"
    if not folder.is_dir():
        return 0
    total = 0
    for f in folder.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def report_sizes(sizes, ignore=None):
    """Print a per-directory size table; returns the total that would download."""
    from fnmatch import fnmatch
    ignore = ignore or []
    kept, skipped = {}, 0
    for f, n in sizes.items():
        if any(fnmatch(f, pat) for pat in ignore):
            skipped += n
            continue
        kept[os.path.dirname(f) or "."] = kept.get(os.path.dirname(f) or ".", 0) + n
    for d, n in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"    {n / 2**30:8.2f} GB  {d}/")
    total = sum(kept.values())
    print(f"    {total / 2**30:8.2f} GB  TOTAL")
    if skipped:
        print(f"    {skipped / 2**30:8.2f} GB  skipped by --slim")
    return total


def place(cached: str, rel_dest: str) -> None:
    dest = REPO / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    # hf_hub_download returns a path under snapshots/, which is a SYMLINK into
    # blobs/ with a *relative* target. Hard-linking that symlink copies the link
    # rather than the file, and its relative target does not resolve from ckpt/,
    # so the result is a dangling symlink that only fails later. Resolve to the
    # real blob first.
    src = os.path.realpath(cached)
    if not os.path.isfile(src):
        raise FileNotFoundError(f"{cached} does not resolve to a file (-> {src})")

    # lexists, not exists: a dangling symlink from an earlier run is invisible to
    # exists(), which would leave it in place and make os.link fail with EEXIST.
    if os.path.lexists(dest):
        if not dest.is_symlink() and dest.is_file() \
                and dest.stat().st_size > PLACEHOLDER_MAX:
            print(f"    kept {rel_dest} ({dest.stat().st_size / 2**20:.1f} MiB)")
            return
        os.unlink(dest)  # placeholder, or a dangling link from a failed run

    try:
        os.link(src, dest)  # same filesystem: free
    except OSError:
        shutil.copy2(src, dest)

    if not dest.is_file():
        raise RuntimeError(f"could not place {rel_dest} from {src}")
    print(f"    -> {rel_dest} ({dest.stat().st_size / 2**20:.1f} MiB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training", action="store_true",
                    help="also fetch the IP-Adapter weights needed by train_xl.py")
    ap.add_argument("--skip-model", action="store_true",
                    help="only fetch the small preprocessing checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would download and how big, then exit")
    ap.add_argument("--force", action="store_true",
                    help="download even if free disk looks insufficient")
    ap.add_argument("--slim", action="store_true",
                    help="skip .bin weights that have a .safetensors twin; "
                         "roughly halves the model download on a small disk")
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

    ignore = None
    sizes = None
    if not args.skip_model and (args.dry_run or args.slim):
        print(f"\n[sizes] {MODEL_REPO}")
        try:
            sizes = repo_sizes(MODEL_REPO)
        except Exception as exc:  # network, mirror gaps, auth
            print(f"      could not read repo metadata: {exc}", file=sys.stderr)
            if args.dry_run:
                return 1
            sizes = None
        if sizes:
            ignore = slim_ignore_patterns(sizes) if args.slim else None
            report_sizes(sizes, ignore)

    if args.dry_run:
        print("\n[sizes] preprocessing checkpoints (always downloaded)")
        try:
            space = repo_sizes(SPACE_REPO, repo_type="space")
            total = sum(space.get(f, 0) for f in SPACE_FILES)
            print(f"    {total / 2**30:8.2f} GB  ckpt/ (4 files)")
        except Exception as exc:
            print(f"      could not read space metadata: {exc}", file=sys.stderr)
        print("\nDry run only - nothing downloaded.")
        return 0

    if not args.skip_model and not args.force:
        # Filling the container disk does not just fail the download - it can
        # wedge the whole Workshop, so check before starting rather than after.
        need = None
        try:
            sizes_for_check = sizes or repo_sizes(MODEL_REPO)
            from fnmatch import fnmatch
            pats = ignore or []
            need = sum(n for f, n in sizes_for_check.items()
                       if not any(fnmatch(f, p) for p in pats))
        except Exception:
            pass  # metadata unavailable; fall through and let it run

        if need:
            # Whatever is already cached will not be fetched again, so it must
            # not count towards what we need free - otherwise a resumed run is
            # refused precisely because the previous one succeeded.
            have = cached_bytes(MODEL_REPO)
            remaining = max(0, need - have)

            target = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
            probe = target
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            free = shutil.disk_usage(probe).free
            headroom = remaining * 1.15  # .incomplete files need slack
            print(f"\n[disk] repo {need / 2**30:.1f} GB, already cached "
                  f"{have / 2**30:.1f} GB, still to fetch "
                  f"{remaining / 2**30:.1f} GB; free {free / 2**30:.1f} GB on {probe}")
            need = remaining
            if free < headroom:
                print(f"\nERROR: not enough free space for the model.", file=sys.stderr)
                print(f"       need ~{headroom / 2**30:.1f} GB including slack, "
                      f"have {free / 2**30:.1f} GB.", file=sys.stderr)
                if not args.slim:
                    print("       Try --slim first; it skips duplicate weight "
                          "formats.", file=sys.stderr)
                print("       Otherwise attach a PVC, point a local copy at "
                      "IDM_MODEL_PATH\n"
                      "       (scripts/alaya/find_local_models.py), or "
                      "override with --force.", file=sys.stderr)
                return 1

    if not args.skip_model:
        print(f"\n[1/2] {MODEL_REPO} (model) -> HF cache")
        print("      Resumable: re-run this script if it drops.")
        # Resuming is the default in huggingface_hub 0.23+; asking for it
        # explicitly only earns a deprecation warning.
        snapshot_download(repo_id=MODEL_REPO, max_workers=4, ignore_patterns=ignore)
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
