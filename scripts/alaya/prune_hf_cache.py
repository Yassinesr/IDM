#!/usr/bin/env python
"""Reclaim disk by deleting duplicate weight formats from the HF cache.

A diffusers repo usually ships each tensor file twice - once as .safetensors
and once as .bin - and from_pretrained prefers safetensors. The .bin copies are
therefore dead weight once downloaded.

    python scripts/alaya/prune_hf_cache.py            # report only
    python scripts/alaya/prune_hf_cache.py --force    # actually delete

A .bin is removed ONLY when a .safetensors with the same stem sits beside it in
the same snapshot directory, so a subfolder shipping .bin alone is untouched.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

DUPLICABLE = {".bin", ".pth", ".ckpt", ".msgpack", ".h5"}


def hub_dir():
    hub = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return Path(hub)
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(home) / "hub"


def find_duplicates(hub):
    """(symlink, blob, size) for weight files that have a safetensors twin."""
    out = []
    for snap in hub.glob("*/snapshots/*"):
        if not snap.is_dir():
            continue
        for dirpath, _, filenames in os.walk(snap):
            names = set(filenames)
            stems_with_st = {Path(n).stem for n in names if n.endswith(".safetensors")}
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix not in DUPLICABLE:
                    continue
                if p.stem not in stems_with_st:
                    continue  # no twin - this is the only copy, keep it
                blob = Path(os.path.realpath(p))
                if not blob.is_file():
                    continue
                out.append((p, blob, blob.stat().st_size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete (default: report only)")
    args = ap.parse_args()

    hub = hub_dir()
    if not hub.is_dir():
        print(f"ERROR: no HF cache at {hub}", file=sys.stderr)
        return 1
    print(f"cache {hub}\n")

    # Interrupted downloads leave .incomplete files, and a blob whose snapshot
    # symlink was removed is unreachable - neither is ever read again, and both
    # are common after a download that had to be retried.
    incomplete, orphans = [], []
    referenced = set()
    for link in hub.glob("*/snapshots/*/**/*"):
        if link.is_symlink():
            referenced.add(os.path.realpath(link))
    for blobdir in hub.glob("*/blobs"):
        for b in blobdir.iterdir():
            if not b.is_file():
                continue
            if b.name.endswith(".incomplete"):
                incomplete.append((b, b.stat().st_size))
            elif str(b.resolve()) not in referenced:
                orphans.append((b, b.stat().st_size))

    for label, items in (("partial downloads (.incomplete)", incomplete),
                         ("orphaned blobs (nothing links to them)", orphans)):
        if items:
            tot = sum(sz for _, sz in items)
            print(f"  {tot / 2**30:7.2f} GB  {label} - {len(items)} file(s)")

    dupes = find_duplicates(hub)
    if not dupes and not incomplete and not orphans:
        print("Nothing to prune.")
        return 0

    # A blob can be referenced from several revisions; only drop it when every
    # reference to it is itself a duplicate we are removing.
    refs = defaultdict(int)
    for link in hub.glob("*/snapshots/*/**/*"):
        if link.is_symlink():
            refs[os.path.realpath(link)] += 1

    total = 0
    by_repo = defaultdict(int)
    for link, blob, size in dupes:
        if refs[str(blob)] > 1:
            print(f"  skip {link.name}: blob shared by {refs[str(blob)]} revisions")
            continue
        repo = str(link.relative_to(hub)).split("/")[0]
        by_repo[repo] += size
        total += size

    for repo, size in sorted(by_repo.items(), key=lambda kv: -kv[1]):
        print(f"  {size / 2**30:7.2f} GB  {repo}")
    print(f"  {total / 2**30:7.2f} GB  RECLAIMABLE")

    if not args.force:
        print("\nReport only. Add --force to delete.")
        return 0

    freed = 0
    for b, size in incomplete + orphans:
        try:
            b.unlink()
            freed += size
        except OSError as exc:
            print(f"  could not remove {b.name}: {exc}", file=sys.stderr)

    for link, blob, size in dupes:
        if refs[str(blob)] > 1:
            continue
        try:
            link.unlink()
            blob.unlink()
            freed += size
        except OSError as exc:
            print(f"  could not remove {link.name}: {exc}", file=sys.stderr)
    print(f"\nFreed {freed / 2**30:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
