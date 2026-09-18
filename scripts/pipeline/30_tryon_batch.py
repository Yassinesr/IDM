#!/usr/bin/env python
"""Stage 3 - put one garment on every generated view.

Runs in the IDM-VTON env:
    source scripts/alaya/env.sh
    python scripts/pipeline/30_tryon_batch.py \\
        --in-dir work/variants --garment gradio_demo/example/cloth/yoga_pants.jpg \\
        --desc "plain mauve high-waisted leggings"

Loads the model once for the whole directory, which is the point - a per-image
demo_single.py call would reload it every time.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tryon import TryOn, REPO  # noqa: E402  (must follow the sys.path insert)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True, help="the generated views")
    ap.add_argument("--garment", type=Path, required=True)
    ap.add_argument("--desc", required=True, help="garment description; it goes in the prompt")
    ap.add_argument("--out-dir", type=Path, default=Path("work/results"))
    ap.add_argument("--category", default="lower_body",
                    choices=["upper_body", "lower_body", "dresses"])
    ap.add_argument("--model", default="yisol/IDM-VTON")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--save-mask", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N images")
    args = ap.parse_args()

    for p in (args.in_dir, args.garment):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 1

    images = sorted(p for p in args.in_dir.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                    and not p.name.endswith((".mask.png", ".overlay.png")))
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"ERROR: no input images in {args.in_dir}", file=sys.stderr)
        return 1

    from PIL import Image
    garment = Image.open(args.garment)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"garment   {args.garment}")
    print(f"desc      {args.desc!r}")
    print(f"category  {args.category}")
    print(f"images    {len(images)}")
    print("loading the pipeline once for the whole batch")
    t0 = time.time()
    engine = TryOn(model_path=args.model)
    print(f"loaded in {time.time() - t0:.0f}s\n")

    ok = failed = 0
    for i, src in enumerate(images, 1):
        dest = args.out_dir / f"{src.stem}.tryon.png"
        if dest.exists():
            print(f"  [{i}/{len(images)}] {dest.name} exists, skipping")
            continue
        t = time.time()
        try:
            result, mask = engine.run(
                Image.open(src), garment, args.desc, category=args.category,
                steps=args.steps, seed=args.seed, guidance=args.guidance_scale)
        except IndexError:
            # Same cause as in demo_single.py: the pose estimator found nobody.
            # One unusable generated view must not kill the batch.
            print(f"  [{i}/{len(images)}] {src.name}: SKIP - no person detected")
            failed += 1
            continue
        result.save(dest)
        if args.save_mask:
            mask.save(args.out_dir / f"{src.stem}.mask.png")
        print(f"  [{i}/{len(images)}] {src.name} -> {dest.name}  ({time.time()-t:.0f}s)")
        ok += 1

    print(f"\n{ok} result(s) in {args.out_dir.resolve()}" +
          (f", {failed} skipped" if failed else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
