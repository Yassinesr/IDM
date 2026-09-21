#!/usr/bin/env python
"""Stage 2 - binary leg-area masks, feet excluded. A deliverable in its own right.

Runs in the IDM-VTON env (it reuses that human parser), NOT the Qwen one:
    source scripts/alaya/env.sh
    python scripts/pipeline/20_leg_masks.py --in-dir work/variants

White = leg area, black = everything else, one PNG per input image.

Two regions are available:
  leg_area  (default) the limbs and whatever covers them - parse labels
            skirt(5), pants(6), left_leg(12), right_leg(13). A clean
            segmentation, which is what the deliverable asks for.
  inpaint   exactly the region stage 3 will repaint, via get_mask_location.
            Dilated and hole-filled, so it is looser than the true silhouette -
            useful for checking stage 3 rather than as a segmentation.

Feet are excluded by cutting at the ankle, taken from OpenPose keypoints 10
(right ankle) and 13 (left ankle). ATR has no foot label, so bare feet would
otherwise fall under left_leg/right_leg and be included.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gradio_demo"))

PARSE_WIDTH, PARSE_HEIGHT = 384, 512
R_ANKLE, L_ANKLE = 10, 13
LEG_LABELS = {5: "skirt", 6: "pants", 12: "left_leg", 13: "right_leg"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-dir", type=Path, help="directory of images")
    src.add_argument("--image", type=Path, help="a single image")
    ap.add_argument("--out-dir", type=Path, default=Path("work/masks"))
    ap.add_argument("--region", choices=["leg_area", "inpaint"], default="leg_area")
    ap.add_argument("--keep-feet", action="store_true",
                    help="do not cut at the ankle")
    ap.add_argument("--ankle-offset", type=int, default=0,
                    help="shift the ankle cut down (+) or up (-), in 384x512 pixels")
    ap.add_argument("--full-size", action="store_true",
                    help="write at the source image's resolution instead of 768x1024")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="percent white below which a mask counts as a failure "
                         "and is listed for the Qwen fallback (default 0.5)")
    ap.add_argument("--overlay", action="store_true",
                    help="also write a <name>.overlay.png for eyeballing the fit")
    return ap.parse_args()


def ankle_cut_y(keypoints, offset):
    """Y below which everything is foot. None when no ankle was detected."""
    pts = keypoints["pose_keypoints_2d"]
    ys = []
    for idx in (R_ANKLE, L_ANKLE):
        if idx >= len(pts):
            continue
        x, y = pts[idx][0], pts[idx][1]
        # run_openpose.py substitutes [0, 0] for keypoints it could not find.
        if x <= 1.0 and y <= 1.0:
            continue
        ys.append(y)
    if not ys:
        return None
    # The higher ankle (smaller y) is the safe cut: it excludes both feet even
    # when one is forward. --ankle-offset tunes it.
    return int(min(ys)) + offset


def main():
    args = parse_args()

    if args.in_dir:
        images = sorted(p for p in args.in_dir.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"ERROR: no images in {args.in_dir}", file=sys.stderr)
            return 1
    else:
        if not args.image.is_file():
            print(f"ERROR: {args.image} not found", file=sys.stderr)
            return 1
        images = [args.image]

    import numpy as np
    from PIL import Image
    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device (the parser calls torch.cuda.set_device).",
              file=sys.stderr)
        return 1

    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from utils_mask import get_mask_location

    print(f"region    {args.region}")
    print(f"feet      {'kept' if args.keep_feet else 'excluded at the ankle'}")
    print(f"images    {len(images)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to("cuda:0")

    ok = failed = 0
    unsegmented = []
    for i, src in enumerate(images, 1):
        img = Image.open(src).convert("RGB")
        out_size = img.size if args.full_size else (768, 1024)
        small = img.resize((PARSE_WIDTH, PARSE_HEIGHT))

        # OpenPose is needed ONLY for the ankle cut. The parser that actually
        # produces the mask is independent of it, so a photo it cannot read -
        # a crop with no head or shoulders - is not a reason to give up on the
        # segmentation. It is a reason to skip the cut, which a crop with no
        # feet in it did not need anyway.
        try:
            keypoints = openpose_model(small)
        except IndexError:
            keypoints = None
            print(f"  [{i}/{len(images)}] {src.name}: no person found by OpenPose "
                  "- parsing anyway, no ankle cut")

        model_parse, _ = parsing_model(small)

        if args.region == "inpaint":
            if keypoints is None:
                print(f"  [{i}/{len(images)}] {src.name}: SKIP - --region inpaint "
                      "needs keypoints")
                failed += 1
                unsegmented.append(src)
                continue
            mask_img, _ = get_mask_location("hd", "lower_body", model_parse, keypoints)
            arr = (np.array(mask_img.resize((PARSE_WIDTH, PARSE_HEIGHT),
                                            Image.NEAREST)) > 127)
        else:
            parse = np.array(model_parse.resize((PARSE_WIDTH, PARSE_HEIGHT),
                                                Image.NEAREST))
            arr = np.isin(parse, list(LEG_LABELS))

        cut = None
        if not args.keep_feet and keypoints is not None:
            cut = ankle_cut_y(keypoints, args.ankle_offset)
            if cut is None:
                print(f"  [{i}/{len(images)}] {src.name}: no ankle keypoint, feet kept")
            else:
                arr[max(0, min(cut, PARSE_HEIGHT)):, :] = False

        coverage = 100.0 * arr.mean()
        if coverage < args.min_coverage:
            # The parser ran but found essentially no leg. ATR is trained on
            # whole people, so a tight crop can come back empty rather than
            # wrong. Nothing useful to write.
            print(f"  [{i}/{len(images)}] {src.name}: FAILED - only "
                  f"{coverage:.2f}% white, below --min-coverage")
            failed += 1
            unsegmented.append(src)
            continue

        mask = Image.fromarray((arr * 255).astype(np.uint8)).resize(out_size, Image.NEAREST)
        dest = args.out_dir / f"{src.stem}.mask.png"
        mask.save(dest)

        note = f"ankle cut y={cut}" if cut is not None else "no cut"
        print(f"  [{i}/{len(images)}] {src.name} -> {dest.name}  "
              f"{coverage:.1f}% white, {note}")

        if args.overlay:
            base = img.resize(out_size)
            tint = Image.new("RGB", out_size, (255, 0, 0))
            blended = Image.composite(
                Image.blend(base, tint, 0.45), base, mask.convert("L"))
            blended.save(args.out_dir / f"{src.stem}.overlay.png")
        ok += 1

    print(f"\n{ok} mask(s) in {args.out_dir.resolve()}" +
          (f", {failed} failed" if failed else ""))

    listing = args.out_dir / "_unsegmented.txt"
    if unsegmented:
        listing.write_text("".join(f"{p}\n" for p in unsegmented))
        print(f"\n{len(unsegmented)} image(s) the parser could not segment, "
              f"listed in\n  {listing}")
        print("\nThese are usually crops the parser was never trained on - no "
              "head,\nno shoulders, or a frame too tight to read as a person. "
              "To fall back\nto Qwen, in the Qwen env:")
        print(f"  python scripts/pipeline/25_qwen_mask.py --from-failures {listing}")
    elif listing.exists():
        listing.unlink()   # stale list from an earlier run

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
