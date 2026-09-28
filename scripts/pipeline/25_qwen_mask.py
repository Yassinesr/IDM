#!/usr/bin/env python
"""Stage 2b - fallback masks from Qwen, for images the human parser cannot read.

Runs in the QWEN env, not the IDM-VTON one:
    source scripts/alaya/_activate.sh
    IDM_VENV=$IDM_ROOT/venv-qwen idm_activate
    python scripts/pipeline/25_qwen_mask.py --from-failures work/masks/_unsegmented.txt

MEASURED RESULT: THIS DOES NOT WORK. Kept because the negative result is worth
having written down, not because it is a path to take.

On five 768x1024 studio shots, asked for the leggings, it returned masks that
were 92.5% to 97.8% white - the whole frame. The stage 20 parser gave 7.7% to
8.3% on the same images, which is what a pair of legs actually occupies.

The raw output (--keep-raw) says why: Qwen-Image-Edit did not produce a mask at
all. It produced a LINE DRAWING of the photograph - leggings filled white,
black outlines, face and top still in colour. It read "paint this region white
and everything else black" as a stylization instruction, which is the kind of
thing an image editor is trained to do, rather than as segmentation, which is
not. No threshold recovers a leg region from a sketch, so this is not a tuning
problem.

Stage 20 is also ~6x faster per image (4.47s against 26.30s), so there is no
axis on which this wins. Use scripts/pipeline/20_leg_masks.py.

If you want to try anyway - a different prompt, a different model - the
plumbing is here and --keep-raw shows what the model really returned. A mask
whose coverage is implausible for its target is refused rather than written,
because a 97%-white PNG that a later stage consumes silently is worse than no
file at all.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen import (  # noqa: E402  (must follow the sys.path insert)
    DEFAULT_MODEL, load_pipeline, resolve_model, supported,
)
from _timing import Timer  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_TARGET = "the trousers, leggings, or bare legs"

# Stated as a repaint of THIS image rather than a description of a mask: an
# editing model follows "paint X white" far better than "produce a mask", and
# naming the two colours by hex discourages the grey midtones it would
# otherwise shade with.
PROMPT = (
    "Convert this photograph into a two-colour black and white segmentation "
    "mask. Paint every pixel covered by {target} pure white #FFFFFF. Paint "
    "absolutely everything else pure black #000000 - the background, the "
    "torso, the arms, the hair, the shoes and feet, and any text, arrows or "
    "graphics drawn over the photo. Keep every edge exactly where it is in "
    "the photograph; do not move, resize or redraw the subject. Hard edges, "
    "no grey, no shading, no gradient, no text."
)

NEGATIVE = (
    "grey, gray, gradient, soft edges, blurry edges, anti-aliasing, colour, "
    "photograph, skin texture, fabric texture, text, watermark, shifted "
    "subject, cropped"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-failures", type=Path,
                     help="the _unsegmented.txt stage 20 writes")
    src.add_argument("--in-dir", type=Path, help="directory of images")
    src.add_argument("--image", type=Path, help="a single image")
    ap.add_argument("--out-dir", type=Path, default=Path("work/masks"))
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help=f"what to paint white (default: {DEFAULT_TARGET!r})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=int, default=127,
                    help="grey level above which a pixel becomes white (0-255)")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="refuse a mask under this %% white (default 1)")
    ap.add_argument("--max-coverage", type=float, default=60.0,
                    help="refuse a mask over this %% white (default 60; a leg "
                         "region is about 8)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--long-side", type=int, default=1024,
                    help="generation size for the longer edge; the source "
                         "aspect is preserved so the mask lines up")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also write <name>.qwen.png, the unthresholded output")
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload; slower, for when VRAM is tight")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo masks that already exist")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def collect(args):
    if args.from_failures:
        if not args.from_failures.is_file():
            print(f"ERROR: {args.from_failures} not found. Stage 20 writes it "
                  "only when\n       it actually fails on something.",
                  file=sys.stderr)
            return None
        paths = [Path(line.strip())
                 for line in args.from_failures.read_text().splitlines()
                 if line.strip()]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            print(f"ERROR: listed but not on disk: {missing[0]}", file=sys.stderr)
            return None
        return paths
    if args.in_dir:
        return sorted(p for p in args.in_dir.iterdir()
                      if p.suffix.lower() in IMAGE_SUFFIXES
                      and not p.name.endswith((".mask.png", ".overlay.png",
                                               ".qwen.png")))
    if not args.image.is_file():
        print(f"ERROR: {args.image} not found", file=sys.stderr)
        return None
    return [args.image]


def gen_size(w, h, long_side):
    """Source aspect at the generation scale, rounded to a multiple of 32.

    Generating at a different aspect than the source and resizing back would
    stretch the mask off the subject, which is the one thing a mask must not
    do.
    """
    scale = long_side / max(w, h)
    return (max(32, round(w * scale / 32) * 32),
            max(32, round(h * scale / 32) * 32))


def main():
    args = parse_args()

    images = collect(args)
    if images is None:
        return 1          # collect() already said what was wrong
    if not images:
        print("ERROR: no images to process.", file=sys.stderr)
        return 1

    model_dir = resolve_model(args.model)
    if model_dir is None:
        return 1

    if not args.overwrite:
        pending = [p for p in images
                   if not (args.out_dir / f"{p.stem}.mask.png").exists()]
        skipped = len(images) - len(pending)
        if skipped:
            print(f"note      {skipped} already have a mask "
                  "(--overwrite to redo)")
        images = pending
        if not images:
            print("Nothing to do.")
            return 0

    print(f"model     {model_dir}")
    print(f"target    {args.target!r}")
    print(f"out       {args.out_dir}  ({len(images)} image"
          f"{'' if len(images) == 1 else 's'})")
    print("WARNING   these masks are generated, not measured. Check every "
          "overlay.")

    if args.dry_run:
        print("\nWould write:")
        for p in images:
            print(f"  {p.stem}.mask.png")
        print(f"\nPrompt:\n  {PROMPT.format(target=args.target)}")
        print("\nDry run - nothing loaded.")
        return 0

    import numpy as np
    import torch
    from PIL import Image

    timer = Timer("Qwen-Image-Edit")
    timer.start_load()
    pipe = load_pipeline(model_dir, offload=args.offload)
    if pipe is None:
        return 1
    timer.end_load()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT.format(target=args.target)
    ok = refused = 0

    for i, src in enumerate(images, 1):
        timer.start_image()
        img = Image.open(src).convert("RGB")
        w, h = gen_size(*img.size, args.long_side)

        kwargs = supported(pipe.__call__, {
            "image": img,
            "prompt": prompt,
            "negative_prompt": NEGATIVE,
            "num_inference_steps": args.steps,
            "true_cfg_scale": args.guidance,
            "guidance_scale": args.guidance,
            "width": w,
            "height": h,
            "generator": torch.Generator("cuda").manual_seed(args.seed),
            "num_images_per_prompt": 1,
        })
        print(f"  [{i}/{len(images)}] {src.name}  ({w}x{h})")
        raw = pipe(**kwargs).images[0]

        if args.keep_raw:
            raw.save(args.out_dir / f"{src.stem}.qwen.png")

        # Back to the source size first, so the threshold operates on the
        # pixels the mask will actually be used against.
        grey = raw.convert("L").resize(img.size, Image.BILINEAR)
        arr = np.array(grey) > args.threshold
        coverage = 100.0 * arr.mean()

        mask = Image.fromarray((arr * 255).astype(np.uint8))
        mask.save(args.out_dir / f"{src.stem}.mask.png")

        tint = Image.new("RGB", img.size, (255, 0, 0))
        Image.composite(Image.blend(img, tint, 0.45), img, mask) \
             .save(args.out_dir / f"{src.stem}.overlay.png")

        dt = timer.end_image()

        # A mask this far from plausible is not a mask. Writing it anyway means
        # a later stage consumes it without complaint - the 97%-white output
        # this model actually produces would repaint the entire image.
        if not args.min_coverage <= coverage <= args.max_coverage:
            print(f"      {coverage:.1f}% white, {dt:.2f}s  REFUSED - outside "
                  f"{args.min_coverage:g}-{args.max_coverage:g}%")
            if args.keep_raw:
                print(f"      (raw output kept as {src.stem}.qwen.png)")
            for stale in (args.out_dir / f"{src.stem}.mask.png",
                          args.out_dir / f"{src.stem}.overlay.png"):
                stale.unlink(missing_ok=True)
            refused += 1
            continue

        print(f"      {coverage:.1f}% white, {dt:.2f}s")
        ok += 1

    timer.report()

    print(f"\n{ok} mask(s) in {args.out_dir.resolve()}"
          + (f", {refused} refused" if refused else ""))
    if refused:
        print("\nRefused masks had implausible coverage. On the images this was "
              "measured\non, that is every one of them - see the note at the top "
              "of this file.\nUse scripts/pipeline/20_leg_masks.py instead.")
    if ok:
        print("Check the overlays before using these. A generated mask that is "
              "the right\nshape in the wrong place looks fine as a mask and is "
              "useless as one.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
