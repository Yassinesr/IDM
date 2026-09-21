#!/usr/bin/env python
"""Stage 3b - text removal and garment swap via Qwen, for images IDM-VTON cannot take.

Runs in the QWEN env, not the IDM-VTON one:
    source scripts/alaya/_activate.sh
    IDM_VENV=$IDM_ROOT/venv-qwen idm_activate
    python scripts/pipeline/35_qwen_tryon.py \
        --human gradio_demo/example/human/only_lower.jpg \
        --garment gradio_demo/example/cloth/yoga_pants.jpg \
        --desc "plain mauve high-waisted leggings"

WHY THIS EXISTS: IDM-VTON runs OpenPose first and needs a head, neck and
shoulders to find a person at all. A half-body product crop has none, so stage
30 skips it - no amount of tuning changes that. Qwen has no such requirement.

WHAT YOU GIVE UP: IDM-VTON warps the actual pixels of your garment photo onto
the body, so the print, seams and cut survive. Qwen re-imagines the garment
from the reference; the result looks right but the fine detail drifts. For a
plain legging that is usually fine. For a patterned or branded garment it is
not - for those, outpaint the crop to a full body first (stage 10) and use
stage 30, which keeps the garment faithful.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen import (  # noqa: E402  (must follow the sys.path insert)
    DEFAULT_MODEL, load_pipeline, resolve_model, supported,
)

# Text removal is stated first and concretely. A marketing crop carries text,
# measurement arrows and dashed guide lines drawn over the photo; an editing
# model reproduces all of that unless told plainly to reconstruct what is
# underneath.
CLEANUP = (
    "Remove every piece of text, every number, every arrow and every dashed "
    "or dotted guide line drawn over this photograph, reconstructing the "
    "garment and background cleanly underneath them. "
)

# Multi-image: the garment comes from the second image.
SWAP_FROM_IMAGE = (
    "Replace the trousers the person is wearing with the garment shown in the "
    "second image, matching its colour, fabric and length, fitted naturally to "
    "the body with correct folds and shadows. "
)

# Single-image: the garment is described in words instead.
SWAP_FROM_TEXT = (
    "Replace the trousers the person is wearing with {desc}, fitted naturally "
    "to the body with correct folds and shadows. "
)

KEEP = (
    "Keep the same person, the same pose, the same body shape, the same "
    "camera angle and the same crop. Do not add a head or any body part that "
    "is outside the frame. Clean product photograph on a plain background, "
    "soft even lighting, sharp focus. The final image must contain no text of "
    "any kind."
)

NEGATIVE = (
    "text, chinese characters, numbers, arrows, dashed lines, watermark, "
    "logo, caption, diagram, annotation, extra limbs, distorted anatomy, "
    "changed pose, added head, different person, blurry"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, required=True,
                    help="the photo to edit, e.g. the half-body crop")
    ap.add_argument("--garment", type=Path,
                    help="garment photo; without it, --desc alone describes it")
    ap.add_argument("--desc", default="",
                    help="garment in words; used alone when --garment is "
                         "absent, and as extra guidance when it is not")
    ap.add_argument("--out", type=Path,
                    help="default: work/results/<human>.qwen_tryon.png")
    ap.add_argument("--clean-only", action="store_true",
                    help="only remove the text; change no clothing")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--count", type=int, default=1,
                    help="how many results, each with its own seed")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--long-side", type=int, default=1024,
                    help="generation size for the longer edge; the source "
                         "aspect is preserved so the crop is not stretched")
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload; slower, for when VRAM is tight")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def build_prompt(args, multi_image):
    if args.clean_only:
        return CLEANUP + KEEP
    if multi_image:
        swap = SWAP_FROM_IMAGE
        if args.desc:
            swap += f"The garment is {args.desc}. "
        return CLEANUP + swap + KEEP
    return CLEANUP + SWAP_FROM_TEXT.format(desc=args.desc) + KEEP


def gen_size(w, h, long_side):
    """Source aspect at the generation scale, rounded to a multiple of 32."""
    scale = long_side / max(w, h)
    return (max(32, round(w * scale / 32) * 32),
            max(32, round(h * scale / 32) * 32))


def main():
    args = parse_args()

    if not args.human.is_file():
        print(f"ERROR: {args.human} not found", file=sys.stderr)
        return 1
    if args.garment and not args.garment.is_file():
        print(f"ERROR: {args.garment} not found", file=sys.stderr)
        return 1
    if not args.clean_only and not args.garment and not args.desc:
        print("ERROR: give --garment, or --desc, or --clean-only.\n"
              "       With neither there is nothing to put on.", file=sys.stderr)
        return 1

    model_dir = resolve_model(args.model)
    if model_dir is None:
        return 1

    out = args.out or Path("work/results") / f"{args.human.stem}.qwen_tryon.png"
    multi = bool(args.garment)
    prompt = build_prompt(args, multi)

    print(f"model     {model_dir}")
    print(f"human     {args.human}")
    print(f"garment   {args.garment if multi else '(from --desc)'}")
    print(f"out       {out}" + (f"  ({args.count} seeds)" if args.count > 1 else ""))
    if not args.clean_only:
        print("NOTE      Qwen re-imagines the garment rather than warping your\n"
              "          photo of it, so fine detail will drift. Plain "
              "garments\n          survive this; patterned or branded ones do "
              "not.")

    if args.dry_run:
        print(f"\nPrompt:\n  {prompt}")
        print("\nDry run - nothing loaded.")
        return 0

    import torch
    from PIL import Image

    human = Image.open(args.human).convert("RGB")
    w, h = gen_size(*human.size, args.long_side)
    images = [human]
    if multi:
        images.append(Image.open(args.garment).convert("RGB"))

    pipe = load_pipeline(model_dir, offload=args.offload)
    if pipe is None:
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    for i in range(args.count):
        seed = args.seed + i
        dest = out if args.count == 1 else \
            out.with_name(f"{out.stem}_seed{seed}{out.suffix}")
        if dest.exists():
            print(f"  [{i+1}/{args.count}] {dest.name} exists, skipping")
            continue

        def call(image_arg, prompt_text):
            kwargs = supported(pipe.__call__, {
                "image": image_arg,
                "prompt": prompt_text,
                "negative_prompt": NEGATIVE,
                "num_inference_steps": args.steps,
                "true_cfg_scale": args.guidance,
                "guidance_scale": args.guidance,
                "width": w,
                "height": h,
                "generator": torch.Generator("cuda").manual_seed(seed),
                "num_images_per_prompt": 1,
            })
            return pipe(**kwargs)

        print(f"  [{i+1}/{args.count}] seed {seed} -> {dest.name}")
        try:
            result = call(images if multi else human, prompt)
        except (TypeError, ValueError) as exc:
            # Only the "Plus" pipelines take a list of images. An older
            # single-image one raises rather than ignoring the extra, so fall
            # back to describing the garment instead of showing it.
            if not multi:
                raise
            print(f"      multi-image input refused ({type(exc).__name__}: "
                  f"{str(exc)[:80]})")
            if not args.desc:
                print("\nERROR: this pipeline takes one image, and without "
                      "--desc there is\n       no other way to say what the "
                      "garment is. Re-run with\n"
                      '       --desc "a description of the garment".',
                      file=sys.stderr)
                return 1
            print("      falling back to --desc, garment photo unused")
            multi = False
            result = call(human, build_prompt(args, multi_image=False))

        # Back to the source size: the point of this stage is an edit of that
        # crop, not a differently-sized picture of it.
        result.images[0].resize(human.size, Image.LANCZOS).save(dest)
        written += 1

    print(f"\nWrote {written} image(s) to {out.parent.resolve()}")
    print("Check the text is actually gone - generative removal is not "
          "guaranteed,\nand Chinese characters over fabric are a hard case.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
