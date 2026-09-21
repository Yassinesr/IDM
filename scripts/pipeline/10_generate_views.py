#!/usr/bin/env python
"""Stage 1 - generate extra views of a reference model photo with Qwen-Image-Edit.

Runs in the QWEN env, not the IDM-VTON one:
    source $IDM_ROOT/venv-qwen/bin/activate
    python scripts/pipeline/10_generate_views.py --reference <photo> --count 8

Weights are read straight from the read-only shared mount, so nothing is
downloaded and nothing is written outside --out-dir.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen import (  # noqa: E402  (must follow the sys.path insert)
    DEFAULT_MODEL, load_pipeline, resolve_model, supported,
)

# This is an editing model, so the prompt is an instruction about the reference,
# not just a description of the output. Removing source text is stated first and
# concretely: a reference carrying a brand mark is otherwise imitated, and a
# negative prompt alone is a weak lever against something visible in the input.
TEXT_REMOVAL = (
    "Remove all text, logos, watermarks, brand names and corner badges, "
    "reconstructing the background cleanly where they were. "
)

# True of every variant. Deliberately says nothing about which way the model
# faces - that is the pose's job, and a style block repeating "facing the
# camera" would fight every pose that is not a front view.
STYLE = (
    " Keep the same woman, the same face, the same hairstyle and exactly the "
    "same clothing. Taobao e-commerce product photograph: full-body shot, the "
    "top of the head and both feet fully inside the frame, plain seamless "
    "light grey studio background, soft even lighting, sharp focus, high "
    "resolution. The final image must contain no text of any kind."
)

# One entry per variant, because an editing model reproduces its input unless
# told otherwise - changing the seed alone moves almost nothing. Each
# instruction names the new camera angle or body position outright.
#
# Every pose keeps the legs visible and the feet in frame: stage 20 masks the
# leg region from the human parse and cuts at the ankles found by OpenPose, and
# neither works on a crop that loses them.
POSES = [
    ("front",
     "Change the pose: the model stands upright facing the camera directly, "
     "arms relaxed at her sides, feet together."),
    ("front_hands_hips",
     "Change the pose: the model stands facing the camera with both hands "
     "resting on her hips, elbows out, feet shoulder-width apart."),
    ("three_quarter_left",
     "Change the camera angle: the model's body is turned 45 degrees to her "
     "left, a three-quarter view, her head turned back toward the camera, "
     "arms relaxed."),
    ("three_quarter_right",
     "Change the camera angle: the model's body is turned 45 degrees to her "
     "right, a three-quarter view, her head turned back toward the camera, "
     "arms relaxed."),
    ("profile_left",
     "Change the camera angle: a full side profile from the model's left, her "
     "body turned 90 degrees so the camera sees her side, arms at her sides."),
    ("profile_right",
     "Change the camera angle: a full side profile from the model's right, her "
     "body turned 90 degrees so the camera sees her side, arms at her sides."),
    ("back",
     "Change the camera angle: the model has her back to the camera, seen from "
     "directly behind, standing upright with arms relaxed at her sides."),
    ("back_over_shoulder",
     "Change the camera angle: the model has her back to the camera and turns "
     "her head to look back over her shoulder, arms relaxed."),
    ("walking",
     "Change the pose: the model is walking toward the camera, caught "
     "mid-stride with one leg forward and the other back, arms swinging "
     "naturally."),
    ("contrapposto",
     "Change the pose: the model stands facing the camera with her weight on "
     "one leg, the other knee slightly bent and turned outward, one hand on "
     "her hip."),
    ("legs_apart",
     "Change the pose: the model stands facing the camera with her feet set "
     "well apart, legs straight and clearly separated, arms relaxed at her "
     "sides."),
    ("arms_crossed",
     "Change the pose: the model stands facing the camera with her arms "
     "crossed over her chest, feet shoulder-width apart."),
]

POSE_BY_SLUG = {slug: text for slug, text in POSES}

# IDM-VTON is trained on front-facing shots; its warping module has no real
# supervision for a garment seen from behind or edge-on. These still generate
# fine and are worth having, but expect stage 30 to degrade on them.
OFF_FRONT = {"profile_left", "profile_right", "back", "back_over_shoulder"}


def build_prompt(pose_instruction: str) -> str:
    """Edit instruction first, constraints after - that is the order this kind
    of model weights most heavily."""
    return TEXT_REMOVAL + pose_instruction + STYLE


def select_poses(spec: str, count: int):
    """Resolve --poses into a list of (slug, instruction)."""
    if spec in (None, "", "auto"):
        return POSES[:count]
    if spec == "all":
        return POSES
    chosen = []
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        if name not in POSE_BY_SLUG:
            raise SystemExit(
                f"ERROR: unknown pose {name!r}. Run --list-poses to see them all.")
        chosen.append((name, POSE_BY_SLUG[name]))
    return chosen

DEFAULT_NEGATIVE = (
    "text, watermark, logo, brand name, letters, caption, signature, "
    "cropped head, cropped feet, close-up, collage, multiple people, "
    "blurry, distorted anatomy, extra limbs"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    help="the seed photo, e.g. gradio_demo/example/human/model_front.jpg")
    ap.add_argument("--out-dir", type=Path, default=Path("work/variants"))
    ap.add_argument("--count", type=int, default=8,
                    help="how many poses to take from the table, in order")
    ap.add_argument("--poses", default="auto",
                    help="comma-separated pose names, or 'all'; default is the "
                         "first --count entries")
    ap.add_argument("--list-poses", action="store_true",
                    help="print the pose table and exit")
    ap.add_argument("--repeat", type=int, default=1,
                    help="images per pose, each with its own seed")
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL))
    ap.add_argument("--prompt", default=None,
                    help="one prompt for every image, replacing the pose table "
                         "(you then get --count images that differ only by seed)")
    ap.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234, help="first seed; each variant increments it")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--offload", action="store_true",
                    help="sequential CPU offload; slower, for when VRAM is tight")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would run, without loading the model")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.list_poses:
        print(f"{len(POSES)} poses (--poses front,walking,... or --poses all):\n")
        for slug, text in POSES:
            flag = "  [off-front: stage 30 degrades]" if slug in OFF_FRONT else ""
            print(f"  {slug}{flag}")
            print(f"      {text}\n")
        return 0

    if args.reference is None:
        print("ERROR: --reference is required", file=sys.stderr)
        return 1
    if not args.reference.is_file():
        print(f"ERROR: reference not found: {args.reference}", file=sys.stderr)
        return 1

    model_dir = resolve_model(args.model)
    if model_dir is None:
        return 1

    # Build the whole job list up front so --dry-run can show exactly what a
    # real run would write, without loading 54 GB to find out.
    jobs = []   # (slug, prompt, seed, dest)
    if args.prompt:
        for i in range(args.count):
            seed = args.seed + i
            jobs.append(("custom", args.prompt, seed,
                         args.out_dir / f"{i:02d}_custom_seed{seed}.png"))
    else:
        for pi, (slug, instruction) in enumerate(select_poses(args.poses, args.count)):
            for r in range(args.repeat):
                # pose index * 100 keeps each pose's seeds in its own block, so
                # adding a pose never renumbers another one's output.
                seed = args.seed + pi * 100 + r
                jobs.append((slug, build_prompt(instruction), seed,
                             args.out_dir / f"{len(jobs):02d}_{slug}_seed{seed}.png"))

    print(f"model     {model_dir}")
    print(f"reference {args.reference}")
    print(f"out       {args.out_dir}  ({len(jobs)} image"
          f"{'' if len(jobs) == 1 else 's'})")
    if args.prompt:
        print("poses     OVERRIDDEN by --prompt: every image uses the same text, "
              "so they will differ only by seed.\n          On an editing model "
              "that is a weak lever - expect near-identical output.")
    else:
        slugs = [j[0] for j in jobs]
        print(f"poses     {', '.join(dict.fromkeys(slugs))}")
        off = [s_ for s_ in dict.fromkeys(slugs) if s_ in OFF_FRONT]
        if off:
            verb = "is" if len(off) == 1 else "are"
            print(f"          ({', '.join(off)} {verb} off-front: fine to generate, "
                  "but IDM-VTON\n           is trained on front views and stage 30 "
                  "will degrade on them)")
    if "no text" in jobs[0][1].lower() or "remove all text" in jobs[0][1].lower():
        print("text      removal requested - CHECK THE OUTPUT, generative removal "
              "is not guaranteed")

    if args.dry_run:
        print("\nWould write:")
        for slug, prompt, seed, dest in jobs:
            print(f"  {dest.name}")
        print(f"\nPrompt for {jobs[0][0]}:\n  {jobs[0][1]}")
        print("\nDry run - nothing loaded.")
        return 0

    import torch
    from PIL import Image

    pipe = load_pipeline(model_dir, offload=args.offload)
    if pipe is None:
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = Image.open(args.reference).convert("RGB")

    for i, (slug, prompt, seed, dest) in enumerate(jobs):
        if dest.exists():
            print(f"  [{i+1}/{len(jobs)}] {dest.name} exists, skipping")
            continue

        kwargs = supported(pipe.__call__, {
            "image": reference,
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.steps,
            "true_cfg_scale": args.guidance,
            "guidance_scale": args.guidance,
            "width": args.width,
            "height": args.height,
            "generator": torch.Generator("cuda").manual_seed(seed),
            "num_images_per_prompt": 1,
        })
        print(f"  [{i+1}/{len(jobs)}] {slug}, seed {seed} -> {dest.name}")
        out = pipe(**kwargs)
        out.images[0].save(dest)

    print(f"\nWrote {len(jobs)} variants to {args.out_dir.resolve()}")
    print("Next (back in the IDM-VTON env):")
    print("  python scripts/pipeline/20_leg_masks.py --in-dir", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
