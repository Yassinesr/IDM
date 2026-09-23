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
# Arrows and dashed guide lines are named alongside the text: a Taobao
# marketing crop carries all three, and an editing model reproduces whatever it
# is not told to remove.
TEXT_REMOVAL = (
    "Remove all text, numbers, logos, watermarks, brand names, corner badges, "
    "arrows and dashed guide lines, reconstructing the garment and background "
    "cleanly underneath them. "
)

# What to preserve, when the reference already shows a whole person.
IDENTITY = (
    " Keep the same woman, the same face, the same hairstyle and exactly the "
    "same clothing."
)

# What to do instead when it does not - a half-body product crop, where there
# is no face to keep and asking for one is a contradiction. The garment is the
# thing that must survive; the rest is invented.
EXTEND = (
    " This photograph shows only part of the body. Extend it into a complete "
    "full-body shot showing the whole person from the top of the head to the "
    "feet. Keep the garment exactly as it appears - the same colour, the same "
    "fabric, the same cut, the same length."
)

# Added only when no model has been chosen: with --models the person is
# specified, and telling the model to match the visible skin tone as well would
# be two instructions about the same thing.
EXTEND_INFER_PERSON = (
    " Invent a head, face, hair, torso and arms that suit the visible body and "
    "match its skin tone."
)

# One entry per model. Retail sizing and colour read differently on different
# bodies, so a garment shown on one build is a thin sample; these let the same
# pose and the same garment be rendered on several. Slugs name a visible
# feature so a filename says which is which.
MODELS = [
    ("straight_black_slim",
     "a young East Asian woman with long straight black hair, fair skin and a "
     "slim build"),
    ("wavy_auburn_athletic",
     "a young woman with light skin, shoulder-length wavy auburn hair and an "
     "athletic build"),
    ("short_curls_tall",
     "a young Black woman with deep brown skin, short natural curls and a tall "
     "slender build"),
    ("dark_ponytail_curvy",
     "a South Asian woman with medium brown skin, long dark hair in a ponytail "
     "and a curvy build"),
    ("olive_bob_midsize",
     "a woman in her thirties with olive skin, a short dark bob and a mid-size "
     "build"),
    ("blonde_petite",
     "a young woman with pale skin, long blonde hair and a petite build"),
    ("brown_waves_hourglass",
     "a Latina woman with warm brown skin, long dark wavy hair and an "
     "hourglass figure"),
    ("grey_bun_plus",
     "a woman in her forties with light skin, greying hair in a low bun and a "
     "plus-size build"),
]

MODEL_BY_SLUG = {slug: text for slug, text in MODELS}

# Replaces IDENTITY: the point of choosing a model is that the person changes,
# so "keep the same woman" would be the opposite instruction. The garment is
# what must be held fixed instead.
MODEL_CLAUSE = " The model is {desc}."

# Only when EXTEND has not already said it - repeating the garment clause twice
# in one prompt reads as a mistake and buys nothing.
KEEP_GARMENT = (
    " Keep exactly the same clothing as in the reference - the same colour, "
    "the same fabric, the same cut, the same length."
)

# True of every variant. Deliberately says nothing about which way the model
# faces - that is the pose's job, and a style block repeating "facing the
# camera" would fight every pose that is not a front view.
FRAMING = (
    " Taobao e-commerce product photograph: full-body shot, the top of the head "
    "and both feet fully inside the frame, plain seamless light grey studio "
    "background, soft even lighting, sharp focus, high resolution. The final "
    "image must contain no text of any kind."
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


def build_prompt(pose_instruction: str, extend: bool = False,
                 model_desc: str = None) -> str:
    """Edit instruction first, constraints after - that is the order this kind
    of model weights most heavily."""
    parts = [TEXT_REMOVAL, pose_instruction]
    if extend:
        parts.append(EXTEND)
        if model_desc is None:
            parts.append(EXTEND_INFER_PERSON)
    if model_desc:
        parts.append(MODEL_CLAUSE.format(desc=model_desc))
        if not extend:
            parts.append(KEEP_GARMENT)
    else:
        parts.append(IDENTITY)
    parts.append(FRAMING)
    return "".join(parts)


def select_models(spec):
    """Resolve --models into [(slug, description)], or [(None, None)] for one
    run with whatever person the reference or the model itself supplies."""
    if spec in (None, ""):
        return [(None, None)]
    if spec == "all":
        return MODELS
    if spec.isdigit():
        n = int(spec)
        if not 1 <= n <= len(MODELS):
            raise SystemExit(f"ERROR: --models takes 1..{len(MODELS)}, got {n}")
        return MODELS[:n]
    chosen = []
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        if name not in MODEL_BY_SLUG:
            raise SystemExit(
                f"ERROR: unknown model {name!r}. Run --list-models to see them.")
        chosen.append((name, MODEL_BY_SLUG[name]))
    return chosen


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
    "text, watermark, logo, brand name, letters, caption, signature, arrows, "
    "dashed lines, cropped head, cropped feet, half body, waist-up crop, "
    "close-up, collage, multiple people, blurry, distorted anatomy, extra limbs"
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
    ap.add_argument("--models", default=None,
                    help="render every pose on several different people: a "
                         "count (5), a comma-separated list of names, or 'all'")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model table and exit")
    ap.add_argument("--repeat", type=int, default=1,
                    help="images per pose, each with its own seed")
    ap.add_argument("--extend", action="store_true",
                    help="the reference is a partial crop (no head, or no "
                         "feet): invent the missing body instead of asking to "
                         "keep a face that is not there")
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

    if args.list_models:
        print(f"{len(MODELS)} models (--models 5, --models "
              f"{MODELS[0][0]},{MODELS[1][0]}, or --models all):\n")
        for slug, text in MODELS:
            print(f"  {slug}\n      {text}\n")
        return 0

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
        models = select_models(args.models)
        for pi, (slug, instruction) in enumerate(select_poses(args.poses, args.count)):
            for mi, (mslug, mdesc) in enumerate(models):
                for r in range(args.repeat):
                    # Each pose and each model gets its own seed block, so
                    # adding one never renumbers another one's output.
                    seed = args.seed + pi * 1000 + mi * 10 + r
                    name = f"{len(jobs):02d}_{slug}"
                    if mslug:
                        name += f"_{mslug}"
                    jobs.append((slug,
                                 build_prompt(instruction, args.extend, mdesc),
                                 seed, args.out_dir / f"{name}_seed{seed}.png"))

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
        models = select_models(args.models)
        if args.models:
            print(f"models    {', '.join(m for m, _ in models)}")
            print("          the person changes between these; only the "
                  "garment is held fixed")
        if args.extend:
            print("extend    the reference is a crop; the missing body is "
                  "invented.")
            if not args.models:
                print("          Only the garment is held fixed - the face and "
                      "build are not\n          in the input and will differ "
                      "between seeds.")
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
