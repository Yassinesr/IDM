#!/usr/bin/env python
"""Stage 1 - generate extra views of a reference model photo with Qwen-Image-Edit.

Runs in the QWEN env, not the IDM-VTON one:
    source $IDM_ROOT/venv-qwen/bin/activate
    python scripts/pipeline/10_generate_views.py --reference <photo> --count 8

Weights are read straight from the read-only shared mount, so nothing is
downloaded and nothing is written outside --out-dir.
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "/root/public/models/Qwen/Qwen-Image-Edit-2511"

# This is an editing model, so the prompt is an instruction about the reference,
# not just a description of the output. Removing source text is stated first and
# concretely: a reference carrying a brand mark is otherwise imitated, and a
# negative prompt alone is a weak lever against something visible in the input.
DEFAULT_PROMPT = (
    "Remove all text, logos, watermarks, brand names and corner badges from the "
    "image, reconstructing the background cleanly where they were. "
    "Then produce a Taobao product display photograph in the same style as the "
    "reference: full-body shot of a female model, head to feet fully in frame, "
    "standing, facing the camera, plain seamless studio background, soft even "
    "lighting, high resolution e-commerce photography. "
    "The final image must contain no text of any kind."
)

DEFAULT_NEGATIVE = (
    "text, watermark, logo, brand name, letters, caption, signature, "
    "cropped head, cropped feet, close-up, collage, multiple people, "
    "blurry, distorted anatomy, extra limbs"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True,
                    help="the seed photo, e.g. gradio_demo/example/human/model_front.jpg")
    ap.add_argument("--out-dir", type=Path, default=Path("work/variants"))
    ap.add_argument("--count", type=int, default=8, help="how many variants")
    ap.add_argument("--model", default=os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
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


def supported(fn, wanted: dict) -> dict:
    """Keep only kwargs this pipeline actually accepts.

    Qwen's editing pipelines have changed argument names between releases
    (guidance_scale vs true_cfg_scale, image vs control_image), so ask the
    signature instead of assuming.
    """
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return wanted
    keep = {k: v for k, v in wanted.items() if k in params}
    dropped = sorted(set(wanted) - set(keep))
    if dropped:
        print(f"    (pipeline does not take: {', '.join(dropped)})")
    return keep


def main():
    args = parse_args()

    if not args.reference.is_file():
        print(f"ERROR: reference not found: {args.reference}", file=sys.stderr)
        return 1

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"ERROR: model not found at {model_dir}", file=sys.stderr)
        print("       Check: ls /root/public/models/Qwen/", file=sys.stderr)
        return 1
    if not (model_dir / "model_index.json").is_file():
        print(f"ERROR: {model_dir} has no model_index.json, so it is not a\n"
              "       diffusers-format checkout. Look for a subdirectory that has one.",
              file=sys.stderr)
        return 1

    print(f"model     {model_dir}")
    print(f"reference {args.reference}")
    print(f"out       {args.out_dir}  ({args.count} variants, seeds "
          f"{args.seed}..{args.seed + args.count - 1})")
    if "no text" in args.prompt.lower() or "remove all text" in args.prompt.lower():
        print("text      removal requested in the prompt - CHECK THE OUTPUT, "
              "generative removal is not guaranteed")
    if args.dry_run:
        print("\nDry run - nothing loaded.")
        return 0

    # An overlay is a directory of just the NEW libraries (diffusers,
    # transformers), installed with `pip install --target`. Put first on
    # sys.path it shadows the IDM-VTON env's pinned versions while reusing its
    # torch - a few hundred MB instead of a second ~10 GB torch stack.
    overlay = os.environ.get("QWEN_OVERLAY")
    if overlay:
        if not Path(overlay).is_dir():
            print(f"ERROR: QWEN_OVERLAY={overlay} is not a directory", file=sys.stderr)
            return 1
        sys.path.insert(0, overlay)
        print(f"overlay   {overlay} (shadowing the env's diffusers/transformers)")

    import torch
    from PIL import Image
    try:
        import diffusers
        from diffusers import DiffusionPipeline
    except (ImportError, AttributeError) as exc:
        # AttributeError too: modern diffusers touches torch.xpu at import, which
        # torch < 2.4 does not have, and that is not an ImportError.
        if overlay:
            # Classic overlay symptom: a new library importing a symbol that
            # only exists in a newer version of a package it did NOT shadow.
            print(f"\nERROR: {exc}", file=sys.stderr)
            if "xpu" in str(exc):
                print("\n       This is the torch floor, not a missing package.\n"
                      "       QwenImageEditPlusPipeline needs diffusers >= 0.36, and\n"
                      "       diffusers >= 0.34 touches torch.xpu at import, which\n"
                      "       arrived in torch 2.4. The base env has torch 2.0.1, so\n"
                      "       overlay mode cannot work here - it needs a full venv with\n"
                      "       its own newer torch:\n"
                      "         bash scripts/pipeline/00_setup_qwen_env.sh",
                      file=sys.stderr)
                return 1
            print("\n       The overlay shadows only the packages it installed; this one\n"
                  "       came from the base env at its older pinned version. Rebuild\n"
                  "       the overlay with it added:\n"
                  '         QWEN_OVERLAY_PKGS="diffusers>=0.35 transformers>=4.51 \\\n'
                  '             tokenizers huggingface_hub>=0.27 safetensors accelerate" \\\n'
                  "             bash scripts/pipeline/00_setup_qwen_env.sh --overlay",
                  file=sys.stderr)
            return 1
        raise
    import huggingface_hub
    print(f"          torch {torch.__version__}, diffusers {diffusers.__version__}, "
          f"hub {huggingface_hub.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = Image.open(args.reference).convert("RGB")

    # from_pretrained on a local dir reads model_index.json and builds whatever
    # pipeline class it names - so this does not hard-code a class that may be
    # renamed between Qwen releases.
    print("loading (54 GB off the shared mount; first load is slow)")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    print(f"    pipeline: {type(pipe).__name__}")

    if args.offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    for i in range(args.count):
        seed = args.seed + i
        dest = args.out_dir / f"variant_{i:02d}_seed{seed}.png"
        if dest.exists():
            print(f"  [{i+1}/{args.count}] {dest.name} exists, skipping")
            continue

        kwargs = supported(pipe.__call__, {
            "image": reference,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.steps,
            "true_cfg_scale": args.guidance,
            "guidance_scale": args.guidance,
            "width": args.width,
            "height": args.height,
            "generator": torch.Generator("cuda").manual_seed(seed),
            "num_images_per_prompt": 1,
        })
        print(f"  [{i+1}/{args.count}] seed {seed} -> {dest.name}")
        out = pipe(**kwargs)
        out.images[0].save(dest)

    print(f"\nWrote {args.count} variants to {args.out_dir.resolve()}")
    print("Next (back in the IDM-VTON env):")
    print("  python scripts/pipeline/20_leg_masks.py --in-dir", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
