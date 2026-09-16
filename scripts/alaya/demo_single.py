#!/usr/bin/env python
"""Headless single-image try-on - the fastest way to prove a Workshop works.

The gradio demo needs a forwarded port and a browser. This runs the exact same
pipeline from the terminal on one person + one garment and writes a PNG, so a
broken environment shows up as a stack trace you can paste rather than a blank
browser tab.

    source scripts/alaya/env.sh
    python scripts/alaya/demo_single.py                    # bundled examples
    python scripts/alaya/demo_single.py --list             # what's available
    python scripts/alaya/demo_single.py \
        --human  gradio_demo/example/human/00034_00.jpg \
        --garment gradio_demo/example/cloth/04469_00.jpg \
        --desc "a red short-sleeve t-shirt" \
        --output out.png

The mask is generated automatically (human parsing + openpose), which is the
"auto-generated mask" checkbox in the gradio UI.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# apply_net is invoked below with RELATIVE paths ('./configs/...', './ckpt/...'),
# exactly as gradio_demo/app.py does, so the process must run from the repo root.
# Remember where the user actually was, so their own relative paths still work.
ORIG_CWD = Path.cwd()
os.chdir(REPO)
# app.py gets both of these for free: python puts gradio_demo/ on the path as the
# script's own directory, and the file does sys.path.append('./') for the root.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gradio_demo"))

EXAMPLE = REPO / "gradio_demo" / "example"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", type=Path, help="person image (default: a bundled example)")
    ap.add_argument("--garment", type=Path, help="garment image (default: a bundled example)")
    ap.add_argument("--desc", default="a short sleeve t-shirt",
                    help="garment description; it goes into the prompt, so it matters")
    ap.add_argument("--output", type=Path, default=Path("demo_out.png"))
    ap.add_argument("--category", default="upper_body",
                    choices=["upper_body", "lower_body", "dresses"])
    ap.add_argument("--steps", type=int, default=30, help="denoising steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--crop", action="store_true",
                    help="centre-crop to 3:4 first; useful for phone photos, "
                         "not for VITON-HD-style images that are already 3:4")
    ap.add_argument("--save-mask", action="store_true",
                    help="also write <output>.mask.png to debug a bad mask")
    ap.add_argument("--model", default=os.environ.get("IDM_MODEL_PATH", "yisol/IDM-VTON"),
                    help="Hub id, or a local directory holding unet/, unet_encoder/, "
                         "vae/ etc. Defaults to $IDM_MODEL_PATH, else the Hub id. "
                         "Use scripts/alaya/find_local_models.py to locate one.")
    ap.add_argument("--list", action="store_true", help="list bundled examples and exit")
    args = ap.parse_args()
    # Resolve user paths against the caller's cwd, not the repo root we chdir'd to.
    for field in ("human", "garment", "output"):
        val = getattr(args, field)
        if val is not None and not val.is_absolute():
            setattr(args, field, (ORIG_CWD / val).resolve())
    return args


def _rel(p: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def pick_examples(args):
    humans = sorted(p for p in (EXAMPLE / "human").iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png"})
    cloths = sorted(p for p in (EXAMPLE / "cloth").iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png"})
    if args.list:
        print(f"humans ({len(humans)}):")
        for p in humans:
            print(f"  {p.relative_to(REPO)}")
        print(f"\ngarments ({len(cloths)}):")
        for p in cloths:
            print(f"  {p.relative_to(REPO)}")
        raise SystemExit(0)
    # 00034_00.jpg is a plain front-facing VITON-HD shot - the least likely
    # default to produce a confusing first result.
    human = args.human or next((p for p in humans if p.name == "00034_00.jpg"), humans[0])
    garment = args.garment or next((p for p in cloths if p.name == "04469_00.jpg"), cloths[0])
    return human, garment


def main():
    args = parse_args()
    human_path, garment_path = pick_examples(args)

    for p in (human_path, garment_path):
        if not p.exists():
            print(f"ERROR: {p} does not exist", file=sys.stderr)
            return 1

    # Validate the model path before importing torch: loading torch takes long
    # enough that finding out afterwards is needlessly annoying.
    base_path = args.model
    if os.path.isdir(base_path):
        # A local checkout: no network, and it works on a read-only mount.
        missing = [d for d in ("unet", "unet_encoder", "vae", "text_encoder",
                               "text_encoder_2", "image_encoder", "tokenizer",
                               "tokenizer_2", "scheduler")
                   if not os.path.isdir(os.path.join(base_path, d))]
        if missing:
            print(f"ERROR: {base_path} is not a complete IDM-VTON checkout.",
                  file=sys.stderr)
            print(f"       missing subfolders: {', '.join(missing)}", file=sys.stderr)
            return 1
        print(f"model    {base_path} (local, no download)")
    elif "/" in base_path and not base_path.startswith("."):
        print(f"model    {base_path} (Hub)")
    else:
        print(f"ERROR: {base_path} is neither a directory nor a Hub id.",
              file=sys.stderr)
        return 1

    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device. The models load in float16 and the "
              "preprocessors call torch.cuda.set_device(); CPU is not a "
              "supported path here.", file=sys.stderr)
        return 1
    device = "cuda:0"

    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from transformers import (CLIPImageProcessor, CLIPTextModel,
                              CLIPTextModelWithProjection,
                              CLIPVisionModelWithProjection, AutoTokenizer)
    from diffusers import DDPMScheduler, AutoencoderKL

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from detectron2.data.detection_utils import (convert_PIL_to_numpy,
                                                 _apply_exif_orientation)
    import apply_net
    from utils_mask import get_mask_location

    print(f"human    {_rel(human_path)}")
    print(f"garment  {_rel(garment_path)}")
    print(f"desc     {args.desc!r}")
    if not os.path.isdir(base_path):
        print("loading (first run downloads tens of GB into $HF_HOME)")
    else:
        print("loading")
    t0 = time.time()

    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet",
                                                torch_dtype=torch.float16)
    unet.requires_grad_(False)
    tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer",
                                                  revision=None, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2",
                                                  revision=None, use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
    text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder",
                                                     torch_dtype=torch.float16)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path, subfolder="image_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae",
                                        torch_dtype=torch.float16)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

    for m in (unet_encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two):
        m.requires_grad_(False)

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    tensor_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    pipe = TryonPipeline.from_pretrained(
        base_path, unet=unet, vae=vae, feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one, text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler, image_encoder=image_encoder,
        torch_dtype=torch.float16)
    pipe.unet_encoder = unet_encoder
    print(f"loaded in {time.time() - t0:.0f}s")

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = Image.open(garment_path).convert("RGB").resize((768, 1024))
    human_img_orig = Image.open(human_path).convert("RGB")

    if args.crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        cropped = human_img_orig.crop((left, top, left + target_width, top + target_height))
        crop_size = cropped.size
        human_img = cropped.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    print("preprocessing: openpose -> human parsing -> mask -> densepose")
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _ = get_mask_location("hd", args.category, model_parse, keypoints)
    mask = mask.resize((768, 1024))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    dp_args = apply_net.create_argument_parser().parse_args(
        ("show", "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
         "./ckpt/densepose/model_final_162be9.pkl", "dp_segm", "-v",
         "--opts", "MODEL.DEVICE", "cuda"))
    pose_img = dp_args.func(dp_args, human_img_arg)
    pose_img = Image.fromarray(pose_img[:, :, ::-1]).resize((768, 1024))

    print(f"generating ({args.steps} steps, seed {args.seed})")
    t1 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        prompt = "model is wearing " + args.desc
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        with torch.inference_mode():
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)

            # A second, garment-only encoding; the pipeline conditions the
            # garment UNet on this separately.
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                ["a photo of " + args.desc], num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt=[negative_prompt])

            pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(args.seed)

            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=args.steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024, width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=args.guidance_scale,
            )[0]

    if args.crop:
        out = images[0].resize(crop_size)
        human_img_orig.paste(out, (int(left), int(top)))
        result = human_img_orig
    else:
        result = images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"generated in {time.time() - t1:.0f}s")
    print(f"wrote {args.output.resolve()}")
    if args.save_mask:
        mask_path = args.output.with_suffix(".mask.png")
        mask_gray.save(mask_path)
        print(f"wrote {mask_path.resolve()}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak GPU memory: {peak:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
