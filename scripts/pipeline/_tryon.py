"""Shared IDM-VTON pipeline: load once, run many.

demo_single.py loads the model, does one image and exits, which is right for a
single test and wasteful for a batch - the load is minutes. This holds the same
pipeline open across a directory of images.

The per-image steps mirror demo_single.py exactly; if one changes, change both.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# apply_net is called with relative './configs' and './ckpt' paths.
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gradio_demo"))

NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality"


class TryOn:
    def __init__(self, model_path="yisol/IDM-VTON", device="cuda:0"):
        import torch
        from torchvision import transforms
        from transformers import (CLIPImageProcessor, CLIPTextModel,
                                  CLIPTextModelWithProjection,
                                  CLIPVisionModelWithProjection, AutoTokenizer)
        from diffusers import DDPMScheduler, AutoencoderKL
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNetRef
        from src.unet_hacked_tryon import UNet2DConditionModel as UNetTryon
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        self.torch, self.device = torch, device
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        fp16 = dict(torch_dtype=torch.float16)
        unet = UNetTryon.from_pretrained(model_path, subfolder="unet", **fp16)
        unet_encoder = UNetRef.from_pretrained(model_path, subfolder="unet_encoder", **fp16)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", **fp16)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_path, subfolder="image_encoder", **fp16)
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", **fp16)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_path, subfolder="text_encoder_2", **fp16)
        for m in (unet, unet_encoder, vae, image_encoder, text_encoder, text_encoder_2):
            m.requires_grad_(False)

        self.pipe = TryonPipeline.from_pretrained(
            model_path, unet=unet, vae=vae, feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            tokenizer=AutoTokenizer.from_pretrained(
                model_path, subfolder="tokenizer", revision=None, use_fast=False),
            tokenizer_2=AutoTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2", revision=None, use_fast=False),
            scheduler=DDPMScheduler.from_pretrained(model_path, subfolder="scheduler"),
            image_encoder=image_encoder, **fp16)
        self.pipe.unet_encoder = unet_encoder
        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)

        self.parsing = Parsing(0)
        self.openpose = OpenPose(0)
        self.openpose.preprocessor.body_estimation.model.to(device)

    def run(self, human_img, garm_img, desc, category="lower_body",
            steps=30, seed=42, guidance=2.0):
        """One try-on. Returns (result, mask). Raises IndexError if no person."""
        import numpy as np
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image
        from torchvision import transforms
        from detectron2.data.detection_utils import (convert_PIL_to_numpy,
                                                     _apply_exif_orientation)
        import apply_net
        from utils_mask import get_mask_location
        torch = self.torch

        human_img = human_img.convert("RGB").resize((768, 1024))
        garm_img = garm_img.convert("RGB").resize((768, 1024))

        small = human_img.resize((384, 512))
        keypoints = self.openpose(small)          # IndexError when nobody is found
        model_parse, _ = self.parsing(small)
        mask, _ = get_mask_location("hd", category, model_parse, keypoints)
        mask = mask.resize((768, 1024))

        arg_img = convert_PIL_to_numpy(_apply_exif_orientation(small), format="BGR")
        dp = apply_net.create_argument_parser().parse_args(
            ("show", "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
             "./ckpt/densepose/model_final_162be9.pkl", "dp_segm", "-v",
             "--opts", "MODEL.DEVICE", "cuda"))
        pose_img = Image.fromarray(dp.func(dp, arg_img)[:, :, ::-1]).resize((768, 1024))

        with torch.no_grad(), torch.cuda.amp.autocast(), torch.inference_mode():
            (pe, npe, ppe, nppe) = self.pipe.encode_prompt(
                "model is wearing " + desc, num_images_per_prompt=1,
                do_classifier_free_guidance=True, negative_prompt=NEGATIVE)
            (pec, _, _, _) = self.pipe.encode_prompt(
                ["a photo of " + desc], num_images_per_prompt=1,
                do_classifier_free_guidance=False, negative_prompt=[NEGATIVE])

            d, f16 = self.device, torch.float16
            images = self.pipe(
                prompt_embeds=pe.to(d, f16), negative_prompt_embeds=npe.to(d, f16),
                pooled_prompt_embeds=ppe.to(d, f16),
                negative_pooled_prompt_embeds=nppe.to(d, f16),
                num_inference_steps=steps,
                generator=torch.Generator(d).manual_seed(seed),
                strength=1.0,
                pose_img=self.transform(pose_img).unsqueeze(0).to(d, f16),
                text_embeds_cloth=pec.to(d, f16),
                cloth=self.transform(garm_img).unsqueeze(0).to(d, f16),
                mask_image=mask, image=human_img, height=1024, width=768,
                ip_adapter_image=garm_img, guidance_scale=guidance)[0]
        return images[0], mask
