# Generating Taobao display images: reference photo → views → leg masks → try-on

Three stages, each writing files the next one reads.

```
model_front.jpg
   │  stage 10   Qwen-Image-Edit-2511            [venv-qwen]
   ▼
work/variants/variant_NN.png          N synthetic full-body views
   │  stage 20   human parsing + ankle clip      [IDM-VTON venv]
   ▼
work/masks/variant_NN.mask.png        white = leg area, black = rest   ← deliverable
   │  stage 30   IDM-VTON try-on                 [IDM-VTON venv]
   ▼
work/results/variant_NN.tryon.png     the garment on every view
```

## Why two environments

IDM-VTON is pinned to `diffusers==0.25.0` because `src/unet_hacked_*.py` subclass
its internals. Qwen-Image needs a modern diffusers. They cannot share an
environment, so the stages communicate through files rather than in-process.

| | env | diffusers |
|---|---|---|
| stage 10 | `$IDM_ROOT/venv-qwen` | latest |
| stages 20, 30 | `$IDM_ROOT/venv` | 0.25.0 |

## The weights are already on the cluster

```
/root/public/models/Qwen/Qwen-Image             54 GB
/root/public/models/Qwen/Qwen-Image-Edit-2511   54 GB   <- image-to-image
```

Read-only, shared, and loaded straight from that path — nothing is downloaded
and nothing lands on your disk quota. Alaya's `huggingface-cli download`
instructions are therefore unnecessary here; they would also need
`HF_ENDPOINT=https://hf-mirror.com`, since huggingface.co is not reachable from
the cluster.

What *does* cost disk is the second venv — roughly 10 GB, mostly torch's
bundled CUDA libraries.

### When there is no room for a second torch

```bash
source scripts/alaya/env.sh
bash scripts/pipeline/00_setup_qwen_env.sh --overlay
export QWEN_OVERLAY=$IDM_ROOT/qwen-overlay
python scripts/pipeline/10_generate_views.py --reference ... --count 3
```

Overlay mode installs *only* the new libraries — diffusers, transformers,
tokenizers — into a directory stage 10 puts first on `sys.path`. They shadow the
IDM-VTON env's pinned versions while its torch is reused, so the cost is a few
hundred MB instead of ~10 GB.

The trade: Qwen then runs on whatever torch that env has (2.0.1), which is older
than it expects. If it refuses, that is the answer — the full venv is the clean
fix once there is disk. Stage 10 prints the torch and diffusers versions it
actually loaded, so there is no guessing about which took effect.

### The pipeline class

`Qwen-Image-Edit-2511`'s `model_index.json` names **`QwenImageEditPlusPipeline`**.
Running stage 10 in the IDM-VTON env fails with
`module diffusers has no attribute QwenImageEditPlusPipeline` — that is the
pinned 0.25.0 diffusers, not a broken model. It is a useful smoke test: if you
see that error, the wrong environment is active.

## Running it

```bash
# once
bash scripts/pipeline/00_setup_qwen_env.sh

# stage 10 - generate views
source $IDM_ROOT/venv-qwen/bin/activate
python scripts/pipeline/10_generate_views.py \
    --reference gradio_demo/example/human/model_front.jpg \
    --out-dir work/variants --count 8
deactivate

# stages 20 and 30 - masks, then try-on
source scripts/alaya/env.sh
python scripts/pipeline/20_leg_masks.py --in-dir work/variants \
    --out-dir work/masks --overlay
python scripts/pipeline/30_tryon_batch.py --in-dir work/variants \
    --garment gradio_demo/example/cloth/yoga_pants.jpg \
    --desc "plain mauve high-waisted leggings" --out-dir work/results
```

## Stage 20: what "leg area, excluding feet" means here

Two regions, because "leg area" can mean either thing:

- `--region leg_area` (default) — parse labels `skirt(5)`, `pants(6)`,
  `left_leg(12)`, `right_leg(13)`. The limbs and whatever covers them, at their
  true silhouette. This is the segmentation deliverable.
- `--region inpaint` — exactly what stage 30 will repaint, via
  `get_mask_location`. Dilated and hole-filled, so deliberately looser than the
  silhouette. Useful for checking stage 30, misleading as a segmentation.

**Feet** are excluded by cutting at the ankle, from OpenPose keypoints 10 (right)
and 13 (left). ATR has no foot label, so bare feet fall under `left_leg`/
`right_leg` and would otherwise be included. The cut uses the *higher* ankle, so
both feet are excluded when one foot is forward; `--ankle-offset` shifts it, and
`--keep-feet` disables it. When neither ankle is detected the mask is written
uncut, with a warning, rather than silently guessing.

`--overlay` writes a red-tinted composite next to each mask — the fastest way to
check the fit without opening two files.

## Things that will bite

**Text removal is asked of the model, and is not guaranteed.**
`model_front.jpg` carries the "CrzYoga" logo and a 黄油系列 badge. The default
prompt opens with an explicit instruction to remove all text, logos, watermarks
and corner badges and reconstruct the background behind them — Qwen-Image-Edit
is an editing model, so this is the kind of thing it is built for, and an
instruction is a much stronger lever than a negative prompt against something
visible in the input.

It can still fail, or leave a smudge where the mark was. Stage 10 prints a
reminder to check, and the brief treats *no text* as a hard requirement, so
inspect every variant before stage 30. If a particular mark proves stubborn,
cropping or inpainting it out of the reference first is deterministic where the
model is not.

**Artifacts compound.** Stage 10 invents a person, stage 30 repaints her
clothes. Treat stage 10 output as candidates, inspect them, and discard the bad
ones before stage 30 — cheaper than discovering it in the results.

**VRAM.** Qwen-Image is 20B; with its text encoder it approaches the 80 GB card.
Never run stages 10 and 30 at once. `--offload` trades speed for headroom.

**Stage 30 skips undetectable views.** A generated image the pose estimator
cannot read is reported and skipped, not fatal — a bad variant should not kill
the batch.
