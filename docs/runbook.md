# IDM-VTON on Alaya NeW: end-to-end runbook

Bare container to garment-on-model images, in order. Every command is meant to
be copied as written.

Two other documents go deeper where this one is brief:
[`alaya-new-setup.md`](alaya-new-setup.md) for the platform itself, and
[`taobao-pipeline.md`](taobao-pipeline.md) for the three-stage pipeline's
design.

---

## What you are building

```
reference photo
      │
      ▼  stage 10 - Qwen-Image-Edit          (env: venv-qwen)
work/variants/NN_<pose>.png                  one full-body view per pose
      │
      ├──▶ stage 20 - leg masks              (env: venv)
      │    work/masks/NN_<pose>.mask.png     white = leg area, black = rest
      │
      ▼  stage 30 - IDM-VTON try-on          (env: venv)
work/results/NN_<pose>.tryon.png             the garment on every view
```

**Two environments, and they cannot be merged.** IDM-VTON is pinned to
`diffusers==0.25.0` because `src/unet_hacked_*.py` subclass its internals, and
it runs on torch 2.0.1. `QwenImageEditPlusPipeline` needs diffusers ≥ 0.36,
which touches `torch.xpu` at import and so needs torch ≥ 2.4. The stages hand
off through files on disk instead of sharing a process.

**The one rule.** Shut the container down (关机), never release it (释放). A
shutdown saves the image and keeps the 31 GB of weights. A release destroys
them, and re-downloading over a throttled link is a long afternoon.

---

## 0. Create the container

Alaya console → 弹性容器集群 → create a CCI:

| Field | Value | Why |
|---|---|---|
| Framework | `ubuntu` 22.04, python 3.10, CUDA 13.0.2 | 3.10 matches the pins |
| Resource | any 80 GB card (H100/H800) | Qwen is ~55 GB loaded; smaller cards need `--offload` |
| Storage → Container Path | **set it if you can** | this is the only thing that survives a release |
| ENV | see below | |

```
IDM_ROOT=/pvc/idm
IDM_ALLOW_EPHEMERAL=1
HF_ENDPOINT=https://hf-mirror.com
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

`IDM_ALLOW_EPHEMERAL=1` is only correct when no PVC could be attached. It tells
the bootstrap you accept that `IDM_ROOT` dies with the container. Leave it out
if you have real storage — the guard exists to catch the accident.

If you are sharing a container, set `IDM_USER=<yourname>` too and everything
moves under `/pvc/users/$IDM_USER/idm`.

---

## 1. First connection

Open the Workshop in VS Code (Aladdin extension), then in its terminal:

```bash
bash -c 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
```

### SSH to GitHub

HTTPS to github.com is blocked from Beijing, and so is SSH on port 22. GitHub
also serves SSH on **443**, which gets through. If you kept a key from an
earlier container, paste it back:

```bash
vi ~/.ssh/id_ed25519          # paste, save
chmod 600 ~/.ssh/id_ed25519
```

Otherwise the sync script generates one and prints the public half for you to
add at <https://github.com/settings/keys>.

### Clone

```bash
mkdir -p /pvc/idm && cd /pvc/idm
git clone -b claude/alaya-new-cloud-setup-4ode4d \
    git@github.com:Yassinesr/IDM.git IDM
cd /pvc/idm/IDM
```

If the clone fails because SSH is not configured yet, run
`bash scripts/alaya/sync_from_github.sh` from a copy of the repo — it sets up
`ssh.github.com:443`, verifies the host key against GitHub's published
fingerprint, and fetches.

### Make the shell remember

A new SSH session starts with none of this set, and then `$IDM_ROOT/venv`
expands to `/venv` and everything looks deleted:

```bash
cat >> ~/.bashrc <<'EOF'
export IDM_ROOT=/pvc/idm
export IDM_ALLOW_EPHEMERAL=1
EOF
```

Only the variables. Do not source `env.sh` from `.bashrc` — it activates an
environment, which is noise while you are switching between two of them.

---

## 2. Check what storage you actually have

```bash
bash scripts/alaya/check_storage.sh
```

It classifies every mount as persistent or ephemeral. Worth one minute before
you download 31 GB onto something that will not survive the night.

`/root/public` is the read-only shared model library (457 TB). Qwen is read
from there in place. IDM-VTON is not there — checked, only `sdxl-turbo`, which
has no `unet_encoder` and cannot stand in.

---

## 3. Build the IDM-VTON environment

```bash
cd /pvc/idm/IDM
bash scripts/alaya/bootstrap_all.sh
```

That runs storage check → python 3.10 via Miniconda → `requirements.txt` →
checkpoints → preflight → Qwen env. Each stage is skippable and each is safe to
re-run, so a failure means fixing that one thing and running it again:

```
--skip-storage   --skip-idm   --skip-weights   --skip-qwen
--slim           download the smaller weight set (~half)
--qwen-only      stage 10 only: Qwen env, no IDM-VTON, no weights
```

On a 49 GB disk the Qwen env will not fit next to everything else, so expect
that last stage to fail. That is fine — see §6.

### Weights

The bootstrap calls this, but you can drive it directly:

```bash
python scripts/alaya/01_download_checkpoints.py --dry-run   # sizes, no fetch
python scripts/alaya/01_download_checkpoints.py             # ~31 GB
python scripts/alaya/01_download_checkpoints.py --slim      # skip duplicate formats
```

Resumable — re-run it if the link drops. It subtracts what is already cached
before checking free space.

### Preflight

```bash
source scripts/alaya/env.sh
python scripts/alaya/preflight.py
```

Everything must say PASS before you go further. The ones that bite:

- **numpy 2.x is a FAIL, not a warning.** The whole stack is built against the
  1.x ABI. If you ever `pip install` something that pulls numpy 2, fix it with
  `pip install -r requirements.txt`, not by upgrading around it.
- **`huggingface_hub` must be 0.25.2.** 0.26 removed `cached_download`, which
  diffusers 0.25.0 imports at module level.
- **`opencv-python-headless`, not `opencv-python`.** The container has no
  libGL.

---

## 4. Smoke test: one try-on

```bash
source scripts/alaya/env.sh
python scripts/alaya/demo_single.py --list         # bundled examples

python scripts/alaya/demo_single.py \
    --human gradio_demo/example/human/model_front.jpg \
    --garment gradio_demo/example/cloth/yoga_pants.jpg \
    --desc "plain mauve high-waisted leggings" \
    --category lower_body --save-mask --output demo_out.png
```

`--category` is the one that matters for trousers: `lower_body` masks legs,
`upper_body` masks the torso, `dresses` both. `--save-mask` writes the mask
next to the output so you can see what the model was actually told to repaint.

The gradio UI is the alternative:

```bash
bash scripts/alaya/02_run_gradio.sh
```

VS Code's PORTS panel forwards 7860 to your laptop; if it does not, add it by
hand.

**Input requirements.** A roughly 3:4 full-body photo where the head, neck and
shoulders are visible. OpenPose runs first and raises `IndexError` at
`run_openpose.py:51` when it finds no person — a photo cropped at the waist is
the usual cause, not a broken install.

---

## 5. Getting code changes onto the box

You cannot push from the container and you should not need to. Pull:

```bash
bash scripts/alaya/sync_from_github.sh            # fetch + show the plan
bash scripts/alaya/sync_from_github.sh --force    # reset and clean
bash scripts/alaya/sync_from_github.sh --force --assets   # also example photos
```

`--force` does a `git reset --hard`. The script stashes any real file under
`ckpt/` first and restores it after, because the branch tracks byte-sized
placeholders there and a reset would otherwise drop 950 MB of checkpoints back
to placeholder stubs. Untracked outputs (`work/`, `out/`, `*.png`, `*.jpg`) are
preserved too.

---

## 6. Build the Qwen environment

The Qwen **weights** cost nothing — they are read off `/root/public` in place.
The **env** is ~8 GB, mostly torch's bundled CUDA libraries, and that is what
does not fit.

```bash
bash scripts/alaya/reclaim_disk.sh          # report, deletes nothing
bash scripts/alaya/reclaim_disk.sh --yes    # then delete
```

It reports pip's wheel cache, conda package tarballs, apt lists, `__pycache__`
and loose git objects, and refuses to touch the weights or the venvs. If it
clears ~8 GB, build normally and keep both environments:

```bash
bash scripts/pipeline/00_setup_qwen_env.sh
```

### If it does not fit

Two ways out.

**Put it on another mount.** If some other filesystem has room:

```bash
QWEN_VENV=/some/mount/venv-qwen bash scripts/pipeline/00_setup_qwen_env.sh
```

Treat that mount as scratch — a venv rebuilds from a script, so it is the right
thing to put somewhere you do not fully trust. Never the weights, never
`work/`.

**Or swap the environments.** The stages never run at the same time, so the two
envs never have to coexist:

```bash
rm -rf $IDM_ROOT/venv                       # IDM-VTON env, ~5 GB
bash scripts/pipeline/00_setup_qwen_env.sh
# ... run stage 10 ...
rm -rf $IDM_ROOT/venv-qwen
IDM_ALLOW_EPHEMERAL=1 bash scripts/alaya/00_bootstrap_workshop.sh
```

`requirements.txt` is the source of truth, so nothing is lost — but it is ~15
minutes each way, so generate views in batches rather than one at a time.

### Activating it

`00_setup_qwen_env.sh` uses conda when `$IDM_ROOT/miniconda` exists and
`python3 -m venv` otherwise, and those activate differently. Let the repo
decide:

```bash
source scripts/alaya/_activate.sh
IDM_VENV=$IDM_ROOT/venv-qwen idm_activate
python -c 'import torch,diffusers;print(torch.__version__, diffusers.__version__, torch.cuda.is_available())'
```

Want torch ≥ 2.4, diffusers ≥ 0.36, `True`.

---

## 7. Stage 10 — generate views

```bash
python scripts/pipeline/10_generate_views.py --list-poses

python scripts/pipeline/10_generate_views.py \
    --reference gradio_demo/example/human/model_front.jpg \
    --count 4 --dry-run
```

`--dry-run` prints the exact filenames and the full prompt for the first pose
without loading 54 GB. Then drop it.

**Poses are prompts, not seeds.** Qwen-Image-Edit is an editing model: it is
conditioned on the reference and reproduces it unless told otherwise. Running
it N times with N seeds and one prompt gives N nearly identical images — the
seed only moves what the model is least certain about, which in practice is the
hands. So each variant gets its own pose instruction.

```bash
# the first --count entries of the table
python scripts/pipeline/10_generate_views.py --reference <photo> --count 4

# named poses
python scripts/pipeline/10_generate_views.py --reference <photo> \
    --poses front,three_quarter_left,walking,legs_apart

# everything
python scripts/pipeline/10_generate_views.py --reference <photo> --poses all
```

Other flags: `--repeat K` for K seeds within each pose, `--guidance` (maps to
`true_cfg_scale`, raise toward 6 if poses barely move), `--steps`, `--width`
/`--height`, `--offload` for cards under 80 GB, `--prompt` to replace the whole
table with one text.

First load pulls 54 GB off CephFS, so expect minutes of silence before the
first step counter moves.

### Extending a crop into a full-body model

A half-body product crop cannot go through stages 20 and 30 — IDM-VTON runs
OpenPose first and needs a head, neck and shoulders to find a person at all.
`--extend` turns the crop into a full-body model that can:

```bash
python scripts/pipeline/10_generate_views.py \
    --reference gradio_demo/example/human/only_lower.jpg \
    --extend --poses front,three_quarter_left,walking \
    --out-dir work/variants-lower
```

It swaps the "keep the same woman, the same face" clause — a contradiction when
there is no face in the input — for an instruction to invent a head, torso and
arms that suit the visible body, while holding the garment fixed. Text removal
now names arrows and dashed guide lines too, which is what a marketing crop
carries alongside the text.

Then the usual pipeline, in the IDM env:

```bash
python scripts/pipeline/20_leg_masks.py --in-dir work/variants-lower --overlay
python scripts/pipeline/30_tryon_batch.py --in-dir work/variants-lower \
    --garment gradio_demo/example/cloth/yoga_pants.jpg \
    --desc "plain mauve high-waisted leggings" --category lower_body
```

### Several models, one pose

A garment reads differently on different bodies, so one render is a thin
sample:

```bash
python scripts/pipeline/10_generate_views.py \
    --reference gradio_demo/example/human/only_lower.jpg \
    --extend --poses front --models 5 \
    --out-dir work/variants-lower
```

Five people, same pose, same garment — `00_front_straight_black_slim_…` and so
on, the slug naming which. `--list-models` prints the table, `--models all`
takes all eight, `--models blonde_petite,short_curls_tall` picks by name.
Without `--models`, nothing changes from before.

Poses and models multiply, so `--poses front,walking --models 3` is six images.
Start with one pose until you know the garment survives.

**What is real and what is invented.** The garment and the visible part of the
body come from your photo. The face, hair, arms and the rest of the legs do
not — they are generated, and they will differ between seeds and between poses,
because nothing in the input constrains them. That is fine when the model is a
vehicle for showing the garment, and wrong if the person has to be a specific
person.

Check that the garment survived the extension before running stage 30. If Qwen
changed its colour or length, the try-on afterwards is measuring the wrong
garment.

### Check the output before tearing down the env

Rebuilding costs 8 GB and 15 minutes, so judge them now:

- **Did each pose actually change?** Compare `02_three_quarter_left` against
  `00_front`. If not, raise `--guidance`.
- **Text and logo gone?** Generative removal is requested in the prompt but not
  guaranteed.
- **Head and both feet in frame.** Stage 20 cuts at the ankles OpenPose finds;
  a variant cropped at the shin is unusable.
- **Same woman, same outfit?** If identity drifted the comparison means
  nothing.

Four poses — `profile_left`, `profile_right`, `back`, `back_over_shoulder` —
are flagged off-front. They generate fine, but IDM-VTON is trained on
front-facing shots and its warping module has no real supervision for a garment
seen edge-on or from behind, so stage 30 will degrade on them. The script warns
when you select one.

---

## 8. Stage 20 — leg masks

Two ways to segment, and which is better depends on your images. Run both once
on the same batch and compare — each prints its timing in the same format.

### Qwen, staying in the stage 10 environment

```bash
python scripts/pipeline/25_qwen_mask.py --in-dir work/variants-lower
```

No env swap: the mask step follows stage 10 with nothing to rebuild, and the
model is already loaded. The mask is *generated* — edges are plausible rather
than measured, and the model can shift the subject slightly while repainting.

### The IDM-VTON parser

```bash
source scripts/alaya/env.sh
python scripts/pipeline/20_leg_masks.py --in-dir work/variants-lower --overlay
```

A segmentation network trained to label body parts, so its edges follow real
pixels, and it cuts the feet at the ankles OpenPose finds. Loads in seconds,
but needs the IDM-VTON env — on a 49 GB disk, an environment swap.

Writes `work/masks/NN_<pose>.mask.png`: white = leg area, black = everything
else, feet excluded. `--overlay` also writes the mask drawn over the photo,
which is the only practical way to judge the ankle cut.

| flag | effect |
|---|---|
| `--region leg_area` | default: parse labels 5/6/12/13 — the legs as they are |
| `--region inpaint` | the dilated region IDM-VTON would actually repaint |
| `--keep-feet` | do not cut at the ankles |
| `--ankle-offset N` | move the cut line down (+) or up (−) by N pixels |
| `--full-size` | mask at the source resolution rather than the model's |
| `--image <path>` | one image instead of a directory |

Feet are excluded by finding the ankle keypoints (OpenPose 10 and 13) and
cutting there. When neither ankle is detected the mask keeps the full leg
rather than guessing.

---

### When the parser cannot read an image

OpenPose is needed only for the ankle cut, so a photo it cannot read no longer
skips the segmentation — stage 20 parses it anyway and says "no ankle cut".
That covers most crops, since a frame with no head usually has no feet either.

What does fail is the parser itself returning nothing, which happens on crops
it was never trained on. Those images are listed in
`work/masks/_unsegmented.txt`, and there is a fallback in the Qwen env:

```bash
source scripts/alaya/_activate.sh
IDM_VENV=$IDM_ROOT/venv-qwen idm_activate
python scripts/pipeline/25_qwen_mask.py \
    --from-failures work/masks/_unsegmented.txt --keep-raw
```

It asks Qwen-Image-Edit to repaint the photo as a two-colour mask, thresholds
that to binary, and writes `<name>.mask.png` beside stage 20's, at the source
resolution and aspect.

**These masks are generated, not measured.** Stage 20 runs a network trained to
label body parts and its edges follow the pixels. Qwen is an image editor being
asked firmly to paint one region white; its edges are plausible rather than
correct, and it can shift the subject slightly while repainting. The script
writes an overlay for every mask and prints the white coverage — check both.
Where stage 20 produces anything at all, its mask is the better one.

`--target` changes what gets painted white (default: trousers, leggings or bare
legs). `--threshold` tunes the cut between white and black. `--keep-raw` saves
Qwen's unthresholded output as `<name>.qwen.png`.

### Comparing the two

Both report load time apart from per-image time, because that is where they
differ most — a few hundred MB of ONNX against 54 GB off a shared mount. A
single seconds-per-image figure with the load folded in tells you the batch
size, not which method is faster.

```bash
# Qwen env, straight after stage 10
python scripts/pipeline/25_qwen_mask.py --in-dir work/variants-lower \
    --out-dir work/masks-qwen --keep-raw

# IDM env
python scripts/pipeline/20_leg_masks.py --in-dir work/variants-lower \
    --out-dir work/masks-idm --overlay
```

Then open `work/masks-qwen/*.overlay.png` beside `work/masks-idm/*.overlay.png`.
Speed is the easy half of the answer; the overlays are the half that decides
it. Expect the parser to win on edge accuracy wherever it produces anything at
all, and Qwen to win wherever the parser returns nothing.

## 9. Stage 30 — try-on across all views

```bash
python scripts/pipeline/30_tryon_batch.py \
    --in-dir work/variants \
    --garment gradio_demo/example/cloth/yoga_pants.jpg \
    --desc "plain mauve high-waisted leggings" \
    --category lower_body
```

Loads the model once for the whole directory — a per-image `demo_single.py`
call would reload it every time. Views where OpenPose finds no person are
skipped with a message rather than crashing.

Useful flags: `--limit N` to try a few first, `--save-mask` to keep the mask
beside each result, `--steps`, `--seed`, `--guidance-scale`, `--out-dir`.

---

## 9b. Images IDM-VTON cannot take

IDM-VTON runs OpenPose first and needs a head, neck and shoulders to find a
person. A half-body product crop has none, so stage 30 skips it and no amount
of tuning changes that. Qwen has no such requirement, and can remove the
overlay text and swap the garment in one pass:

```bash
source scripts/alaya/_activate.sh
IDM_VENV=$IDM_ROOT/venv-qwen idm_activate

python scripts/pipeline/35_qwen_tryon.py \
    --human gradio_demo/example/human/only_lower.jpg \
    --garment gradio_demo/example/cloth/yoga_pants.jpg \
    --desc "plain mauve high-waisted leggings" \
    --count 3
```

Writes `work/results/<name>.qwen_tryon.png` at the source resolution and
aspect. `--clean-only` removes the text and changes no clothing. `--count N`
gives N seeds, which is worth it here: text removal over fabric is
inconsistent, and picking the best of three costs one extra minute.

**The trade.** IDM-VTON warps the actual pixels of your garment photo onto the
body, so the print, seams and cut survive. Qwen re-imagines the garment from
the reference: the result looks right, but fine detail drifts. Plain garments
survive this; patterned or branded ones do not.

For a garment that must stay faithful, go the other way instead — use stage 10
to outpaint the crop into a full-body model, then run stage 30 on that. You
lose the half-body framing and gain a garment that is actually yours.

## 10. Getting results off the box

From your laptop, not the container:

```bash
scp -r idm-box:/pvc/idm/IDM/work/results ./results
scp -r idm-box:/pvc/idm/IDM/work/masks   ./masks
```

Or just open them in the VS Code explorer — it renders PNG thumbnails, which is
enough to judge a batch.

---

## 11. Shutting down

```
关机  (shutdown)  →  saves the image, keeps the 31 GB. Do this.
释放  (release)   →  destroys everything on the container disk.
```

With no PVC, a release costs the full download. There is no undo.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `bash: /pvc/idm/venv/bin/python: No such file or directory` | stale shell pointing at a deleted env | `hash -r`, then `source scripts/alaya/env.sh` |
| `ls: cannot access '/venv'` | `IDM_ROOT` unset in a new SSH session | `source scripts/alaya/env.sh`, and put it in `.bashrc` |
| `ERROR: /pvc/idm is on the container's root filesystem` | no PVC attached | `IDM_ALLOW_EPHEMERAL=1`, accepting the loss on release |
| `ImportError: libGL.so.1` | `opencv-python` instead of headless | `pip install -r requirements.txt` |
| `ModuleNotFoundError: skimage` | incomplete install | `pip install -r requirements.txt` |
| `cannot import name 'cached_download'` | `huggingface_hub` ≥ 0.26 | `pip install huggingface_hub==0.25.2` |
| numpy errors, `SystemError` on import | numpy 2.x against a 1.x-ABI stack | `pip install -r requirements.txt` |
| gradio `TypeError: unhashable type: 'dict'` | starlette 1.0 dropped the old `TemplateResponse` signature | `pip install 'starlette<1.0'` |
| `IndexError` at `run_openpose.py:51` | OpenPose found no person | use a full-body ~3:4 photo with head and shoulders visible |
| `INVALID_PROTOBUF` on `parsing_atr.onnx` | a `git reset` restored the placeholder stubs | `python scripts/alaya/01_download_checkpoints.py` |
| `module diffusers has no attribute QwenImageEditPlusPipeline` | stage 10 running in the IDM-VTON env | activate `venv-qwen` |
| `torch.xpu` AttributeError | modern diffusers on torch < 2.4 | full venv, not `--overlay` |
| `OSError: No space left on device` | the 49 GB disk | `bash scripts/alaya/reclaim_disk.sh` |
| torch version not found on the index | mirror lag | the bootstrap probes candidates; set `TORCH_VERSION` to pin one |

---

## Command reference

```bash
# setup
bash scripts/alaya/check_storage.sh
bash scripts/alaya/bootstrap_all.sh [--qwen-only|--skip-*|--slim]
python scripts/alaya/01_download_checkpoints.py [--dry-run|--slim|--force]
source scripts/alaya/env.sh
python scripts/alaya/preflight.py

# housekeeping
bash scripts/alaya/sync_from_github.sh [--force] [--assets]
bash scripts/alaya/reclaim_disk.sh [--yes]
python scripts/alaya/prune_hf_cache.py [--force]
python scripts/alaya/find_local_models.py [dir]

# single try-on
python scripts/alaya/demo_single.py --list
python scripts/alaya/demo_single.py --human <img> --garment <img> \
    --desc "..." --category lower_body --save-mask
bash scripts/alaya/02_run_gradio.sh

# pipeline
bash scripts/pipeline/00_setup_qwen_env.sh          # QWEN_VENV=... to relocate
python scripts/pipeline/10_generate_views.py --list-poses [--list-models]
python scripts/pipeline/10_generate_views.py --reference <img> --count 4
python scripts/pipeline/10_generate_views.py --reference <img> --extend \
    --poses front --models 5
python scripts/pipeline/20_leg_masks.py --in-dir <dir> --overlay
python scripts/pipeline/25_qwen_mask.py --in-dir <dir>
python scripts/pipeline/25_qwen_mask.py --from-failures work/masks/_unsegmented.txt
python scripts/pipeline/30_tryon_batch.py --in-dir work/variants \
    --garment <img> --desc "..." --category lower_body
python scripts/pipeline/35_qwen_tryon.py --human <img> --garment <img> --desc "..." 
```

## Environment variables

| variable | default | meaning |
|---|---|---|
| `IDM_ROOT` | `/pvc/idm` | everything except the repo lives here |
| `IDM_USER` | unset | scopes `IDM_ROOT` to `/pvc/users/$IDM_USER/idm` |
| `IDM_ALLOW_EPHEMERAL` | unset | accept that `IDM_ROOT` dies with the container |
| `IDM_MODEL_PATH` | `yisol/IDM-VTON` | local IDM-VTON checkout instead of the Hub |
| `QWEN_MODEL_PATH` | `/root/public/models/Qwen/Qwen-Image-Edit-2511` | |
| `QWEN_VENV` | `$IDM_ROOT/venv-qwen` | put the Qwen env on another mount |
| `HF_ENDPOINT` | `https://hf-mirror.com` | huggingface.co is unreachable |
| `PIP_INDEX_URL` | Tsinghua mirror | |
| `BRANCH` | `claude/alaya-new-cloud-setup-4ode4d` | what `sync_from_github.sh` tracks |
