# Running IDM-VTON on Alaya NeW (九章智算云) from VS Code

Adapted from the platform manual *九章智算云平台使用手册*, with the parts that are
specific to this repository filled in.

The platform gives you a **Workshop**: a GPU pod on an elastic container cluster
that you attach to as a VS Code Remote window through the **Aladdin** extension.
Two ways to get an environment into it:

| | Path A — official image + venv | Path B — custom Docker image |
|---|---|---|
| Effort | ~20 min, all inside the Workshop | hours; local Docker, registry push, kubectl |
| Repeatable | scripted, but rebuilt per Workshop | yes, image is fixed |
| Manual's advice | **recommended** | "特别特别麻烦" (a real pain) |

**Start with Path A.** Path B is documented below for when you need a pinned image.

---

## 0. Before you start

1. **Accounts.** Log in to the platform, then install the **Aladdin** extension
   in VS Code and click *Login Business Account* with the same credentials.
2. **Pick the cluster.** 产品中心 → 弹性容器集群 shows the available GPUs. The
   manual says **use GPU1** — GPU2 is routinely occupied by model training.
3. **Check storage.** 产品中心 → 存储管理 shows your PVCs. You need one, and it
   should be roomy: the `yisol/IDM-VTON` weights are tens of GB before you add
   any dataset. **100 GB+ is a comfortable target.**
4. Note whether your cluster is **共享型 (shared)** or **独享型 (dedicated)** — it
   changes whether you need the kubectl work in Path B.

> **On the credentials in the PDF:** it ships a shared platform username and
> password in plain text. They are deliberately **not** written into this repo or
> any script here — every script takes them from environment variables you set
> locally. Since that PDF has been passed around, rotating the password is worth
> doing.

---

## 1. Why the PVC matters more than anything else here

A Workshop's root filesystem is **ephemeral**. Delete or reschedule the Workshop
and everything outside the PVC mount is gone — including a 30 GB model download.

So every path in this setup points at the PVC:

```
$IDM_ROOT            <- your PVC mount, e.g. /pvc/idm
├── IDM-VTON/        <- this repo, cloned onto the PVC
├── venv/            <- the python environment
├── hf/              <- $HF_HOME, where the model weights cache
├── pipcache/        <- so a rebuild is minutes, not an hour
└── data/            <- VITON-HD / DressCode
```

`scripts/alaya/env.sh` sets all of this. **Source it at the start of every
session** — a fresh Workshop is a fresh container.

---

## 2. Path A — official image + venv (recommended)

### 2.1 Create the Workshop

In VS Code: Aladdin icon → **`+`** next to *Workshop*. Field by field:

| Field | Set it to | Why |
|---|---|---|
| **Name** | anything, e.g. `IDM-VTON` | — |
| **Image** | **Base Image** | you build the env yourself in §2.2 |
| **Framework / Version** | pytorch, any version | irrelevant — the venv installs its own torch 2.0.1 and ignores the image's |
| **Python** | **3.10 if the dropdown offers it** | see below |
| **CUDA** | any | pip torch wheels bundle their own CUDA runtime; only the host driver matters, and it is new enough |
| **Resource** | GPU, 1× is plenty | an H800-80G has far more VRAM than the ~24 GB this needs |
| **Storage** | **must be filled in — see §2.1.1** | the default leaves it empty, which is the trap |

**Python is the one field that genuinely constrains you.** Pick 3.10 if it's
offered. If the only choice is 3.12 (the current Alaya default), that's fine —
the bootstrap script detects it and provisions 3.10 via Miniconda onto the PVC,
costing about 5 extra minutes once. What you cannot do is run the stack *on*
3.12: `torch==2.0.1` publishes no cp312 wheel at all, and neither do
`bitsandbytes==0.39.0` or `onnxruntime==1.16.2`.

### 2.1.1 Storage — provision it before you open this dialog

The Storage row has a type dropdown (`nas-capacity`), a **volume** dropdown, and
a **Container Path** box.

**If the volume dropdown says "no data to select", you have no NAS volume yet.**
Aladdin only lists volumes that already exist; it cannot create one. Go to the
platform web console → 产品中心 → **存储管理**, create a file/NAS volume of
**100 GB+** in the *same cluster* as the GPU you'll use (GPU1 per §0), then
reopen this dialog. If 存储管理 offers no create button, storage has not been
authorised for your account and an admin has to allocate it.

Once the volume exists, fill in all three parts:

- volume: the one you just created;
- capacity: 100 GB+ — the weights alone are tens of GB;
- **Container Path**: e.g. `/pvc`.

A blank Container Path means **no PVC is mounted at all**. Everything then lands
on the container's own disk (50 GB on the standard H800 flavour), which fills
partway through the model download and is wiped when the Workshop goes away.
Then `export IDM_ROOT=/pvc/idm` in §2.2 — a subdirectory of the mount, so the
repo, env and caches stay tidy under one root.

Also worth setting: **Namespace** is pre-assigned on a dedicated cluster; on a
shared cluster see §3.3. GPU count 1 is enough for inference — training
(`train_xl.sh`) assumes 4.

VS Code opens a new window attached to the Workshop.

### 2.2 Set up the environment

In the Workshop terminal:

```bash
df -h                                   # confirm the PVC mount path
export IDM_ROOT=/pvc/idm                # <- your actual mount path

mkdir -p "$IDM_ROOT" && cd "$IDM_ROOT"
# -b matters: these scripts live on the setup branch, not on main.
git clone -b claude/alaya-new-cloud-setup-4ode4d \
    https://github.com/Yassinesr/IDM.git IDM-VTON
cd IDM-VTON

bash scripts/alaya/00_bootstrap_workshop.sh
```

That builds the environment on the PVC — a plain venv if the image already has
python 3.10, otherwise a Miniconda-provisioned 3.10 env (§2.1) — installs
torch 2.0.1+cu118 and `requirements.txt`, and prints the GPUs it can see. It is
idempotent: re-run it after a Workshop rebuild and pip serves most of it from
the PVC cache.

### 2.3 Every session after that

A new Workshop terminal starts with none of this set, so:

```bash
export IDM_ROOT=/pvc/idm
cd "$IDM_ROOT/IDM-VTON"
source scripts/alaya/env.sh
```

### 2.4 Download the checkpoints

The `ckpt/*` files in git are **placeholders** — literally files containing
`put ip adapter ckpt here`. Nothing runs until you replace them.

```bash
python scripts/alaya/01_download_checkpoints.py --dry-run   # measure first
python scripts/alaya/01_download_checkpoints.py             # inference
python scripts/alaya/01_download_checkpoints.py --training  # + IP-Adapter
```

`--dry-run` reads the Hub metadata and prints a per-directory size table without
downloading a byte — worth running once so you know what you are committing to
before it starts.

`--slim` skips any `.bin` that has a `.safetensors` twin (same tensors, two
formats) plus the Flax/TF ports nothing here loads. A `.bin` with no safetensors
sibling is still downloaded, so nothing the pipeline needs goes missing. On a
typical diffusers repo this roughly halves the download; combine with
`--dry-run` to see the exact saving for this repo.

This pulls, via `$HF_ENDPOINT` (defaults to `hf-mirror.com`, because
huggingface.co is not routable from the cluster):

| What | From | To |
|---|---|---|
| unet, unet_encoder, vae, text encoders, image_encoder, tokenizers | `yisol/IDM-VTON` (model) | `$HF_HOME` cache |
| densepose `model_final_162be9.pkl` | `yisol/IDM-VTON` (**space**) | `ckpt/densepose/` |
| humanparsing `parsing_atr.onnx`, `parsing_lip.onnx` | same space | `ckpt/humanparsing/` |
| openpose `body_pose_model.pth` | same space | `ckpt/openpose/ckpts/` |
| `ip-adapter-plus_sdxl_vit-h.bin` + image encoder *(training only)* | `h94/IP-Adapter` | `ckpt/ip_adapter/`, `ckpt/image_encoder/` |

The download is resumable — if it drops, just run it again.

### 2.5 Verify before you burn GPU time

```bash
python scripts/alaya/preflight.py
```

One PASS/FAIL line per check: venv, PVC paths, free disk, GPU and VRAM, the
version pins, and whether each `ckpt/*` file is real or still a placeholder. If
something is wrong, paste the whole output when asking for help — it is meant to
be the only thing needed to diagnose a Workshop.

### 2.6 Run one image, headlessly

Do this **before** the gradio demo. It runs the identical pipeline from the
terminal on the examples already bundled in the repo, so a broken environment
surfaces as a stack trace instead of a blank browser tab — and there is no port
forwarding in the way.

```bash
python scripts/alaya/demo_single.py                 # writes demo_out.png
python scripts/alaya/demo_single.py --list          # 9 people, 16 garments
```

Pick your own pair, and describe the garment — the description goes into the
prompt, so it changes the result:

```bash
python scripts/alaya/demo_single.py \
    --human   "gradio_demo/example/human/00034_00.jpg" \
    --garment "gradio_demo/example/cloth/04469_00.jpg" \
    --desc    "a red short-sleeve t-shirt" \
    --output  "$IDM_ROOT/out/demo.png" --save-mask
```

Useful flags: `--steps` (30 default, 20 is faster and usually fine), `--seed`,
`--category {upper_body,lower_body,dresses}`, `--crop` for phone photos that are
not already 3:4, and `--save-mask` when a result looks wrong — a bad
auto-generated mask is the usual cause.

The **first** run downloads the model, so it takes a while and most of that is
network, not GPU. Later runs reuse the PVC cache. The script prints load time,
generate time and peak GPU memory.

To look at the PNG: it is on the PVC, so the VS Code Explorer in the Workshop
window opens it directly — click the file.

### 2.7 Run the gradio demo

```bash
bash scripts/alaya/02_run_gradio.sh
```

It binds `0.0.0.0:7860`. VS Code's **PORTS** panel forwards it to your laptop
automatically; if it doesn't, *Forward a Port* → `7860` and open the localhost
link.

### 2.8 Batch inference on VITON-HD

```bash
export IDM_DATA_DIR=$IDM_ROOT/data/zalando   # per the README layout
bash scripts/alaya/03_run_inference.sh       # add --paired for the paired setting
```

This exists because the repo's own `inference.sh` hardcodes
`/home/omnious/workspace/yisol/...`, which does not exist on your Workshop.

Training still uses the repo's `train_xl.sh`; edit `--data_dir` in it, and
note it sets `CUDA_VISIBLE_DEVICES=0,1,2,3`.

---

## 3. Path B — custom Docker image

Only if you need a pinned image in the private registry. `Dockerfile` and
`.dockerignore` in the repo root are ready to build; the steps below are the
platform-side plumbing, run **on your laptop**, not in a Workshop.

### 3.1 kubectl

```powershell
# Windows PowerShell, in a new D:\kubectl folder
curl.exe -LO "https://rancher-mirror.rancher.cn/kubectl/v1.32.3/windows-amd64-v1.32.3-kubectl.exe"
Rename-Item -Path "windows-amd64-v1.32.3-kubectl.exe" -NewName "kubectl.exe"
setx PATH "%PATH%;D:\kubectl"
kubectl version --client
```

Download the cluster's kubeconfig (*kubeconfig下载* on the cluster page), then:

```powershell
$env:KUBECONFIG="D:\Downloads\gpu1-config.json"
kubectl cluster-info
```

`$env:` is **per-window** — it does not survive closing PowerShell.

### 3.2 Build and push

Open the registry from 头像 → 访问管理 → 镜像仓库; the username and password
arrive by SMS on creation and can be reset there. The page shows the address,
e.g. `registry.hd-01.alayanew.com:8443/alayanew-<uuid>`.

```bash
docker login registry.hd-01.alayanew.com:8443
docker build -t idm-vton:local-1.0 .
docker tag  idm-vton:local-1.0 registry.hd-01.alayanew.com:8443/alayanew-<uuid>/idm-vton:v1.0
docker push registry.hd-01.alayanew.com:8443/alayanew-<uuid>/idm-vton:v1.0
```

Disk filling up: `docker system df`, then `docker system prune -a` (which forces
base images to be re-pulled on the next build).

### 3.3 Fix `Failed to pull image` (shared clusters only)

Aladdin fails to pull private images until the namespace has its own pull
secret. **Dedicated (独享型) clusters skip this entirely** — namespace and secret
are provisioned when access is granted; go straight to §3.4.

```bash
export ALAYA_NAMESPACE=user-yassine
export ALAYA_REGISTRY=registry.hd-01.alayanew.com:8443
export ALAYA_REGISTRY_USER=... ALAYA_REGISTRY_PASS=...
export KUBECONFIG=/path/to/gpu1-config.json
bash scripts/alaya/k8s_bootstrap.sh
```

It creates your namespace, creates the `regcred` secret **inside it** (secrets
are namespace-scoped), and patches that namespace's `default` ServiceAccount
with `imagePullSecrets`. Every Pod binds to that ServiceAccount, so Workshops
stop hitting `ImagePullBackOff`.

Your own namespace is worth having regardless: independent secrets, mounts and
quotas, and you cannot collide with anyone else in `default`.

<details>
<summary>Equivalent PowerShell (per the manual)</summary>

```powershell
kubectl get ns
kubectl create namespace user-wzk
kubectl create secret docker-registry regcred --docker-server=registry.hd-01.alayanew.com:8443 --docker-username=yourname --docker-password=yourpassword -n user-wzk
kubectl get secrets -n user-wzk
kubectl --% patch serviceaccount default -p "{\"imagePullSecrets\": [{\"name\": \"regcred\"}]}" -n user-wzk
kubectl get serviceaccount default -n user-wzk -o yaml
```

`--%` stops PowerShell from mangling the JSON.
</details>

### 3.4 Point Aladdin at the registry

VS Code → Aladdin → **ENVIRONMENTS** → gear (*Setting Registry*) → widen the
left pane → blue pencil → registry username and password.

Then create the Workshop with **Environment** = your image, **Namespace** = the
one you created, **PVC MOUNTS** = your storage. Continue from §2.2 (skip the
bootstrap script — the image already has the deps; you still need §2.4).

---

## 4. Repo-specific gotchas

These are real and will cost you time otherwise:

- **`diffusers==0.25.0` needs `huggingface_hub<0.26`.** It does
  `from huggingface_hub import cached_download`, removed in 0.26.0, and that
  import is on the `DiffusionPipeline` path — so it fails at import, not at
  download. `requirements.txt` pins `0.25.2`.
- **`numpy` must stay on 1.x.** torch 2.0.1 is not built against the NumPy 2
  ABI. Pinned to `1.26.4`.
- **Install torch before `requirements.txt`.** `basicsr` imports torch inside
  its own `setup.py`. Both bootstrap paths order it correctly.
- **`ckpt/*` in git are placeholders**, not weights (§2.4).
- **Python must be 3.10.** `torch==2.0.1` has no cp311/cp312 wheel, so a 3.12
  base image cannot run this stack directly. `00_bootstrap_workshop.sh`
  provisions 3.10 via Miniconda on the PVC when it finds anything else, and
  refuses to continue if the resulting env is still not 3.10.
- **The bundled detectron2 is compiled for Python 3.9.**
  `gradio_demo/detectron2/_C.cpython-39-x86_64-linux-gnu.so` will not import on
  3.10 — this is *harmless*. The import sits in a `try/except ImportError` in
  `layers/deform_conv.py`, and the DensePose config used here never calls
  deformable convolutions. You do not need to build detectron2.
- **Human parsing runs on CPU by design.** `run_parsing.py` hardcodes
  `providers=['CPUExecutionProvider']`, so plain `onnxruntime` is correct; don't
  swap in `onnxruntime-gpu` expecting a speedup.
- **VRAM**: budget ~24 GB for the default 768×1024 at `--test_batch_size 2`.
  Drop to 1 if you OOM.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to pull image ... check your registry configuration` | no pull secret in the namespace | §3.3, then §3.4 |
| `ImagePullBackOff` | same | §3.3 |
| `ImportError: cannot import name 'cached_download'` | `huggingface_hub>=0.26` | `pip install huggingface_hub==0.25.2` |
| Hub download hangs / times out | huggingface.co is blocked | `export HF_ENDPOINT=https://hf-mirror.com` (env.sh does it) |
| Weights re-download after a Workshop restart | `HF_HOME` not on the PVC | `source scripts/alaya/env.sh` before anything |
| `FileNotFoundError` on a `ckpt/...` path | still the placeholder | `python scripts/alaya/01_download_checkpoints.py` |
| `RuntimeError: Numpy is not available` | NumPy 2 got pulled in | `pip install "numpy==1.26.4"` |
| `torch.cuda.is_available()` is False | Workshop has no GPU attached | check the GPU count in the Workshop settings |
| `No matching distribution found for torch==2.0.1` | env is on python 3.11/3.12 | `rm -rf $IDM_VENV && bash scripts/alaya/00_bootstrap_workshop.sh` |
| `No space left on device` mid-download | Container Path was left blank, so there is no PVC | recreate the Workshop with Storage set (§2.1.1) |
| Storage volume dropdown is empty ("no data to select") | no NAS volume exists yet | create one in 产品中心 → 存储管理 (§2.1.1) |
| Demo unreachable in the browser | gradio on 127.0.0.1 | `export GRADIO_SERVER_NAME=0.0.0.0`, forward 7860 in PORTS |
| `kubectl` works, then stops after reopening PowerShell | `$env:KUBECONFIG` is per-window | re-export it |
| `CUDA out of memory` during generation | 768×1024 is heavy | `--steps 20`, or a Workshop with more VRAM |
| Try-on output looks wrong / garment in the wrong place | bad auto-mask | `--save-mask` and inspect; try `--category`, or `--crop` |
| `FileNotFoundError: ./configs/densepose_...yaml` | run from the wrong directory | `demo_single.py` chdirs for you; for `app.py`, run from the repo root |

---

## 6. File map

| File | Purpose |
|---|---|
| `scripts/alaya/env.sh` | per-session env: PVC paths, caches, HF mirror |
| `scripts/alaya/_activate.sh` | activates either the venv or the conda fallback |
| `scripts/alaya/00_bootstrap_workshop.sh` | build the venv on the PVC (Path A) |
| `scripts/alaya/01_download_checkpoints.py` | fetch all weights, mirror-aware |
| `scripts/alaya/preflight.py` | PASS/FAIL environment check; paste its output when stuck |
| `scripts/alaya/demo_single.py` | headless one-image try-on, no browser needed |
| `scripts/alaya/02_run_gradio.sh` | launch the demo with port forwarding |
| `scripts/alaya/03_run_inference.sh` | VITON-HD inference with PVC paths |
| `scripts/alaya/k8s_bootstrap.sh` | namespace + pull secret + SA patch (Path B) |
| `Dockerfile`, `.dockerignore` | custom image (Path B) |
| `requirements.txt` | pip deps with the pins that matter |
