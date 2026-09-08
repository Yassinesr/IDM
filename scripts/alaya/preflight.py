#!/usr/bin/env python
"""Check that a Workshop is actually ready to run IDM-VTON.

    source scripts/alaya/env.sh
    python scripts/alaya/preflight.py

Prints a PASS/FAIL line per check. Paste the whole output when asking for help -
it is designed to be the only thing anyone needs to diagnose a broken Workshop.
"""

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAILURES = []
WARNINGS = []


def ok(msg):
    print(f"  [ok]   {msg}")


def warn(msg):
    print(f"  [warn] {msg}")
    WARNINGS.append(msg)


def fail(msg, fix=""):
    print(f"  [FAIL] {msg}")
    if fix:
        print(f"         fix: {fix}")
    FAILURES.append(msg)


def section(name):
    print(f"\n== {name}")


def main():
    print(f"IDM-VTON preflight  (repo: {REPO})")

    # ---------------------------------------------------------------- env --
    section("environment")
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        ok(f"venv active: {venv}")
    else:
        warn("no VIRTUAL_ENV - did you `source scripts/alaya/env.sh`?")

    idm_root = os.environ.get("IDM_ROOT")
    if idm_root and Path(idm_root).is_dir():
        ok(f"IDM_ROOT={idm_root}")
    else:
        fail(f"IDM_ROOT={idm_root!r} is unset or missing",
             "export IDM_ROOT=<your PVC mount>; source scripts/alaya/env.sh")

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        fail("HF_HOME unset - weights would cache to the ephemeral container disk",
             "source scripts/alaya/env.sh")
    elif idm_root and not str(Path(hf_home).resolve()).startswith(str(Path(idm_root).resolve())):
        warn(f"HF_HOME={hf_home} is outside IDM_ROOT - it will not survive a rebuild")
    else:
        ok(f"HF_HOME={hf_home} (on the PVC)")

    ok(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co (default)')}")

    # -------------------------------------------------------------- disk ---
    section("disk")
    for label, path in [("IDM_ROOT", idm_root), ("/tmp", "/tmp")]:
        if not path:
            continue
        try:
            free = shutil.disk_usage(path).free / 2**30
        except OSError as exc:
            warn(f"{label}: cannot stat ({exc})")
            continue
        if label == "IDM_ROOT" and free < 60:
            fail(f"{label} ({path}): {free:.0f} GiB free - the model alone is tens of GB",
                 "grow the PVC or clear $HF_HOME")
        else:
            ok(f"{label} ({path}): {free:.0f} GiB free")

    # --------------------------------------------------------------- gpu ---
    section("gpu / torch")
    try:
        import torch
    except ImportError:
        fail("torch not installed", "bash scripts/alaya/00_bootstrap_workshop.sh")
        return summarize()

    ok(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False - no GPU attached to this Workshop",
             "check the GPU count in the Workshop settings, then recreate it")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            ok(f"gpu {i}: {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
            if p.total_memory / 2**30 < 20:
                warn(f"gpu {i} has < 20 GiB - use --steps 20 and expect OOM at 768x1024")

    # -------------------------------------------------------------- deps ---
    section("dependencies")
    import importlib
    for mod, want in [("diffusers", "0.25.0"), ("transformers", "4.36.2"),
                      ("huggingface_hub", "0.25.2"), ("numpy", "1."),
                      ("onnxruntime", "1.16.2"), ("gradio", "4.24.0")]:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            fail(f"{mod} not installed", "pip install -r requirements.txt")
            continue
        got = getattr(m, "__version__", "?")
        (ok if got.startswith(want) else warn)(f"{mod} {got} (expected {want}*)")

    # The pin that actually bites: diffusers 0.25 imports a symbol hub 0.26 dropped.
    try:
        from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # noqa: F401
        ok("diffusers imports cleanly")
    except ImportError as exc:
        fail(f"diffusers import failed: {exc}",
             "pip install 'huggingface_hub==0.25.2'")

    # ------------------------------------------------------- checkpoints ---
    section("checkpoints")
    required = [
        "ckpt/densepose/model_final_162be9.pkl",
        "ckpt/humanparsing/parsing_atr.onnx",
        "ckpt/humanparsing/parsing_lip.onnx",
        "ckpt/openpose/ckpts/body_pose_model.pth",
    ]
    for rel in required:
        p = REPO / rel
        if not p.exists():
            fail(f"{rel} missing", "python scripts/alaya/01_download_checkpoints.py")
        elif p.stat().st_size < 4096:
            fail(f"{rel} is still the committed placeholder "
                 f"({p.read_text(errors='replace').strip()[:40]!r})",
                 "python scripts/alaya/01_download_checkpoints.py")
        else:
            ok(f"{rel} ({p.stat().st_size / 2**20:.0f} MiB)")

    if hf_home:
        hub = Path(hf_home) / "hub"
        cached = list(hub.glob("models--yisol--IDM-VTON")) if hub.is_dir() else []
        if cached:
            size = sum(f.stat().st_size for f in cached[0].rglob("*") if f.is_file())
            ok(f"yisol/IDM-VTON cached ({size / 2**30:.1f} GiB)")
        else:
            warn("yisol/IDM-VTON not in the cache yet - the first run downloads it")

    # ------------------------------------------------------------ config ---
    section("repo files")
    for rel in ["configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "configs/Base-DensePose-RCNN-FPN.yaml",
                "gradio_demo/example/human", "gradio_demo/example/cloth"]:
        (ok if (REPO / rel).exists() else fail)(rel)

    return summarize()


def summarize():
    print()
    if FAILURES:
        print(f"FAILED - {len(FAILURES)} problem(s), {len(WARNINGS)} warning(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All checks passed ({len(WARNINGS)} warning(s)).")
    print("Next: python scripts/alaya/demo_single.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
