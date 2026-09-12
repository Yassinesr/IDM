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
    # The bootstrap builds a venv when the image has python 3.10, and a conda
    # env at a prefix when it had to provision 3.10 itself.
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        ok(f"env active: {venv}")
    else:
        warn("neither VIRTUAL_ENV nor CONDA_PREFIX set - "
             "did you `source scripts/alaya/env.sh`?")

    pyver = "%d.%d" % sys.version_info[:2]
    if pyver == "3.10":
        ok(f"python {pyver}")
    else:
        # torch is no longer the binding constraint (the bootstrap falls back to
        # a newer version when needed); onnxruntime 1.16.2 publishes no cp311/
        # cp312 wheel, and transformers 4.36.2 predates 3.12 support.
        fail(f"python {pyver} - this stack targets 3.10 "
             "(onnxruntime 1.16.2 and transformers 4.36.2)",
             "recreate the env: rm -rf $IDM_VENV && "
             "bash scripts/alaya/00_bootstrap_workshop.sh")

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
    if idm_root and Path(idm_root).is_dir():
        try:
            # Same device as / means this is the container's own ephemeral disk,
            # not a PVC - a plain mkdir looks identical until the Workshop is
            # released and the weights go with it.
            if os.stat(idm_root).st_dev == os.stat("/").st_dev:
                if os.environ.get("IDM_ALLOW_EPHEMERAL") == "1":
                    warn(f"{idm_root} is on the ephemeral container disk "
                         "(IDM_ALLOW_EPHEMERAL=1) - everything here is lost when "
                         "the Workshop is released")
                else:
                    fail(f"{idm_root} is on the container root filesystem, not a PVC",
                         "recreate the Workshop with Storage > Container Path set, "
                         "or set IDM_ALLOW_EPHEMERAL=1 to accept the loss")
            else:
                ok(f"{idm_root} is a real mount, separate from /")
        except OSError as exc:
            warn(f"could not stat {idm_root}: {exc}")

    # Alaya mounts a read-only library of common models/datasets here. Worth a
    # look before downloading tens of GB over a slow link.
    public = Path("/root/public")
    if public.is_dir():
        ok(f"{public} exists (read-only shared models/datasets) - "
           "check it before downloading")

    # How much free space is actually needed depends on whether the weights are
    # already here. Demanding room for a download that has already happened is
    # how a working setup gets reported as broken.
    model_cached = 0
    if hf_home:
        folder = Path(hf_home) / "hub" / "models--yisol--IDM-VTON"
        if folder.is_dir():
            for f in folder.rglob("*"):
                try:
                    if f.is_file() and not f.is_symlink():
                        model_cached += f.stat().st_size
                except OSError:
                    pass
    need_gb = 5 if model_cached > 10 * 2**30 else 60

    for label, path in [("IDM_ROOT", idm_root), ("/tmp", "/tmp")]:
        if not path:
            continue
        try:
            free = shutil.disk_usage(path).free / 2**30
        except OSError as exc:
            warn(f"{label}: cannot stat ({exc})")
            continue
        if label == "IDM_ROOT" and free < need_gb:
            why = ("outputs and temporary files need room" if need_gb == 5
                   else "the model still has to be downloaded")
            fail(f"{label} ({path}): {free:.0f} GiB free, want {need_gb} - {why}",
                 "clear space, or grow the PVC")
        elif label == "IDM_ROOT" and model_cached:
            ok(f"{label} ({path}): {free:.0f} GiB free "
               f"({model_cached / 2**30:.0f} GiB of it already the cached model)")
        else:
            ok(f"{label} ({path}): {free:.0f} GiB free")

    # --------------------------------------------------------------- gpu ---
    section("gpu / torch")
    # Never bail out early here: on a shared Workshop the sharing checks below
    # are exactly what you want to see even before the env is built.
    try:
        import torch
    except ImportError:
        torch = None
        fail("torch not installed", "bash scripts/alaya/00_bootstrap_workshop.sh")

    if torch is not None:
        ok(f"torch {torch.__version__}")
        if not torch.cuda.is_available():
            fail("torch.cuda.is_available() is False - no GPU attached to this Workshop",
                 "check the GPU count in the Workshop settings, then recreate it")
        else:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                ok(f"gpu {i}: {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
                if p.major < 8 and torch.__version__.startswith("2."):
                    warn(f"gpu {i} is sm_{p.major}{p.minor}; make sure the wheel has kernels for it")
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
        except Exception as exc:
            # A module compiled against NumPy 1.x raises SystemError, not
            # ImportError, under NumPy 2 - report it rather than printing a
            # traceback and then declaring the module fine.
            fail(f"{mod} is installed but fails to import: "
                 f"{type(exc).__name__}: {str(exc)[:120]}",
                 "pip install -r requirements.txt")
            continue
        got = getattr(m, "__version__", "?")
        if got.startswith(want):
            ok(f"{mod} {got} (expected {want}*)")
        elif mod == "numpy":
            # Not a nit: torch, scipy and onnxruntime here are all built against
            # the NumPy 1.x ABI and fail outright on 2.x. Easy to reintroduce,
            # since installing almost anything can pull numpy forward.
            fail(f"numpy {got} - this stack is built against the 1.x ABI",
                 'pip install "numpy==1.26.4"')
        else:
            warn(f"{mod} {got} (expected {want}*)")

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

    # ---------------------------------------------------- sharing / GPU ----
    section("shared use")
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        apps = [l for l in out.stdout.strip().splitlines() if l.strip()]
        if not apps:
            ok("no other processes are using the GPU")
        else:
            mine = str(os.getpid())
            for line in apps:
                pid = line.split(",")[0].strip()
                label = "this process" if pid == mine else "SOMEONE ELSE"
                warn(f"GPU in use by pid {pid} ({label}): {line}")
            warn("starting a run now competes for VRAM - check before you launch")
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        warn(f"could not query GPU processes ({exc})")

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        for line in out.stdout.strip().splitlines():
            ok(f"gpu memory {line}")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    if os.environ.get("IDM_USER"):
        ok(f"IDM_USER={os.environ['IDM_USER']} - your paths are scoped to you")
    elif Path("/pvc/users").is_dir():
        warn("/pvc/users exists (shared PVC) but IDM_USER is unset - "
             "you are writing into the common root")
    else:
        ok("IDM_USER unset (fine if this PVC is yours alone)")

    # A pip.conf in the shared home changes pip's index for every user of it.
    for cfg in (Path.home() / ".config/pip/pip.conf", Path.home() / ".pip/pip.conf"):
        if cfg.exists():
            warn(f"{cfg} exists and affects everyone sharing this home directory")

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
