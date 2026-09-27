"""Shared Qwen-Image-Edit plumbing for the stages that use it.

Both stage 10 (generate views) and stage 25 (fallback masks) load the same
model the same way, so the awkward parts live here once: resolving the
pipeline class from model_index.json rather than hard-coding a name that moves
between releases, filtering kwargs against the real signature, and explaining
the two import failures that look like missing packages but are version
floors.
"""

import inspect
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "/root/public/models/Qwen/Qwen-Image-Edit-2511"


def resolve_model(path):
    """Validate a local diffusers checkout, or explain what is wrong with it."""
    model_dir = Path(path)
    if not model_dir.is_dir():
        print(f"ERROR: model not found at {model_dir}", file=sys.stderr)
        print("       Check: ls /root/public/models/Qwen/", file=sys.stderr)
        return None
    if not (model_dir / "model_index.json").is_file():
        print(f"ERROR: {model_dir} has no model_index.json, so it is not a\n"
              "       diffusers-format checkout. Look for a subdirectory that "
              "has one.", file=sys.stderr)
        return None
    return model_dir


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


def torch_version():
    import torch
    return tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])


def install_torch_compat():
    """Close the one gap between diffusers >= 0.36 and torch 2.4.

    diffusers' native attention backend passes enable_gqa= to
    scaled_dot_product_attention, which torch added in 2.5. On 2.4 that is a
    TypeError deep in the first denoising step, long after the 54 GB load.

    enable_gqa=False is exactly what torch 2.4 does anyway, so dropping it is
    not a behaviour change. enable_gqa=True would be, so that raises instead of
    being silently ignored.
    """
    import torch
    import torch.nn.functional as F
    if torch_version() >= (2, 5):
        return
    original = F.scaled_dot_product_attention
    if getattr(original, "_gqa_shim", False):
        return

    def sdpa(*args, enable_gqa=False, **kwargs):
        if enable_gqa:
            raise NotImplementedError(
                f"enable_gqa=True needs torch >= 2.5; this env has "
                f"{torch.__version__}. Rebuild the base env on a newer torch: "
                "TORCH_VERSION=2.5.1 bash scripts/alaya/00_bootstrap_workshop.sh")
        return original(*args, **kwargs)

    sdpa._gqa_shim = True
    F.scaled_dot_product_attention = sdpa
    print(f"          (scaled_dot_product_attention shimmed for torch "
          f"{torch.__version__}: no enable_gqa)")


def import_diffusers():
    """Import diffusers, honouring QWEN_OVERLAY, or explain the version floor.

    Returns (diffusers, DiffusionPipeline) or None after printing why not.
    """
    # An overlay is a directory of just the NEW libraries (diffusers,
    # transformers), installed with `pip install --target`. Put first on
    # sys.path it shadows the IDM-VTON env's pinned versions while reusing its
    # torch - a few hundred MB instead of a second ~8 GB torch stack.
    overlay = os.environ.get("QWEN_OVERLAY")
    if overlay:
        if not Path(overlay).is_dir():
            print(f"ERROR: QWEN_OVERLAY={overlay} is not a directory",
                  file=sys.stderr)
            return None
        sys.path.insert(0, overlay)
        print(f"overlay   {overlay} (shadowing the env's diffusers/transformers)")

    # Before diffusers, so the patched function is the one its modules resolve.
    install_torch_compat()

    try:
        import diffusers
        from diffusers import DiffusionPipeline
    except (ImportError, AttributeError) as exc:
        # AttributeError too: modern diffusers touches torch.xpu at import,
        # which torch < 2.4 does not have, and that is not an ImportError.
        if not overlay:
            raise
        print(f"\nERROR: {exc}", file=sys.stderr)
        if "xpu" in str(exc):
            print("\n       This is the torch floor, not a missing package.\n"
                  "       QwenImageEditPlusPipeline needs diffusers >= 0.36, and\n"
                  "       diffusers >= 0.34 touches torch.xpu at import, which\n"
                  "       arrived in torch 2.4. The base env has torch 2.0.1, so\n"
                  "       overlay mode cannot work here - it needs a full venv "
                  "with\n       its own newer torch:\n"
                  "         bash scripts/pipeline/00_setup_qwen_env.sh",
                  file=sys.stderr)
        else:
            print("\n       The overlay shadows only the packages it installed; "
                  "this one\n       came from the base env at its older pinned "
                  "version. Rebuild\n       the overlay with it added:\n"
                  '         QWEN_OVERLAY_PKGS="diffusers>=0.36 transformers>=4.51 \\\n'
                  '             tokenizers huggingface_hub>=0.27 safetensors '
                  'accelerate" \\\n'
                  "             bash scripts/pipeline/00_setup_qwen_env.sh --overlay",
                  file=sys.stderr)
        return None
    return diffusers, DiffusionPipeline


def load_pipeline(model_dir, offload=False):
    """Load the model onto the GPU. Returns the pipeline, or None on failure."""
    imported = import_diffusers()
    if imported is None:
        return None
    diffusers, DiffusionPipeline = imported

    import torch
    import huggingface_hub
    print(f"          torch {torch.__version__}, diffusers "
          f"{diffusers.__version__}, hub {huggingface_hub.__version__}")
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.", file=sys.stderr)
        return None

    # from_pretrained on a local dir reads model_index.json and builds whatever
    # pipeline class it names - so this does not hard-code a class that may be
    # renamed between Qwen releases.
    # ~33s measured with the page cache warm from an earlier stage; a cold
    # read off CephFS is slower.
    print("loading (54 GB off the shared mount)")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    print(f"    pipeline: {type(pipe).__name__}")

    if offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe
