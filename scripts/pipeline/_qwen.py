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
                  '         QWEN_OVERLAY_PKGS="diffusers>=0.35 transformers>=4.51 \\\n'
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
    print("loading (54 GB off the shared mount; first load is slow)")
    pipe = DiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    print(f"    pipeline: {type(pipe).__name__}")

    if offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe
