# Sourced by env.sh and 00_bootstrap_workshop.sh - not meant to be run directly.
#
# $IDM_VENV is one of two things depending on what the base image offered:
#   - a plain venv, when the image already had python 3.10 (has bin/activate)
#   - a conda env at a prefix, when we had to provision 3.10 ourselves
#     (has conda-meta but NO bin/activate; `conda activate` needs a shell hook
#     that is not initialised in a fresh Workshop terminal)
# Putting the env's bin/ first on PATH is enough for both cases.
idm_activate() {
    if [ -f "$IDM_VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        . "$IDM_VENV/bin/activate"
        return 0
    fi
    if [ -d "$IDM_VENV/conda-meta" ]; then
        CONDA_PREFIX="$IDM_VENV"; export CONDA_PREFIX
        PATH="$IDM_VENV/bin:$PATH"; export PATH
        return 0
    fi
    return 1
}
