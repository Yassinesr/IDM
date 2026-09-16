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

# Deleting $IDM_VENV does not undo a previous activation in the same shell: its
# bin/ stays on PATH, bash's command hash still maps `python` to a file that no
# longer exists, and VIRTUAL_ENV still advertises it. The result is a confusing
# "No such file or directory" for a python that is plainly on PATH. Call this
# when activation fails, to leave the shell in a clean state.
idm_clean_stale() {
    case ":$PATH:" in
        *":$IDM_VENV/bin:"*)
            PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
                    | grep -vxF "$IDM_VENV/bin" | paste -sd: -)"
            export PATH
            echo "[env.sh] removed the deleted $IDM_VENV/bin from PATH" >&2
            ;;
    esac
    [ -n "${VIRTUAL_ENV:-}" ] && [ ! -d "${VIRTUAL_ENV}" ] && unset VIRTUAL_ENV
    [ -n "${CONDA_PREFIX:-}" ] && [ ! -d "${CONDA_PREFIX}" ] && unset CONDA_PREFIX
    hash -r 2>/dev/null || true
    return 0
}
