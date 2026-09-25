#!/usr/bin/env bash
# Every stage, one command, one environment.
#
#   bash scripts/pipeline/run_all.sh \
#       --reference gradio_demo/example/human/model_front.jpg \
#       --garment  gradio_demo/example/cloth/yoga_pants.jpg \
#       --desc "plain mauve high-waisted leggings"
#
# Generates one pose on several models, masks them both ways so the two can be
# compared on the same images, and puts the garment on each. Requires the
# single-environment build (bootstrap_all.sh --single-torch); with two separate
# venvs the Qwen stages and the IDM stages cannot run in one process, and this
# says so rather than failing halfway.
#
# Stages are independent deliverables, so a failure in one is reported and the
# rest still run. The summary at the end says what actually got produced.

set -uo pipefail

REFERENCE="gradio_demo/example/human/model_front.jpg"
GARMENT="gradio_demo/example/cloth/yoga_pants.jpg"
DESC="plain mauve high-waisted leggings"
POSE="front"
MODELS="5"
OUT="work/run"
CATEGORY="lower_body"
EXTEND=""
SKIP_TRYON=0
SKIP_GEN=0

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

  --reference PATH   seed photo            (default: $REFERENCE)
  --garment PATH     garment photo         (default: $GARMENT)
  --desc TEXT        garment in words      (default: "$DESC")
  --pose NAME        one pose              (default: $POSE; --list-poses on
                                            10_generate_views.py shows them)
  --models N|NAMES   how many people       (default: $MODELS)
  --out DIR          output root           (default: $OUT)
  --category NAME    lower_body|upper_body|dresses  (default: $CATEGORY)
  --extend           the reference is a partial crop, not a whole person
  --skip-gen         reuse \$OUT/variants from a previous run
  --skip-tryon       masks only
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --reference) REFERENCE="$2"; shift 2 ;;
        --garment)   GARMENT="$2";   shift 2 ;;
        --desc)      DESC="$2";      shift 2 ;;
        --pose)      POSE="$2";      shift 2 ;;
        --models)    MODELS="$2";    shift 2 ;;
        --out)       OUT="$2";       shift 2 ;;
        --category)  CATEGORY="$2";  shift 2 ;;
        --extend)    EXTEND="--extend"; shift ;;
        --skip-gen)  SKIP_GEN=1;     shift ;;
        --skip-tryon) SKIP_TRYON=1;  shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source scripts/alaya/env.sh

# env.sh reports a missing environment but still returns 0, so the shell is
# left on the system python and the first stage would fail on `import torch`
# several minutes in. Check here instead.
if ! python -c 'import torch' 2>/dev/null; then
    echo "ERROR: no usable environment - python has no torch." >&2
    echo "       IDM_ALLOW_EPHEMERAL=1 bash scripts/alaya/bootstrap_all.sh --single-torch" >&2
    exit 1
fi

if [ -z "${QWEN_OVERLAY:-}" ]; then
    echo "ERROR: QWEN_OVERLAY is not set, so the Qwen stages would import the" >&2
    echo "       pinned diffusers 0.25.0 and fail on QwenImageEditPlusPipeline." >&2
    echo >&2
    echo "       This script needs the single-environment build:" >&2
    echo "         IDM_ALLOW_EPHEMERAL=1 bash scripts/alaya/bootstrap_all.sh --single-torch" >&2
    echo >&2
    echo "       With two separate venvs, run the stages by hand instead -" >&2
    echo "       10 and 25 in venv-qwen, 20 and 30 in venv. See docs/runbook.md." >&2
    exit 1
fi

for f in "$REFERENCE" "$GARMENT"; do
    [ -f "$f" ] || { echo "ERROR: $f not found" >&2; exit 1; }
done

VARIANTS="$OUT/variants"
STATUS=""
note() { STATUS="$STATUS
  $1"; }
step() { echo; echo "════════ $* ════════"; }

echo "reference $REFERENCE"
echo "garment   $GARMENT  ($CATEGORY)"
echo "pose      $POSE on $MODELS model(s)"
echo "out       $OUT/"

# ---------------------------------------------------------------- stage 10 ---
if [ "$SKIP_GEN" = "0" ]; then
    step "stage 10  views"
    if python scripts/pipeline/10_generate_views.py \
            --reference "$REFERENCE" --poses "$POSE" --models "$MODELS" \
            --out-dir "$VARIANTS" $EXTEND; then
        note "variants  $VARIANTS"
    else
        echo "stage 10 failed - nothing downstream can run." >&2
        exit 1
    fi
else
    echo; echo "stage 10 skipped, reusing $VARIANTS"
fi
[ -d "$VARIANTS" ] || { echo "ERROR: no $VARIANTS" >&2; exit 1; }

# ------------------------------------------------------- stage 25 then 20 ----
# Qwen first: it is the one that needs no environment change, and running it
# before the parser means a failure there does not cost the comparison.
step "stage 25  masks via Qwen"
if python scripts/pipeline/25_qwen_mask.py \
        --in-dir "$VARIANTS" --out-dir "$OUT/masks-qwen" --keep-raw; then
    note "masks     $OUT/masks-qwen   (Qwen, generated)"
else
    note "masks     Qwen masking FAILED"
fi

step "stage 20  masks via the IDM-VTON parser"
if python scripts/pipeline/20_leg_masks.py \
        --in-dir "$VARIANTS" --out-dir "$OUT/masks-idm" --overlay; then
    note "masks     $OUT/masks-idm    (parser, measured)"
else
    note "masks     parser masking FAILED"
fi

# ---------------------------------------------------------------- stage 30 ---
if [ "$SKIP_TRYON" = "0" ]; then
    step "stage 30  try-on"
    if python scripts/pipeline/30_tryon_batch.py \
            --in-dir "$VARIANTS" --out-dir "$OUT/results" \
            --garment "$GARMENT" --desc "$DESC" --category "$CATEGORY"; then
        note "results   $OUT/results"
    else
        note "results   try-on FAILED"
    fi
fi

# ----------------------------------------------------------------- summary ---
cat <<EOF

════════ done ════════
$STATUS

Compare the two maskers on the same images:
  $OUT/masks-qwen/*.overlay.png
  $OUT/masks-idm/*.overlay.png
Each stage printed its own load and per-image timing above. Load is separated
because the two differ far more in what they cost to start than per image, so
at this batch size the wall clock is mostly the load.

$(df -BG --output=avail "$IDM_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9') GB free on \$IDM_ROOT
EOF
