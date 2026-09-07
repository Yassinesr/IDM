#!/usr/bin/env bash
# Batch inference on VITON-HD, with paths pointed at the PVC instead of the
# authors' /home/omnious/... hardcodes in inference.sh.
#
#   source scripts/alaya/env.sh
#   bash scripts/alaya/03_run_inference.sh [--paired] [extra args...]
#
# Expects the dataset laid out per the README under $IDM_DATA_DIR:
#   test/{image,image-densepose,agnostic-mask,cloth,vitonhd_test_tagged.json}

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -n "${VIRTUAL_ENV:-}" ] || { echo "Run 'source scripts/alaya/env.sh' first." >&2; exit 1; }

DATA_DIR="${IDM_DATA_DIR:-${IDM_ROOT:-/pvc/idm}/data/zalando}"
OUT_DIR="${IDM_OUTPUT_DIR:-${IDM_ROOT:-/pvc/idm}/result}"

[ -d "$DATA_DIR" ] || { echo "ERROR: dataset not found at $DATA_DIR. Set IDM_DATA_DIR." >&2; exit 1; }

PAIRING=(--unpaired)
if [ "${1:-}" = "--paired" ]; then PAIRING=(); shift; fi

mkdir -p "$OUT_DIR"
echo "data=$DATA_DIR  out=$OUT_DIR  ${PAIRING[*]:-paired}"

exec accelerate launch inference.py \
    --pretrained_model_name_or_path "yisol/IDM-VTON" \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" \
    "${PAIRING[@]}" \
    --seed 42 --test_batch_size 2 --guidance_scale 2.0 \
    "$@"
