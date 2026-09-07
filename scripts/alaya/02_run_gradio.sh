#!/usr/bin/env bash
# Launch the gradio try-on demo inside the Workshop.
#
#   source scripts/alaya/env.sh
#   bash scripts/alaya/02_run_gradio.sh
#
# The Workshop is a VS Code Remote window, so once gradio prints
# "Running on http://0.0.0.0:7860", VS Code's PORTS panel forwards 7860 to your
# laptop automatically. If it does not, add it by hand: PORTS -> Forward a Port
# -> 7860, then open the localhost link.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[ -n "${VIRTUAL_ENV:-}" ] || { echo "Run 'source scripts/alaya/env.sh' first." >&2; exit 1; }

for f in ckpt/densepose/model_final_162be9.pkl \
         ckpt/humanparsing/parsing_atr.onnx \
         ckpt/humanparsing/parsing_lip.onnx \
         ckpt/openpose/ckpts/body_pose_model.pth; do
    if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -lt 4096 ]; then
        echo "ERROR: $f is still the committed placeholder." >&2
        echo "       Run: python scripts/alaya/01_download_checkpoints.py" >&2
        exit 1
    fi
done

echo "Serving on ${GRADIO_SERVER_NAME:-127.0.0.1}:${GRADIO_SERVER_PORT:-7860}"
exec python gradio_demo/app.py
