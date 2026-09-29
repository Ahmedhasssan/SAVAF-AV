#!/bin/bash
# Stage-2 audio-visual training for a single scene.
#
# Usage:
#   bash scripts/train_av.sh           # trains scene 1 on the GPU set by HIP_VISIBLE_DEVICES (default 0)
#   bash scripts/train_av.sh 5         # trains scene 5
#   HIP_VISIBLE_DEVICES=3 bash scripts/train_av.sh 5   # trains scene 5 on GPU 3
#   AV_RESOLUTION="64 180" bash scripts/train_av.sh 6
#
# AV_RESOLUTION controls MixDiffWithCrossAttention feature-map size (H W).
# Default 64 180 (~5.8 MB). Use 85 240 (~10 MB) to stay near the upper target.
# Audio checkpoints are resolution-specific — retrain stage 2 after changing this.

set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"

SCENE="${1:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"

DATA_DIR="${DATA_DIR:-/workspace/data/release}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/SAVAF-AV/output}"

read -r AV_RES_H AV_RES_W <<< "${AV_RESOLUTION:-64 180}"

# Unique network_gui port per scene so parallel runs don't collide on bind(6099).
NETWORK_GUI_PORT=$((6099 + SCENE))

echo "==> Audio-visual training for scene ${SCENE} on GPU(s) ${HIP_VISIBLE_DEVICES}"
echo "    Feature-map resolution: ${AV_RES_H}x${AV_RES_W} (gui port ${NETWORK_GUI_PORT})"
python train_av.py -s "${DATA_DIR}/${SCENE}" \
    -m "${OUTPUT_DIR}/${SCENE}" \
    --eval \
    --start_checkpoint "${OUTPUT_DIR}/${SCENE}" \
    --iterations 10000 \
    --checkpoint_iterations 2000 \
    --checkpoint_path "${OUTPUT_DIR}/${SCENE}" \
    --av-resolution "${AV_RES_H}" "${AV_RES_W}" \
    --port "${NETWORK_GUI_PORT}"
