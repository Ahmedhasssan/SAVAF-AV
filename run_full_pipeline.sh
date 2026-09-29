#!/bin/bash
# End-to-end RWAVS pipeline:
#   Stage 1 — visual 3DGS (train.py) per scene
#   Stage 2 — audio-visual training (train_av_parallel.sh)
#   Stage 3 — checkpoint sweep + MAG/ENV summary (eval_checkpoints.sh)
#
# Usage examples:
#   bash run_full_pipeline.sh
#   SCENES="1 2 3 4 5" N_GPUS=2 bash run_full_pipeline.sh
#   SKIP_STAGE1=1 bash run_full_pipeline.sh          # gaussians already exist
#   SKIP_STAGE2=1 bash run_full_pipeline.sh          # eval checkpoints only
#   SKIP_EVAL=1 bash run_full_pipeline.sh            # train both stages, skip summary
#   SKIP_EXISTING=1 bash run_full_pipeline.sh        # skip scenes with final ckpts
#   AV_RESOLUTION="64 180" N_GPUS=2 bash run_full_pipeline.sh
#
# Environment variables:
#   SCENES          Space-separated scene ids (default: 1..13)
#   N_GPUS          GPUs for stage-2 parallelism (default: 2)
#   AV_RESOLUTION   Feature-map H W for stage 2/eval (default: 64 180)
#   GPU_IDS         Optional explicit GPU list for stage 2, e.g. "0,1"
#   STAGE1_GPU      GPU used for sequential stage-1 runs (default: 0)
#   EVAL_GPU        GPU used for stage-3 eval sweep (default: 0)
#   DATA_DIR        Dataset root (default: /workspace/data/release)
#   OUTPUT_DIR      Checkpoint root (default: /workspace/SAVAF-AV/output)
#   LOG_DIR_STAGE1  Stage-1 logs (default: logs/stage1)
#   LOG_DIR_STAGE2  Stage-2 logs (default: logs/stage2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

SCENES="${SCENES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
REQUESTED_SCENES="${SCENES}"
N_GPUS="${N_GPUS:-2}"
GPU_IDS="${GPU_IDS:-}"
STAGE1_GPU="${STAGE1_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"
DATA_DIR="${DATA_DIR:-/workspace/data/release}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/SAVAF-AV/output}"
LOG_DIR_STAGE1="${LOG_DIR_STAGE1:-${SCRIPT_DIR}/logs/stage1}"
LOG_DIR_STAGE2="${LOG_DIR_STAGE2:-${SCRIPT_DIR}/logs/stage2}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

STAGE1_ITERATIONS="${STAGE1_ITERATIONS:-30010}"
STAGE2_ITERATIONS="${STAGE2_ITERATIONS:-10000}"
read -r AV_RES_H AV_RES_W <<< "${AV_RESOLUTION:-64 180}"
export AV_RESOLUTION="${AV_RES_H} ${AV_RES_W}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR_STAGE1}" "${LOG_DIR_STAGE2}"

echo "============================================================"
echo " SAVAF-AV full pipeline"
echo "============================================================"
echo " Scenes      : ${SCENES}"
echo " Stage-1 GPU : ${STAGE1_GPU}  (sequential, ${STAGE1_ITERATIONS} iters/scene)"
echo " Stage-2 GPUs: ${N_GPUS}${GPU_IDS:+ (${GPU_IDS})}  (${STAGE2_ITERATIONS} iters/scene)"
echo " AV resolution: ${AV_RES_H}x${AV_RES_W}"
echo " Eval GPU    : ${EVAL_GPU}"
echo " Data        : ${DATA_DIR}"
echo " Output      : ${OUTPUT_DIR}"
echo " Skip        : stage1=${SKIP_STAGE1} stage2=${SKIP_STAGE2} eval=${SKIP_EVAL} existing=${SKIP_EXISTING}"
echo "============================================================"
echo ""

stage1_fail=0
stage2_fail=0

run_stage1() {
    echo "==> Stage 1: Visual 3DGS (train.py)"
    for scene in ${SCENES}; do
        scene_dir="${OUTPUT_DIR}/${scene}"
        ckpt="${scene_dir}/chkpnt${STAGE1_ITERATIONS}.pth"
        log="${LOG_DIR_STAGE1}/scene_${scene}.log"

        if [[ "${SKIP_EXISTING}" == "1" && -f "${ckpt}" ]]; then
            echo "[skip] scene=${scene} stage1 — found ${ckpt}"
            continue
        fi

        mkdir -p "${scene_dir}"
        echo "[launch] scene=${scene} gpu=${STAGE1_GPU} log=${log}"
        if ! HIP_VISIBLE_DEVICES="${STAGE1_GPU}" CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" \
            python train.py \
                -s "${DATA_DIR}/${scene}" \
                -m "${scene_dir}" \
                --eval \
                --iterations "${STAGE1_ITERATIONS}" \
                --checkpoint_iterations "${STAGE1_ITERATIONS}" \
                --checkpoint_path "${scene_dir}" \
            > "${log}" 2>&1; then
            echo "[fail]   scene=${scene} stage1 — see ${log}"
            stage1_fail=$((stage1_fail + 1))
        else
            echo "[done]   scene=${scene} stage1"
        fi
    done
    echo ""
    if [[ ${stage1_fail} -gt 0 ]]; then
        echo "==> Stage 1 finished with ${stage1_fail} failure(s)."
        return 1
    fi
    echo "==> Stage 1 completed successfully."
}

run_stage2() {
    echo "==> Stage 2: Audio-visual training (train_av_parallel.sh)"
    export SCENES N_GPUS GPU_IDS LOG_DIR="${LOG_DIR_STAGE2}" DATA_DIR OUTPUT_DIR AV_RESOLUTION

    if [[ "${SKIP_EXISTING}" == "1" ]]; then
        filtered_scenes=""
        for scene in ${SCENES}; do
            av_ckpt="${OUTPUT_DIR}/${scene}/audio_chkpnt${STAGE2_ITERATIONS}.pth"
            if [[ -f "${av_ckpt}" ]]; then
                echo "[skip] scene=${scene} stage2 — found ${av_ckpt}"
            else
                filtered_scenes="${filtered_scenes} ${scene}"
            fi
        done
        SCENES="${filtered_scenes# }"
        export SCENES
        if [[ -z "${SCENES// /}" ]]; then
            echo "==> Stage 2: all scenes already have audio checkpoints, skipping."
            echo ""
            return 0
        fi
        echo "    Remaining scenes: ${SCENES}"
    fi

    if ! bash "${SCRIPT_DIR}/train_av_parallel.sh"; then
        echo "==> Stage 2 failed."
        return 1
    fi
    echo ""
}

run_eval() {
    echo "==> Stage 3: Checkpoint sweep + MAG/ENV summary (RWAVS categories)"
    export DATA_DIR OUTPUT_DIR AV_RESOLUTION
    HIP_VISIBLE_DEVICES="${EVAL_GPU}" CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        bash "${SCRIPT_DIR}/eval_checkpoints.sh" ${REQUESTED_SCENES}
    echo ""
}

if [[ "${SKIP_STAGE1}" != "1" ]]; then
    run_stage1
else
    echo "==> Stage 1 skipped (SKIP_STAGE1=1)"
    echo ""
fi

if [[ "${SKIP_STAGE2}" != "1" ]]; then
    run_stage2
else
    echo "==> Stage 2 skipped (SKIP_STAGE2=1)"
    echo ""
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
    run_eval
else
    echo "==> Stage 3 skipped (SKIP_EVAL=1)"
fi

echo "============================================================"
echo " Pipeline complete."
echo "  Stage-1 logs : ${LOG_DIR_STAGE1}/scene_<N>.log"
echo "  Stage-2 logs : ${LOG_DIR_STAGE2}/scene_<N>.log"
echo "============================================================"
