#!/bin/bash
# Parallel Stage-2 audio-visual training across scenes, one GPU per scene.
#
# Each scene gets a dedicated GPU via HIP_VISIBLE_DEVICES. If there are more
# scenes than GPUs, scenes queue up automatically: as soon as a GPU is free,
# the next scene starts on it.
#
# Usage examples:
#   bash scripts/train_av_parallel.sh                 # all 13 scenes, 4 GPUs, default settings
#   N_GPUS=8 bash scripts/train_av_parallel.sh        # use 8 GPUs
#   SCENES="1 3 7" N_GPUS=2 bash scripts/train_av_parallel.sh
#   GPU_IDS="2,4,6" bash scripts/train_av_parallel.sh # explicit GPU ids (3 GPUs from those)
#   AV_RESOLUTION="64 180" N_GPUS=2 bash scripts/train_av_parallel.sh
#
# AV_RESOLUTION (default: 64 180) is forwarded to train_av.sh / train_av.py.
# Per-scene logs land in ${LOG_DIR}/scene_<N>.log so they don't interleave.
# Final summary prints exit codes for each scene.

set -u

cd "$(cd "$(dirname "$0")/.." && pwd)"

# --- configurable knobs (override via env vars) ------------------------------
SCENES="${SCENES:-1 2 3 4 5 6 7 8 9 10 11 12 13}"
N_GPUS="${N_GPUS:-4}"
GPU_IDS="${GPU_IDS:-}"  # e.g. "0,2,4,6" to skip specific GPUs; otherwise 0..N_GPUS-1
LOG_DIR="${LOG_DIR:-/workspace/SAVAF-AV/logs/stage2}"

# --- derive GPU id list ------------------------------------------------------
if [[ -z "${GPU_IDS}" ]]; then
    GPU_LIST=()
    for ((g=0; g<N_GPUS; g++)); do GPU_LIST+=("$g"); done
else
    IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
fi
N_SLOTS="${#GPU_LIST[@]}"

mkdir -p "${LOG_DIR}"
echo "==> Stage-2 parallel launcher"
echo "    Scenes : ${SCENES}"
echo "    GPUs   : ${GPU_LIST[*]} (${N_SLOTS} slots)"
echo "    AV res : ${AV_RESOLUTION:-64 180}"
echo "    Logs   : ${LOG_DIR}"
echo ""

# --- launch with a per-GPU job slot pool -------------------------------------
declare -A SLOT_PID         # SLOT_PID[gpu_id] = currently running PID, or empty
declare -A SCENE_BY_SLOT    # SCENE_BY_SLOT[gpu_id] = currently training scene
declare -A SCENE_LOG        # SCENE_LOG[scene] = log file
declare -A SCENE_EXIT       # SCENE_EXIT[scene] = exit code once done

launch_scene() {
    local scene="$1"
    local gpu="$2"
    local log="${LOG_DIR}/scene_${scene}.log"
    SCENE_LOG[$scene]="$log"
    echo "[launch] scene=${scene} gpu=${gpu} log=${log}"
    HIP_VISIBLE_DEVICES="${gpu}" CUDA_VISIBLE_DEVICES="${gpu}" \
        bash "$(dirname "$0")/train_av.sh" "${scene}" \
        > "${log}" 2>&1 &
    SLOT_PID[$gpu]=$!
    SCENE_BY_SLOT[$gpu]=$scene
}

find_free_slot() {
    for g in "${GPU_LIST[@]}"; do
        local pid="${SLOT_PID[$g]:-}"
        if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            # GPU is free; harvest exit code if a job just finished
            if [[ -n "${pid}" ]]; then
                wait "${pid}" 2>/dev/null
                local rc=$?
                local s="${SCENE_BY_SLOT[$g]}"
                SCENE_EXIT[$s]=$rc
                echo "[done]   scene=${s} gpu=${g} rc=${rc}"
                SLOT_PID[$g]=""
                SCENE_BY_SLOT[$g]=""
            fi
            echo "$g"
            return 0
        fi
    done
    return 1
}

# Submit scenes one at a time, waiting for a free GPU when all slots are busy.
for scene in ${SCENES}; do
    while true; do
        free_gpu="$(find_free_slot)" && break
        sleep 5
    done
    launch_scene "${scene}" "${free_gpu}"
done

# Wait for the remaining jobs to finish, collecting exit codes.
echo ""
echo "==> All scenes submitted; waiting for in-flight jobs..."
for g in "${GPU_LIST[@]}"; do
    pid="${SLOT_PID[$g]:-}"
    if [[ -n "${pid}" ]]; then
        wait "${pid}" 2>/dev/null
        rc=$?
        s="${SCENE_BY_SLOT[$g]}"
        SCENE_EXIT[$s]=$rc
        echo "[done]   scene=${s} gpu=${g} rc=${rc}"
    fi
done

# --- final summary -----------------------------------------------------------
echo ""
echo "==> Summary"
fail=0
for scene in ${SCENES}; do
    rc="${SCENE_EXIT[$scene]:-?}"
    status="ok"
    if [[ "${rc}" != "0" ]]; then status="FAIL"; fail=$((fail+1)); fi
    printf "  scene %2s : rc=%s [%s]  log=%s\n" "${scene}" "${rc}" "${status}" "${SCENE_LOG[$scene]}"
done
echo ""
if [[ ${fail} -gt 0 ]]; then
    echo "==> ${fail} scene(s) failed. Check the logs above."
    exit 1
fi
echo "==> All scenes completed successfully."
