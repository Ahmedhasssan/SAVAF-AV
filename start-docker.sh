#!/bin/bash
set -e

CONTAINER_NAME="savaf-av-dev"
IMAGE_NAME="savaf-av:latest"
WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${WORKSPACE_DIR}/data"

if [[ "$1" == "--rebuild" ]]; then
    echo "==> Forcing image rebuild..."
    docker build --no-cache -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
    shift
elif [[ "$1" == "--fresh" ]]; then
    echo "==> Removing existing container..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    shift
fi

if ! docker image inspect "${IMAGE_NAME}" &>/dev/null; then
    echo "==> Building Docker image..."
    docker build -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "==> Attaching to running container '${CONTAINER_NAME}'..."
        docker exec -it "${CONTAINER_NAME}" bash
    else
        echo "==> Starting stopped container '${CONTAINER_NAME}'..."
        docker start -ai "${CONTAINER_NAME}"
    fi
else
    echo "==> Creating new container '${CONTAINER_NAME}'..."
    docker run -it \
        --name "${CONTAINER_NAME}" \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --shm-size 64G \
        -v "${WORKSPACE_DIR}:/workspace/SAVAF-AV" \
        -v "${DATA_DIR}:/workspace/data" \
        -e HSA_OVERRIDE_GFX_VERSION=9.4.2 \
        -e ROCM_PATH=/opt/rocm \
        -e HIP_VISIBLE_DEVICES=0 \
        "${IMAGE_NAME}"
fi
