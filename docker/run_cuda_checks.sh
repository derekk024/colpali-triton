#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${COLPALI_CUDA_IMAGE:-colpali-triton:phase10}"
DOCKER_PREFIX=(
    docker
    run
    --rm
    --platform
    linux/amd64
    --gpus
    all
    --shm-size
    1g
    "${IMAGE_NAME}"
)

if (( $# > 0 )); then
    exec "${DOCKER_PREFIX[@]}" "$@"
fi

"${DOCKER_PREFIX[@]}" \
    python -c \
    'import torch, triton; assert torch.cuda.is_available(); print({"gpu": torch.cuda.get_device_name(0), "memory_bytes": torch.cuda.get_device_properties(0).total_memory, "torch": torch.__version__, "cuda_build": torch.version.cuda, "triton": triton.__version__})'

exec "${DOCKER_PREFIX[@]}" \
    python -m pytest tests/test_maxsim.py tests/test_triton_maxsim.py
