#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${COLPALI_CUDA_IMAGE:-colpali-triton:phase10}"

if [[ -n "${COLPALI_SOURCE_COMMIT:-}" ]]; then
    SOURCE_COMMIT="${COLPALI_SOURCE_COMMIT}"
elif git -C "${PROJECT_ROOT}" rev-parse --verify HEAD >/dev/null 2>&1; then
    SOURCE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
elif [[ -f "${PROJECT_ROOT}/.source_commit" ]]; then
    SOURCE_COMMIT="$(<"${PROJECT_ROOT}/.source_commit")"
else
    SOURCE_COMMIT="unknown"
fi

if [[ ! "${SOURCE_COMMIT}" =~ ^([0-9a-f]{40}|unknown)$ ]]; then
    echo "Invalid source commit: ${SOURCE_COMMIT}" >&2
    exit 2
fi

exec docker build \
    --platform linux/amd64 \
    --pull \
    --build-arg "SOURCE_COMMIT=${SOURCE_COMMIT}" \
    --file "${PROJECT_ROOT}/docker/Dockerfile.cuda" \
    --tag "${IMAGE_NAME}" \
    "${PROJECT_ROOT}"
