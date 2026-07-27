#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_USER="ubuntu"
SSH_PORT="22"
REMOTE_BASE="/home/ubuntu/colpali-triton"
IDENTITY_FILE=""
HOST=""
IMAGE_NAME="colpali-triton:phase10"

usage() {
    cat <<'EOF'
Usage:
  infra/aws/deploy_committed_source.sh \
    --host HOST \
    --identity-file PATH \
    [--port 22] \
    [--remote-base /home/ubuntu/colpali-triton] \
    [--image-name colpali-triton:phase10]

Streams the clean, committed HEAD to a new commit-specific directory, builds
the pinned linux/amd64 CUDA image, and runs its GPU smoke/unit checks.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --host)
            HOST="${2:-}"
            shift 2
            ;;
        --identity-file)
            IDENTITY_FILE="${2:-}"
            shift 2
            ;;
        --port)
            SSH_PORT="${2:-}"
            shift 2
            ;;
        --remote-base)
            REMOTE_BASE="${2:-}"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${HOST}" =~ ^[A-Za-z0-9.-]+$ ]]; then
    echo "--host must be an IPv4 address or DNS name" >&2
    exit 2
fi
if [[ ! -f "${IDENTITY_FILE}" ]]; then
    echo "--identity-file must name an existing regular file" >&2
    exit 2
fi
if [[ ! "${SSH_PORT}" =~ ^[0-9]{1,5}$ ]] \
    || (( SSH_PORT < 1 || SSH_PORT > 65535 )); then
    echo "--port must be between 1 and 65535" >&2
    exit 2
fi
if [[ ! "${REMOTE_BASE}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "--remote-base must be a safe absolute path without spaces" >&2
    exit 2
fi
if [[ ! "${IMAGE_NAME}" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]]; then
    echo "--image-name must be an explicit safe name:tag" >&2
    exit 2
fi

if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain)" ]]; then
    echo "Refusing deployment: commit or remove all worktree changes first." >&2
    exit 2
fi

SOURCE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse --verify HEAD)"
if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Could not resolve an exact source commit." >&2
    exit 2
fi

REMOTE_DIRECTORY="${REMOTE_BASE}/${SOURCE_COMMIT}"
SSH_TARGET="${SSH_USER}@${HOST}"
SSH_OPTIONS=(
    -o BatchMode=yes
    -o IdentitiesOnly=yes
    -o StrictHostKeyChecking=accept-new
    -i "${IDENTITY_FILE}"
    -p "${SSH_PORT}"
)

ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET}" \
    "sudo test -f /var/lib/colpali-triton/bootstrap.ready"

git -C "${PROJECT_ROOT}" archive --format=tar HEAD \
    | ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET}" \
        "set -eu; test ! -e '${REMOTE_DIRECTORY}'; mkdir -p '${REMOTE_DIRECTORY}'; tar -xf - -C '${REMOTE_DIRECTORY}'; printf '%s\n' '${SOURCE_COMMIT}' > '${REMOTE_DIRECTORY}/.source_commit'"

ssh "${SSH_OPTIONS[@]}" "${SSH_TARGET}" \
    "set -eu; cd '${REMOTE_DIRECTORY}'; COLPALI_SOURCE_COMMIT='${SOURCE_COMMIT}' COLPALI_CUDA_IMAGE='${IMAGE_NAME}' ./docker/build_cuda.sh; COLPALI_CUDA_IMAGE='${IMAGE_NAME}' ./docker/run_cuda_checks.sh"

printf 'Deployed commit %s to %s:%s\n' \
    "${SOURCE_COMMIT}" "${SSH_TARGET}" "${REMOTE_DIRECTORY}"
