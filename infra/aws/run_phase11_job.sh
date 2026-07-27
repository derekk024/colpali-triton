#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIRECTORY=""
RESULT_DIRECTORY=""
SOURCE_COMMIT=""
RUN_ID=""
IMAGE_NAME=""

usage() {
    cat <<'EOF'
Usage:
  infra/aws/run_phase11_job.sh \
    --source-directory ABSOLUTE_PATH \
    --result-directory ABSOLUTE_PATH \
    --source-commit 40_HEX_COMMIT \
    --run-id SAFE_RUN_ID \
    --image-name NAME:TAG

Runs the full Phase 11 CUDA validation, tuning, canonical benchmark, and one
bounded profiler probe. The result directory must not already exist.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --source-directory)
            SOURCE_DIRECTORY="${2:-}"
            shift 2
            ;;
        --result-directory)
            RESULT_DIRECTORY="${2:-}"
            shift 2
            ;;
        --source-commit)
            SOURCE_COMMIT="${2:-}"
            shift 2
            ;;
        --run-id)
            RUN_ID="${2:-}"
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

if [[ ! "${SOURCE_DIRECTORY}" =~ ^/[A-Za-z0-9._/-]+$ ]] \
    || [[ ! -d "${SOURCE_DIRECTORY}" ]]; then
    echo "--source-directory must be an existing safe absolute path" >&2
    exit 2
fi
if [[ ! "${RESULT_DIRECTORY}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "--result-directory must be a safe absolute path" >&2
    exit 2
fi
if [[ ! "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "--source-commit must be exactly 40 lowercase hexadecimal digits" >&2
    exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
    echo "--run-id must be 1-80 safe characters" >&2
    exit 2
fi
if [[ ! "${IMAGE_NAME}" =~ ^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$ ]]; then
    echo "--image-name must be an explicit safe name:tag" >&2
    exit 2
fi
if [[ ! -f "${SOURCE_DIRECTORY}/.source_commit" ]] \
    || [[ "$(<"${SOURCE_DIRECTORY}/.source_commit")" != "${SOURCE_COMMIT}" ]]; then
    echo "deployed source marker does not match --source-commit" >&2
    exit 2
fi
if [[ -e "${RESULT_DIRECTORY}" ]]; then
    echo "Refusing to overwrite existing remote results: ${RESULT_DIRECTORY}" >&2
    exit 2
fi

umask 022
mkdir -p "$(dirname "${RESULT_DIRECTORY}")"
mkdir "${RESULT_DIRECTORY}"

DOCKER_OPTIONS=(
    docker
    run
    --rm
    --platform
    linux/amd64
    --gpus
    all
    --shm-size
    1g
    --volume
    "${RESULT_DIRECTORY}:/phase11-results"
)

python3 "${SOURCE_DIRECTORY}/infra/aws/phase11_host_metadata.py" \
    --output "${RESULT_DIRECTORY}/host_metadata.json" \
    --source-commit "${SOURCE_COMMIT}" \
    --run-id "${RUN_ID}" \
    --image-name "${IMAGE_NAME}"

{
    printf '{\n'
    printf '  "format": "colpali-triton-phase11-command-contract-v1",\n'
    printf '  "source_commit": "%s",\n' "${SOURCE_COMMIT}"
    printf '  "run_id": "%s",\n' "${RUN_ID}"
    printf '  "mandatory_steps": [\n'
    printf '    "python -m pytest --junitxml /phase11-results/full_cuda_tests.xml",\n'
    printf '    "python benchmarks/tune_triton_maxsim.py --preflight",\n'
    printf '    "python benchmarks/tune_triton_maxsim.py --output /phase11-results/triton_tuning.json",\n'
    printf '    "python benchmarks/benchmark_maxsim.py --output /phase11-results/maxsim_cuda_benchmark.json"\n'
    printf '  ],\n'
    printf '  "profiling": "one launch, 300 second hard timeout when ncu or nsys is available"\n'
    printf '}\n'
} > "${RESULT_DIRECTORY}/commands.json"

"${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    python -m pytest \
    --junitxml /phase11-results/full_cuda_tests.xml \
    2>&1 | tee "${RESULT_DIRECTORY}/full_cuda_tests.log"

"${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    python benchmarks/tune_triton_maxsim.py --preflight \
    > "${RESULT_DIRECTORY}/tuning_preflight.json"

"${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    python benchmarks/tune_triton_maxsim.py \
    --output /phase11-results/triton_tuning.json \
    2>&1 | tee "${RESULT_DIRECTORY}/triton_tuning.log"

"${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    python benchmarks/benchmark_maxsim.py \
    --output /phase11-results/maxsim_cuda_benchmark.json \
    2>&1 | tee "${RESULT_DIRECTORY}/maxsim_cuda_benchmark.log"

PROFILE_STATUS="${RESULT_DIRECTORY}/profiling_status.json"
PROFILE_LOG="${RESULT_DIRECTORY}/profiling.log"
if "${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    sh -c 'command -v ncu >/dev/null 2>&1'; then
    set +e
    timeout --signal=TERM --kill-after=30s 300s \
        "${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
        ncu \
        --target-processes application-only \
        --set basic \
        --launch-count 1 \
        --force-overwrite false \
        --export /phase11-results/maxsim_profile \
        python infra/aws/profile_phase11_maxsim.py \
        > "${PROFILE_LOG}" 2>&1
    PROFILE_EXIT_CODE=$?
    set -e
    printf '{"profiler":"ncu","attempted":true,"timeout_seconds":300,"exit_code":%d}\n' \
        "${PROFILE_EXIT_CODE}" > "${PROFILE_STATUS}"
elif "${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
    sh -c 'command -v nsys >/dev/null 2>&1'; then
    set +e
    timeout --signal=TERM --kill-after=30s 300s \
        "${DOCKER_OPTIONS[@]}" "${IMAGE_NAME}" \
        nsys profile \
        --force-overwrite=false \
        --sample=none \
        --trace=cuda,nvtx,osrt \
        --output=/phase11-results/maxsim_profile \
        python infra/aws/profile_phase11_maxsim.py \
        > "${PROFILE_LOG}" 2>&1
    PROFILE_EXIT_CODE=$?
    set -e
    printf '{"profiler":"nsys","attempted":true,"timeout_seconds":300,"exit_code":%d}\n' \
        "${PROFILE_EXIT_CODE}" > "${PROFILE_STATUS}"
else
    printf '%s\n' \
        '{"profiler":null,"attempted":false,"reason":"ncu and nsys are absent from the pinned container"}' \
        > "${PROFILE_STATUS}"
fi

sudo chmod -R a+rX "${RESULT_DIRECTORY}"

for REQUIRED_ARTIFACT in \
    host_metadata.json \
    commands.json \
    full_cuda_tests.xml \
    full_cuda_tests.log \
    tuning_preflight.json \
    triton_tuning.json \
    triton_tuning.log \
    maxsim_cuda_benchmark.json \
    maxsim_cuda_benchmark.log \
    profiling_status.json; do
    if [[ ! -f "${RESULT_DIRECTORY}/${REQUIRED_ARTIFACT}" ]]; then
        echo "Mandatory artifact was not produced: ${REQUIRED_ARTIFACT}" >&2
        exit 1
    fi
done

python3 - "${RESULT_DIRECTORY}" "${SOURCE_COMMIT}" "${RUN_ID}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
source_commit = sys.argv[2]
run_id = sys.argv[3]

files = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    if relative in {"artifact_manifest.json", "COMPLETE.sha256"}:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append(
        {
            "path": relative,
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    )

manifest = {
    "format": "colpali-triton-phase11-artifact-manifest-v1",
    "completed_at": datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    ),
    "source_commit": source_commit,
    "run_id": run_id,
    "files": files,
}
manifest_path = root / "artifact_manifest.json"
with manifest_path.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
with (root / "COMPLETE.sha256").open("x", encoding="ascii") as handle:
    handle.write(f"{manifest_digest}  artifact_manifest.json\n")
PY

printf 'Completed immutable Phase 11 result directory: %s\n' \
    "${RESULT_DIRECTORY}"
