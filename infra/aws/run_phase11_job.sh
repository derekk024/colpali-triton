#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIRECTORY=""
RESULT_DIRECTORY=""
SOURCE_COMMIT=""
RUN_ID=""
IMAGE_NAME=""

BUILD_TIMEOUT_SECONDS=1200
HARDWARE_TIMEOUT_SECONDS=120
HOST_METADATA_TIMEOUT_SECONDS=120
CUDA_TESTS_TIMEOUT_SECONDS=900
CUDA_ASSERTION_TIMEOUT_SECONDS=120
TUNING_PREFLIGHT_TIMEOUT_SECONDS=120
TUNING_TIMEOUT_SECONDS=3600
DERIVE_TIMEOUT_SECONDS=120
BENCHMARK_TIMEOUT_SECONDS=1800
RESULT_CONTRACT_TIMEOUT_SECONDS=120
PROFILER_TIMEOUT_SECONDS=300
PROFILE_STATUS_TIMEOUT_SECONDS=120
TIMEOUT_KILL_AFTER_SECONDS=30

usage() {
    cat <<'EOF'
Usage:
  infra/aws/run_phase11_job.sh \
    --source-directory ABSOLUTE_PATH \
    --result-directory ABSOLUTE_PATH \
    --source-commit 40_HEX_COMMIT \
    --run-id SAFE_RUN_ID \
    --image-name NAME:TAG

Runs the Phase 11 CUDA validation, tuning, canonical benchmark, and bounded
profiling attempts. The result directory must not already exist. Once created,
it is sealed and remains retrievable even when a mandatory step fails.
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
    echo "--source-commit must be 40 lowercase hexadecimal digits" >&2
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

# No command after result-directory creation may abort before sealing useful
# diagnostics. Exit codes are captured explicitly and become manifest state.
set +e

run_logged_timed() {
    local TIMEOUT_SECONDS="$1"
    local LOG_PATH="$2"
    shift 2
    timeout \
        --signal=TERM \
        "--kill-after=${TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TIMEOUT_SECONDS}s" \
        "$@" 2>&1 | tee "${LOG_PATH}"
    return "${PIPESTATUS[0]}"
}

run_capture_timed() {
    local TIMEOUT_SECONDS="$1"
    local STDOUT_PATH="$2"
    local STDERR_PATH="$3"
    shift 3
    timeout \
        --signal=TERM \
        "--kill-after=${TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TIMEOUT_SECONDS}s" \
        "$@" > "${STDOUT_PATH}" 2> "${STDERR_PATH}"
}

skip_log() {
    local LOG_PATH="$1"
    local REASON="$2"
    printf 'SKIPPED: %s\n' "${REASON}" > "${LOG_PATH}"
}

cleanup_timed_container() {
    local STEP_NAME="$1"
    local EXIT_CODE="$2"
    local CID_FILE="$3"
    if (( EXIT_CODE == 124 || EXIT_CODE == 137 || EXIT_CODE == 143 )) \
        && [[ -s "${CID_FILE}" ]]; then
        local CONTAINER_ID
        CONTAINER_ID="$(<"${CID_FILE}")"
        if [[ "${CONTAINER_ID}" =~ ^[0-9a-f]{12,64}$ ]]; then
            docker rm --force "${CONTAINER_ID}" \
                > "${RESULT_DIRECTORY}/${STEP_NAME}_container_cleanup.log" \
                2>&1
        else
            printf 'Invalid timed-out container ID: %s\n' "${CONTAINER_ID}" \
                > "${RESULT_DIRECTORY}/${STEP_NAME}_container_cleanup.log"
        fi
    fi
    rm -f "${CID_FILE}"
}

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
PROFILER_DOCKER_OPTIONS=(
    "${DOCKER_OPTIONS[@]}"
    --cap-add
    SYS_ADMIN
)
CID_DIRECTORY="${RESULT_DIRECTORY}/.docker-cids"
mkdir "${CID_DIRECTORY}"

run_docker_logged_timed() {
    local STEP_NAME="$1"
    local TIMEOUT_SECONDS="$2"
    local LOG_PATH="$3"
    local PROFILE_CAPABILITY="$4"
    shift 4
    local CID_FILE="${CID_DIRECTORY}/${STEP_NAME}.cid"
    local OPTIONS=("${DOCKER_OPTIONS[@]}")
    if [[ "${PROFILE_CAPABILITY}" == "profile" ]]; then
        OPTIONS=("${PROFILER_DOCKER_OPTIONS[@]}")
    fi
    timeout \
        --signal=TERM \
        "--kill-after=${TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TIMEOUT_SECONDS}s" \
        "${OPTIONS[@]}" \
        --cidfile "${CID_FILE}" \
        "${IMAGE_NAME}" \
        "$@" 2>&1 | tee "${LOG_PATH}"
    local EXIT_CODE="${PIPESTATUS[0]}"
    cleanup_timed_container "${STEP_NAME}" "${EXIT_CODE}" "${CID_FILE}"
    return "${EXIT_CODE}"
}

run_docker_capture_timed() {
    local STEP_NAME="$1"
    local TIMEOUT_SECONDS="$2"
    local STDOUT_PATH="$3"
    local STDERR_PATH="$4"
    local PROFILE_CAPABILITY="$5"
    shift 5
    local CID_FILE="${CID_DIRECTORY}/${STEP_NAME}.cid"
    local OPTIONS=("${DOCKER_OPTIONS[@]}")
    if [[ "${PROFILE_CAPABILITY}" == "profile" ]]; then
        OPTIONS=("${PROFILER_DOCKER_OPTIONS[@]}")
    fi
    timeout \
        --signal=TERM \
        "--kill-after=${TIMEOUT_KILL_AFTER_SECONDS}s" \
        "${TIMEOUT_SECONDS}s" \
        "${OPTIONS[@]}" \
        --cidfile "${CID_FILE}" \
        "${IMAGE_NAME}" \
        "$@" > "${STDOUT_PATH}" 2> "${STDERR_PATH}"
    local EXIT_CODE=$?
    cleanup_timed_container "${STEP_NAME}" "${EXIT_CODE}" "${CID_FILE}"
    return "${EXIT_CODE}"
}

CUDA_TEST_FILES=(
    tests/test_maxsim.py
    tests/test_triton_maxsim.py
    tests/test_benchmarking.py
    tests/test_benchmark_maxsim_script.py
    tests/test_tune_triton_maxsim_script.py
    tests/test_phase10_container_aws.py
    tests/test_phase11_job_timeouts.py
    tests/test_phase11_remote_workflow.py
    tests/test_phase11_tuning_preflight.py
)

export \
    SOURCE_COMMIT RUN_ID IMAGE_NAME \
    BUILD_TIMEOUT_SECONDS HARDWARE_TIMEOUT_SECONDS \
    HOST_METADATA_TIMEOUT_SECONDS CUDA_TESTS_TIMEOUT_SECONDS \
    CUDA_ASSERTION_TIMEOUT_SECONDS TUNING_PREFLIGHT_TIMEOUT_SECONDS \
    TUNING_TIMEOUT_SECONDS DERIVE_TIMEOUT_SECONDS BENCHMARK_TIMEOUT_SECONDS \
    RESULT_CONTRACT_TIMEOUT_SECONDS PROFILER_TIMEOUT_SECONDS \
    PROFILE_STATUS_TIMEOUT_SECONDS
python3 - "${RESULT_DIRECTORY}/commands.json" <<'PY'
import json
import os
from pathlib import Path
import sys

timeouts = {
    "container_build": int(os.environ["BUILD_TIMEOUT_SECONDS"]),
    "hardware_gate": int(os.environ["HARDWARE_TIMEOUT_SECONDS"]),
    "host_metadata": int(os.environ["HOST_METADATA_TIMEOUT_SECONDS"]),
    "cuda_tests": int(os.environ["CUDA_TESTS_TIMEOUT_SECONDS"]),
    "cuda_assertion": int(os.environ["CUDA_ASSERTION_TIMEOUT_SECONDS"]),
    "tuning_preflight": int(
        os.environ["TUNING_PREFLIGHT_TIMEOUT_SECONDS"]
    ),
    "tuning": int(os.environ["TUNING_TIMEOUT_SECONDS"]),
    "derive_winner": int(os.environ["DERIVE_TIMEOUT_SECONDS"]),
    "benchmark": int(os.environ["BENCHMARK_TIMEOUT_SECONDS"]),
    "result_contract": int(os.environ["RESULT_CONTRACT_TIMEOUT_SECONDS"]),
    "profiling_ncu": int(os.environ["PROFILER_TIMEOUT_SECONDS"]),
    "profiling_nsys": int(os.environ["PROFILER_TIMEOUT_SECONDS"]),
    "profiling_status": int(os.environ["PROFILE_STATUS_TIMEOUT_SECONDS"]),
}

record = {
    "format": "colpali-triton-phase11-command-contract-v3",
    "source_commit": os.environ["SOURCE_COMMIT"],
    "run_id": os.environ["RUN_ID"],
    "container_image": os.environ["IMAGE_NAME"],
    "timeouts_seconds": timeouts,
    "remote_test_scope": "cuda-relevant-minimal-dependency",
    "local_full_suite_is_separate": True,
    "hardware_gate": {
        "device": "cuda:0",
        "name": "NVIDIA L4",
        "compute_capability": [8, 9],
        "assertion": "infra/aws/assert_phase11_l4.py",
    },
    "cuda_test_files": [
        "tests/test_maxsim.py",
        "tests/test_triton_maxsim.py",
        "tests/test_benchmarking.py",
        "tests/test_benchmark_maxsim_script.py",
        "tests/test_tune_triton_maxsim_script.py",
        "tests/test_phase10_container_aws.py",
        "tests/test_phase11_job_timeouts.py",
        "tests/test_phase11_remote_workflow.py",
        "tests/test_phase11_tuning_preflight.py",
    ],
    "gpu_parity_gate": {
        "required_cases": 5,
        "skips_allowed": 0,
        "assertion": "infra/aws/assert_phase11_cuda_tests.py",
    },
    "tuning": (
        "python benchmarks/tune_triton_maxsim.py "
        "--output /phase11-results/triton_tuning.json"
    ),
    "tuning_preflight_assertion": {
        "assertion": "infra/aws/assert_phase11_tuning_preflight.py",
        "required_candidate_count": 12,
        "required_dtype_count": 2,
        "required_case_count": 4,
        "required_status": "ready",
    },
    "winner_derivation": (
        "infra/aws/derive_phase11_winner.py replaces only triton_launch "
        "in the canonical six-case benchmark config"
    ),
    "canonical_benchmark": (
        "python benchmarks/benchmark_maxsim.py "
        "--output /phase11-results/maxsim_cuda_benchmark.json"
    ),
    "result_cross_check": (
        "infra/aws/phase11_result_contract.py requires tuning, winner, "
        "derived config, source, and benchmark agreement"
    ),
    "profiling": {
        "order": ["ncu", "nsys"],
        "timeout_seconds_per_attempt": timeouts["profiling_ncu"],
        "success_requires_nonempty_report": True,
        "launch_configuration": "exact tuning winner",
        "container_capabilities": ["SYS_ADMIN"],
        "privileged_container": False,
    },
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
COMMAND_CONTRACT_EXIT=$?

if (( COMMAND_CONTRACT_EXIT == 0 )); then
    (
        cd "${SOURCE_DIRECTORY}" || exit 1
        run_logged_timed \
            "${BUILD_TIMEOUT_SECONDS}" \
            "${RESULT_DIRECTORY}/container_build.log" \
            env \
            COLPALI_SOURCE_COMMIT="${SOURCE_COMMIT}" \
            COLPALI_CUDA_IMAGE="${IMAGE_NAME}" \
            ./docker/build_cuda.sh
    )
    CONTAINER_BUILD_EXIT=$?
else
    CONTAINER_BUILD_EXIT=125
    skip_log \
        "${RESULT_DIRECTORY}/container_build.log" \
        "command contract could not be recorded"
fi

if (( CONTAINER_BUILD_EXIT == 0 )); then
    run_docker_logged_timed \
        "hardware_gate" \
        "${HARDWARE_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/hardware_gate.log" \
        "normal" \
        python infra/aws/assert_phase11_l4.py \
        --output /phase11-results/hardware_gate.json
    HARDWARE_GATE_EXIT=$?
else
    HARDWARE_GATE_EXIT=125
    skip_log "${RESULT_DIRECTORY}/hardware_gate.log" "container build failed"
fi

run_capture_timed \
    "${HOST_METADATA_TIMEOUT_SECONDS}" \
    "${RESULT_DIRECTORY}/host_metadata_stdout.log" \
    "${RESULT_DIRECTORY}/host_metadata_capture.log" \
    python3 "${SOURCE_DIRECTORY}/infra/aws/phase11_host_metadata.py" \
    --output "${RESULT_DIRECTORY}/host_metadata.json" \
    --source-commit "${SOURCE_COMMIT}" \
    --run-id "${RUN_ID}" \
    --image-name "${IMAGE_NAME}"
HOST_METADATA_EXIT=$?

if (( COMMAND_CONTRACT_EXIT == 0 \
    && CONTAINER_BUILD_EXIT == 0 \
    && HARDWARE_GATE_EXIT == 0 \
    && HOST_METADATA_EXIT == 0 )); then
    run_docker_logged_timed \
        "cuda_tests" \
        "${CUDA_TESTS_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/cuda_tests.log" \
        "normal" \
        python -m pytest \
        "${CUDA_TEST_FILES[@]}" \
        --junitxml /phase11-results/cuda_tests.xml
    CUDA_TESTS_EXIT=$?

    run_capture_timed \
        "${CUDA_ASSERTION_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/cuda_test_assertion.stdout.log" \
        "${RESULT_DIRECTORY}/cuda_test_assertion.log" \
        python3 \
        "${SOURCE_DIRECTORY}/infra/aws/assert_phase11_cuda_tests.py" \
        --junit "${RESULT_DIRECTORY}/cuda_tests.xml" \
        --output "${RESULT_DIRECTORY}/cuda_test_assertion.json"
    CUDA_ASSERTION_EXIT=$?
else
    CUDA_TESTS_EXIT=125
    CUDA_ASSERTION_EXIT=125
    if (( COMMAND_CONTRACT_EXIT != 0 )); then
        WORK_SKIP_REASON="command contract could not be recorded"
    elif (( CONTAINER_BUILD_EXIT != 0 )); then
        WORK_SKIP_REASON="container build failed"
    elif (( HARDWARE_GATE_EXIT != 0 )); then
        WORK_SKIP_REASON="NVIDIA L4 hardware gate failed"
    else
        WORK_SKIP_REASON="host metadata capture failed"
    fi
    skip_log "${RESULT_DIRECTORY}/cuda_tests.log" "${WORK_SKIP_REASON}"
    skip_log \
        "${RESULT_DIRECTORY}/cuda_test_assertion.log" \
        "${WORK_SKIP_REASON}"
fi

if (( CUDA_TESTS_EXIT == 0 && CUDA_ASSERTION_EXIT == 0 )); then
    run_docker_capture_timed \
        "tuning_preflight" \
        "${TUNING_PREFLIGHT_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/tuning_preflight.json" \
        "${RESULT_DIRECTORY}/tuning_preflight.log" \
        "normal" \
        python benchmarks/tune_triton_maxsim.py --preflight
    TUNING_PREFLIGHT_COMMAND_EXIT=$?
    if (( TUNING_PREFLIGHT_COMMAND_EXIT == 0 )); then
        run_capture_timed \
            "${TUNING_PREFLIGHT_TIMEOUT_SECONDS}" \
            "${RESULT_DIRECTORY}/tuning_preflight_assertion.stdout.log" \
            "${RESULT_DIRECTORY}/tuning_preflight_assertion.log" \
            python3 \
            "${SOURCE_DIRECTORY}/infra/aws/assert_phase11_tuning_preflight.py" \
            --preflight "${RESULT_DIRECTORY}/tuning_preflight.json" \
            --config "${SOURCE_DIRECTORY}/configs/triton_tuning.json" \
            --output \
                "${RESULT_DIRECTORY}/tuning_preflight_assertion.json"
        TUNING_PREFLIGHT_EXIT=$?
    else
        TUNING_PREFLIGHT_EXIT="${TUNING_PREFLIGHT_COMMAND_EXIT}"
        skip_log \
            "${RESULT_DIRECTORY}/tuning_preflight_assertion.log" \
            "containerized tuning preflight failed"
    fi
else
    TUNING_PREFLIGHT_EXIT=125
    skip_log \
        "${RESULT_DIRECTORY}/tuning_preflight.log" \
        "CUDA tests or their assertion failed"
    skip_log \
        "${RESULT_DIRECTORY}/tuning_preflight_assertion.log" \
        "CUDA tests or their assertion failed"
fi

if (( CUDA_TESTS_EXIT == 0 \
    && CUDA_ASSERTION_EXIT == 0 \
    && TUNING_PREFLIGHT_EXIT == 0 )); then
    run_docker_logged_timed \
        "tuning" \
        "${TUNING_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/triton_tuning.log" \
        "normal" \
        python benchmarks/tune_triton_maxsim.py \
        --output /phase11-results/triton_tuning.json
    TUNING_EXIT=$?
else
    TUNING_EXIT=125
    skip_log \
        "${RESULT_DIRECTORY}/triton_tuning.log" \
        "CUDA test, assertion, or tuning preflight gate failed"
fi

if (( TUNING_EXIT == 0 )); then
    run_docker_capture_timed \
        "derive_winner" \
        "${DERIVE_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/tuning_winner_derivation.stdout.log" \
        "${RESULT_DIRECTORY}/tuning_winner_derivation.log" \
        "normal" \
        python infra/aws/derive_phase11_winner.py \
        --tuning-result /phase11-results/triton_tuning.json \
        --base-benchmark-config \
            configs/maxsim_benchmark.json \
        --output-config \
            /phase11-results/maxsim_benchmark_tuned_config.json \
        --winner-record /phase11-results/tuning_winner.json \
        --source-commit "${SOURCE_COMMIT}"
    DERIVE_WINNER_EXIT=$?
else
    DERIVE_WINNER_EXIT=125
    skip_log \
        "${RESULT_DIRECTORY}/tuning_winner_derivation.log" \
        "tuning failed"
fi

if (( COMMAND_CONTRACT_EXIT == 0 \
    && CONTAINER_BUILD_EXIT == 0 \
    && HARDWARE_GATE_EXIT == 0 \
    && HOST_METADATA_EXIT == 0 \
    && CUDA_TESTS_EXIT == 0 \
    && CUDA_ASSERTION_EXIT == 0 \
    && TUNING_PREFLIGHT_EXIT == 0 \
    && TUNING_EXIT == 0 \
    && DERIVE_WINNER_EXIT == 0 )); then
    run_docker_logged_timed \
        "benchmark" \
        "${BENCHMARK_TIMEOUT_SECONDS}" \
        "${RESULT_DIRECTORY}/maxsim_cuda_benchmark.log" \
        "normal" \
        python benchmarks/benchmark_maxsim.py \
        --config /phase11-results/maxsim_benchmark_tuned_config.json \
        --output /phase11-results/maxsim_cuda_benchmark.json
    BENCHMARK_EXIT=$?
else
    BENCHMARK_EXIT=125
    skip_log \
        "${RESULT_DIRECTORY}/maxsim_cuda_benchmark.log" \
        "one or more prerequisite gates failed"
fi

run_capture_timed \
    "${RESULT_CONTRACT_TIMEOUT_SECONDS}" \
    "${RESULT_DIRECTORY}/result_contract_validation.stdout.log" \
    "${RESULT_DIRECTORY}/result_contract_validation.log" \
    python3 "${SOURCE_DIRECTORY}/infra/aws/phase11_result_contract.py" \
    --result-directory "${RESULT_DIRECTORY}" \
    --source-commit "${SOURCE_COMMIT}" \
    --output "${RESULT_DIRECTORY}/result_contract_validation.json"
RESULT_CONTRACT_EXIT=$?

NCU_ATTEMPTED=0
NCU_EXIT=-1
NCU_REPORT=""
NCU_METADATA=""
NSYS_ATTEMPTED=0
NSYS_EXIT=-1
NSYS_REPORT=""
NSYS_METADATA=""
PROFILE_SUCCEEDED=0

if (( RESULT_CONTRACT_EXIT == 0 )); then
    NCU_ATTEMPTED=1
    NCU_CID_FILE="${CID_DIRECTORY}/profiling_ncu.cid"
    timeout --signal=TERM --kill-after=30s 300s \
        "${PROFILER_DOCKER_OPTIONS[@]}" \
        --cidfile "${NCU_CID_FILE}" \
        "${IMAGE_NAME}" \
        ncu \
        --target-processes application-only \
        --set basic \
        --launch-count 1 \
        --force-overwrite false \
        --export /phase11-results/maxsim_profile \
        python infra/aws/profile_phase11_maxsim.py \
        --winner-record /phase11-results/tuning_winner.json \
        --metadata-output /phase11-results/profile_probe_ncu.json \
        > "${RESULT_DIRECTORY}/profiling_ncu.log" 2>&1
    NCU_EXIT=$?
    cleanup_timed_container \
        "profiling_ncu" "${NCU_EXIT}" "${NCU_CID_FILE}"
    if [[ -s "${RESULT_DIRECTORY}/maxsim_profile.ncu-rep" ]]; then
        NCU_REPORT="maxsim_profile.ncu-rep"
    fi
    if [[ -s "${RESULT_DIRECTORY}/profile_probe_ncu.json" ]]; then
        NCU_METADATA="profile_probe_ncu.json"
    fi
    if (( NCU_EXIT == 0 )) \
        && [[ -n "${NCU_REPORT}" && -n "${NCU_METADATA}" ]]; then
        PROFILE_SUCCEEDED=1
    fi
else
    skip_log \
        "${RESULT_DIRECTORY}/profiling_ncu.log" \
        "interim result contract did not pass"
fi

if (( PROFILE_SUCCEEDED == 0 && RESULT_CONTRACT_EXIT == 0 )); then
    NSYS_ATTEMPTED=1
    NSYS_CID_FILE="${CID_DIRECTORY}/profiling_nsys.cid"
    timeout --signal=TERM --kill-after=30s 300s \
        "${PROFILER_DOCKER_OPTIONS[@]}" \
        --cidfile "${NSYS_CID_FILE}" \
        "${IMAGE_NAME}" \
        nsys profile \
        --force-overwrite=false \
        --sample=none \
        --trace=cuda,nvtx,osrt \
        --output=/phase11-results/maxsim_profile \
        python infra/aws/profile_phase11_maxsim.py \
        --winner-record /phase11-results/tuning_winner.json \
        --metadata-output /phase11-results/profile_probe_nsys.json \
        > "${RESULT_DIRECTORY}/profiling_nsys.log" 2>&1
    NSYS_EXIT=$?
    cleanup_timed_container \
        "profiling_nsys" "${NSYS_EXIT}" "${NSYS_CID_FILE}"
    if [[ -s "${RESULT_DIRECTORY}/maxsim_profile.nsys-rep" ]]; then
        NSYS_REPORT="maxsim_profile.nsys-rep"
    fi
    if [[ -s "${RESULT_DIRECTORY}/profile_probe_nsys.json" ]]; then
        NSYS_METADATA="profile_probe_nsys.json"
    fi
    if (( NSYS_EXIT == 0 )) \
        && [[ -n "${NSYS_REPORT}" && -n "${NSYS_METADATA}" ]]; then
        PROFILE_SUCCEEDED=1
    fi
elif (( RESULT_CONTRACT_EXIT != 0 )); then
    skip_log \
        "${RESULT_DIRECTORY}/profiling_nsys.log" \
        "interim result contract did not pass"
else
    skip_log \
        "${RESULT_DIRECTORY}/profiling_nsys.log" \
        "ncu profiling already succeeded"
fi

run_capture_timed \
    "${PROFILE_STATUS_TIMEOUT_SECONDS}" \
    "${RESULT_DIRECTORY}/profiling_status.stdout.log" \
    "${RESULT_DIRECTORY}/profiling_status_capture.log" \
    python3 "${SOURCE_DIRECTORY}/infra/aws/phase11_profile_status.py" \
    --result-directory "${RESULT_DIRECTORY}" \
    --output "${RESULT_DIRECTORY}/profiling_status.json" \
    --winner-record "${RESULT_DIRECTORY}/tuning_winner.json" \
    --ncu-attempted "${NCU_ATTEMPTED}" \
    --ncu-exit "${NCU_EXIT}" \
    --ncu-report "${NCU_REPORT}" \
    --ncu-metadata "${NCU_METADATA}" \
    --nsys-attempted "${NSYS_ATTEMPTED}" \
    --nsys-exit "${NSYS_EXIT}" \
    --nsys-report "${NSYS_REPORT}" \
    --nsys-metadata "${NSYS_METADATA}"
PROFILING_STATUS_EXIT=$?

if (( PROFILING_STATUS_EXIT != 0 )) \
    || [[ ! -s "${RESULT_DIRECTORY}/profiling_status.json" ]]; then
    rm -f "${RESULT_DIRECTORY}/profiling_status.json"
    if (( NCU_ATTEMPTED == 1 )); then
        NCU_ATTEMPTED_JSON=true
        NCU_EXIT_JSON="${NCU_EXIT}"
    else
        NCU_ATTEMPTED_JSON=false
        NCU_EXIT_JSON=null
    fi
    if (( NSYS_ATTEMPTED == 1 )); then
        NSYS_ATTEMPTED_JSON=true
        NSYS_EXIT_JSON="${NSYS_EXIT}"
    else
        NSYS_ATTEMPTED_JSON=false
        NSYS_EXIT_JSON=null
    fi
    printf '%s\n' \
        '{' \
        '  "format": "colpali-triton-phase11-profiling-status-v1",' \
        '  "succeeded": false,' \
        '  "profiler": null,' \
        '  "report": null,' \
        '  "metadata": null,' \
        '  "winner_record": null,' \
        '  "winner_record_sha256": null,' \
        '  "winner_record_error": "profile status helper failed",' \
        '  "timeout_seconds_per_attempt": 300,' \
        '  "container_capabilities": ["SYS_ADMIN"],' \
        '  "privileged_container": false,' \
        '  "attempts": [' \
        "    {\"profiler\": \"ncu\", \"attempted\": ${NCU_ATTEMPTED_JSON}, \"exit_code\": ${NCU_EXIT_JSON}, \"report\": null, \"report_valid\": false, \"metadata\": null, \"metadata_matches_winner\": false, \"succeeded\": false}," \
        "    {\"profiler\": \"nsys\", \"attempted\": ${NSYS_ATTEMPTED_JSON}, \"exit_code\": ${NSYS_EXIT_JSON}, \"report\": null, \"report_valid\": false, \"metadata\": null, \"metadata_matches_winner\": false, \"succeeded\": false}" \
        '  ]' \
        '}' \
        > "${RESULT_DIRECTORY}/profiling_status.json"
fi

rmdir "${CID_DIRECTORY}"

export \
    COMMAND_CONTRACT_EXIT CONTAINER_BUILD_EXIT HARDWARE_GATE_EXIT \
    HOST_METADATA_EXIT \
    CUDA_TESTS_EXIT CUDA_ASSERTION_EXIT TUNING_PREFLIGHT_EXIT \
    TUNING_EXIT DERIVE_WINNER_EXIT BENCHMARK_EXIT PROFILING_STATUS_EXIT \
    RESULT_CONTRACT_EXIT NCU_ATTEMPTED NCU_EXIT NSYS_ATTEMPTED NSYS_EXIT
python3 - "${RESULT_DIRECTORY}/step_status.json" <<'PY'
import json
import os
from pathlib import Path
import sys

names = (
    "COMMAND_CONTRACT",
    "CONTAINER_BUILD",
    "HARDWARE_GATE",
    "HOST_METADATA",
    "CUDA_TESTS",
    "CUDA_ASSERTION",
    "TUNING_PREFLIGHT",
    "TUNING",
    "DERIVE_WINNER",
    "BENCHMARK",
    "PROFILING_STATUS",
    "RESULT_CONTRACT",
)
timeouts = {
    "container_build": int(os.environ["BUILD_TIMEOUT_SECONDS"]),
    "hardware_gate": int(os.environ["HARDWARE_TIMEOUT_SECONDS"]),
    "host_metadata": int(os.environ["HOST_METADATA_TIMEOUT_SECONDS"]),
    "cuda_tests": int(os.environ["CUDA_TESTS_TIMEOUT_SECONDS"]),
    "cuda_assertion": int(os.environ["CUDA_ASSERTION_TIMEOUT_SECONDS"]),
    "tuning_preflight": int(
        os.environ["TUNING_PREFLIGHT_TIMEOUT_SECONDS"]
    ),
    "tuning": int(os.environ["TUNING_TIMEOUT_SECONDS"]),
    "derive_winner": int(os.environ["DERIVE_TIMEOUT_SECONDS"]),
    "benchmark": int(os.environ["BENCHMARK_TIMEOUT_SECONDS"]),
    "result_contract": int(os.environ["RESULT_CONTRACT_TIMEOUT_SECONDS"]),
    "profiling_ncu": int(os.environ["PROFILER_TIMEOUT_SECONDS"]),
    "profiling_nsys": int(os.environ["PROFILER_TIMEOUT_SECONDS"]),
    "profiling_status": int(os.environ["PROFILE_STATUS_TIMEOUT_SECONDS"]),
}
steps = {
    name.lower(): {
        "exit_code": int(os.environ[f"{name}_EXIT"]),
        "succeeded": int(os.environ[f"{name}_EXIT"]) == 0,
    }
    for name in names
}
path = Path(sys.argv[1])
with path.open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "format": "colpali-triton-phase11-step-status-v1",
            "steps": steps,
            "timeouts_seconds": timeouts,
            "profiling_attempts": {
                profiler: {
                    "attempted": int(
                        os.environ[f"{profiler.upper()}_ATTEMPTED"]
                    )
                    == 1,
                    "exit_code": (
                        int(os.environ[f"{profiler.upper()}_EXIT"])
                        if int(
                            os.environ[
                                f"{profiler.upper()}_ATTEMPTED"
                            ]
                        )
                        == 1
                        else None
                    ),
                    "timeout_seconds": timeouts[f"profiling_{profiler}"],
                    "timed_out": int(
                        os.environ[f"{profiler.upper()}_EXIT"]
                    )
                    in {124, 137, 143},
                }
                for profiler in ("ncu", "nsys")
            },
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
STEP_STATUS_EXIT=$?

sudo chmod -R a+rX "${RESULT_DIRECTORY}"

set -e
python3 - "${RESULT_DIRECTORY}" "${SOURCE_COMMIT}" "${RUN_ID}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
source_commit = sys.argv[2]
run_id = sys.argv[3]
step_status = json.loads((root / "step_status.json").read_text())
profiling = json.loads((root / "profiling_status.json").read_text())

failed_steps = [
    name
    for name, record in step_status["steps"].items()
    if not record["succeeded"]
]
failure_reasons = [f"mandatory step failed: {name}" for name in failed_steps]
profile_report = profiling.get("report")
profile_metadata = profiling.get("metadata")
winner_record = profiling.get("winner_record")
profile_valid = (
    profiling.get("succeeded") is True
    and isinstance(profile_report, str)
    and (root / profile_report).is_file()
    and (root / profile_report).stat().st_size > 0
    and isinstance(profile_metadata, str)
    and (root / profile_metadata).is_file()
    and (root / profile_metadata).stat().st_size > 0
    and isinstance(winner_record, str)
    and (root / winner_record).is_file()
    and (root / winner_record).stat().st_size > 0
)
if not profile_valid:
    failure_reasons.append(
        "no profiler completed with exit code 0 and a nonempty report"
    )
status = "complete" if not failure_reasons else "incomplete"

files = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    if relative in {"artifact_manifest.json", "SEALED.sha256"}:
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
    "format": "colpali-triton-phase11-artifact-manifest-v2",
    "sealed_at": datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    ),
    "status": status,
    "failure_reasons": failure_reasons,
    "source_commit": source_commit,
    "run_id": run_id,
    "steps": step_status["steps"],
    "profiling": profiling,
    "files": files,
}
manifest_path = root / "artifact_manifest.json"
with manifest_path.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
with (root / "SEALED.sha256").open("x", encoding="ascii") as handle:
    handle.write(f"{manifest_digest}  artifact_manifest.json\n")
print(json.dumps({"status": status, "failure_reasons": failure_reasons}))
PY

printf 'Sealed Phase 11 result directory without terminating the host: %s\n' \
    "${RESULT_DIRECTORY}"
