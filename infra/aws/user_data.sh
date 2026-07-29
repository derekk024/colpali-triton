#!/usr/bin/env bash
set -Eeuo pipefail

exec > >(tee -a /var/log/colpali-triton-bootstrap.log | logger -t colpali-user-data -s 2>/dev/console) 2>&1

BASE_IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56"
STATE_DIRECTORY="/var/lib/colpali-triton"

install -d -m 0755 "${STATE_DIRECTORY}"
rm -f "${STATE_DIRECTORY}/bootstrap.ready"

for command_name in docker nvidia-ctk nvidia-smi python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required DLAMI command is missing: ${command_name}" >&2
        exit 1
    fi
done

nvidia-smi
systemctl enable --now docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
usermod -aG docker ubuntu

docker pull --platform linux/amd64 "${BASE_IMAGE}"
CONTAINER_ENVIRONMENT="$(
    docker run \
        --rm \
        --platform linux/amd64 \
        --gpus all \
        "${BASE_IMAGE}" \
        python -c 'import json, platform, torch, triton; print(json.dumps({"architecture": platform.machine(), "torch": torch.__version__, "cuda_build": torch.version.cuda, "triton": triton.__version__, "cuda_available": torch.cuda.is_available(), "gpu_name": torch.cuda.get_device_name(0), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory}, sort_keys=True))' \
    | tail -n 1
)"

export BASE_IMAGE CONTAINER_ENVIRONMENT
python3 - <<'PY' >"${STATE_DIRECTORY}/bootstrap-environment.json"
import json
import os
import subprocess

record = {
    "base_image": os.environ["BASE_IMAGE"],
    "container": json.loads(os.environ["CONTAINER_ENVIRONMENT"]),
    "docker_version": subprocess.check_output(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        text=True,
    ).strip(),
    "nvidia_container_toolkit": subprocess.check_output(
        ["nvidia-ctk", "--version"],
        text=True,
    ).strip(),
    "nvidia_smi": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip(),
}
print(json.dumps(record, indent=2, sort_keys=True))
PY

chmod 0644 "${STATE_DIRECTORY}/bootstrap-environment.json"
touch "${STATE_DIRECTORY}/bootstrap.ready"
echo "ColPali Triton L4 host bootstrap complete."
