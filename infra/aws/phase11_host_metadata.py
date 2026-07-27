#!/usr/bin/env python3
"""Capture and validate the exact Phase 11 AWS host/container environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib import error, request


IMDS_BASE = "http://169.254.169.254/latest"
ENVIRONMENT_CONFIG_PATH = Path(__file__).with_name("l4_environment.json")
REQUIRED_INSTANCE_TYPES = frozenset({"g6.xlarge", "g6.2xlarge"})
AMI_PATTERN = re.compile(r"^ami-[0-9a-f]{8,17}$")
INSTANCE_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")
REGION_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+-\d+$"
)
ACCOUNT_PATTERN = re.compile(r"^\d{12}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
GPU_UUID_PATTERN = re.compile(r"^GPU-[A-Fa-f0-9-]+$")
MINIMUM_L4_MEMORY_BYTES = 21 * 1024**3
MAXIMUM_L4_MEMORY_BYTES = 25 * 1024**3
SOURCE_LABEL = "local-worktree"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": list(command),
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _imds_identity() -> dict[str, Any]:
    token_request = request.Request(
        f"{IMDS_BASE}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with request.urlopen(token_request, timeout=2) as response:
            token = response.read().decode("utf-8")
        identity_request = request.Request(
            f"{IMDS_BASE}/dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with request.urlopen(identity_request, timeout=2) as response:
            raw = response.read()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("identity document is not a JSON object")
        allowed = (
            "accountId",
            "architecture",
            "availabilityZone",
            "imageId",
            "instanceId",
            "instanceType",
            "privateIp",
            "region",
            "version",
        )
        return {
            "available": True,
            "identity_document": {
                key: value[key] for key in allowed if key in value
            },
        }
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_environment_contract(
    path: Path = ENVIRONMENT_CONFIG_PATH,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load the committed environment contract and retain load evidence."""

    evidence: dict[str, Any] = {
        "path": str(path),
        "available": False,
    }
    try:
        raw = path.read_bytes()
        evidence["sha256"] = hashlib.sha256(raw).hexdigest()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(value, dict):
            raise ValueError("environment contract is not a JSON object")
        required = {
            "schema_version",
            "format",
            "platform",
            "recommended_instance_type",
            "allowed_instance_types",
            "ami",
            "root_volume",
            "container",
            "bootstrap",
            "tags",
            "official_sources",
        }
        if set(value) != required:
            raise ValueError("environment contract keys differ from the schema")
        if value.get("schema_version") != 1 or value.get("format") != (
            "colpali-triton-aws-l4-environment"
        ):
            raise ValueError("environment contract format is unsupported")
        if value.get("platform") != "linux/amd64":
            raise ValueError("environment contract platform is not linux/amd64")

        allowed = value.get("allowed_instance_types")
        if not isinstance(allowed, dict) or set(allowed) != (
            REQUIRED_INSTANCE_TYPES
        ):
            raise ValueError(
                "environment contract instance types differ from the g6 scope"
            )
        ami = value.get("ami")
        if not isinstance(ami, dict) or ami.get("architecture") != "x86_64":
            raise ValueError("environment contract AMI is not x86_64")
        container = value.get("container")
        if not isinstance(container, dict):
            raise ValueError("environment contract container is invalid")
        required_container = {
            "base_tag",
            "base_manifest_digest",
            "base_reference",
            "torch_version",
            "cuda_version",
            "triton_version",
        }
        if not required_container.issubset(container):
            raise ValueError("environment contract container pins are incomplete")
        if (
            container.get("base_reference")
            != (
                f"{container.get('base_tag')}@"
                f"{container.get('base_manifest_digest')}"
            )
            or not isinstance(container.get("base_tag"), str)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(container.get("base_manifest_digest")),
            )
            or not all(
                isinstance(container.get(key), str) and container.get(key)
                for key in (
                    "torch_version",
                    "cuda_version",
                    "triton_version",
                )
            )
        ):
            raise ValueError("environment contract container pins are invalid")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return None, evidence

    evidence.update(
        {
            "available": True,
            "format": value["format"],
            "schema_version": value["schema_version"],
        }
    )
    return value, evidence


def _bootstrap_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "available": False,
        "path": str(path),
    }
    try:
        raw = path.read_bytes()
        evidence["sha256"] = hashlib.sha256(raw).hexdigest()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(value, dict):
            raise ValueError("bootstrap environment is not a JSON object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        return evidence
    evidence.update({"available": True, "record": value})
    return evidence


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, dict) else None


def _command_succeeded(value: object) -> bool:
    record = _mapping(value)
    returncode = record.get("returncode") if record is not None else None
    return bool(
        record is not None
        and record.get("available") is True
        and isinstance(returncode, int)
        and not isinstance(returncode, bool)
        and returncode == 0
    )


def _parse_json_object(raw: object) -> Mapping[str, Any] | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _parse_docker_image(raw: object) -> Mapping[str, Any] | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError):
        return None
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], dict)
    ):
        return None
    return value[0]


def _valid_instance_ip(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        address.is_unspecified
        or address.is_loopback
        or address.is_multicast
        or address.is_link_local
    )


def _validate_nvidia_smi(
    command_record: object,
    errors: list[str],
) -> None:
    if not _command_succeeded(command_record):
        errors.append("nvidia-smi query did not complete successfully")
        return
    command = _mapping(command_record)
    assert command is not None
    stdout = command.get("stdout")
    lines = (
        [line for line in stdout.splitlines() if line.strip()]
        if isinstance(stdout, str)
        else []
    )
    if len(lines) != 1:
        errors.append("nvidia-smi did not report exactly one GPU")
        return
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) < 7:
        errors.append("nvidia-smi output does not match the requested schema")
        return
    if fields[0] != "0":
        errors.append("nvidia-smi did not report the L4 as GPU index zero")
    if fields[1] != "NVIDIA L4":
        errors.append("nvidia-smi GPU name is not NVIDIA L4")
    if not GPU_UUID_PATTERN.fullmatch(fields[2]):
        errors.append("nvidia-smi GPU UUID is invalid")
    if fields[6] != "8.9":
        errors.append("nvidia-smi compute capability is not 8.9")


def _validate_bootstrap(
    bootstrap: object,
    *,
    container_contract: Mapping[str, Any],
    errors: list[str],
) -> None:
    evidence = _mapping(bootstrap)
    record = (
        _mapping(evidence.get("record"))
        if evidence is not None and evidence.get("available") is True
        else None
    )
    if record is None:
        errors.append("bootstrap environment evidence is unavailable or invalid")
        return
    if record.get("base_image") != container_contract["base_reference"]:
        errors.append("bootstrap base image does not match the pinned reference")
    container = _mapping(record.get("container"))
    if container is None:
        errors.append("bootstrap container record is missing")
        return
    if container.get("architecture") != "x86_64":
        errors.append("bootstrap container architecture is not x86_64")
    torch_version = container.get("torch")
    if (
        not isinstance(torch_version, str)
        or torch_version.split("+", 1)[0]
        != container_contract["torch_version"]
    ):
        errors.append("bootstrap PyTorch version does not match the pin")
    if container.get("cuda_build") != container_contract["cuda_version"]:
        errors.append("bootstrap CUDA build does not match the pin")
    if container.get("triton") != container_contract["triton_version"]:
        errors.append("bootstrap Triton version does not match the pin")
    if container.get("cuda_available") is not True:
        errors.append("bootstrap container did not have CUDA available")
    if container.get("gpu_name") != "NVIDIA L4":
        errors.append("bootstrap container GPU is not NVIDIA L4")
    memory = container.get("gpu_memory_bytes")
    if (
        not isinstance(memory, int)
        or isinstance(memory, bool)
        or not MINIMUM_L4_MEMORY_BYTES
        <= memory
        <= MAXIMUM_L4_MEMORY_BYTES
    ):
        errors.append("bootstrap GPU memory is outside the full-L4 range")
    for key, label in (
        ("docker_version", "Docker version"),
        ("nvidia_container_toolkit", "NVIDIA Container Toolkit version"),
        ("nvidia_smi", "bootstrap nvidia-smi record"),
    ):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} is missing from bootstrap evidence")


def _validate_docker(
    record: Mapping[str, Any],
    *,
    source_commit: str,
    container_contract: Mapping[str, Any],
    errors: list[str],
) -> None:
    docker_version = record.get("docker_version")
    if not _command_succeeded(docker_version):
        errors.append("Docker daemon version query did not succeed")
    else:
        version_command = _mapping(docker_version)
        assert version_command is not None
        version = _parse_json_object(version_command.get("stdout"))
        server = _mapping(version.get("Server")) if version is not None else None
        if (
            server is None
            or not isinstance(server.get("Version"), str)
            or not server.get("Version")
        ):
            errors.append("Docker daemon version evidence is invalid")

    image_command = record.get("docker_image")
    if not _command_succeeded(image_command):
        errors.append("Docker image inspection did not succeed")
        return
    image_record = _mapping(image_command)
    assert image_record is not None
    image = _parse_docker_image(image_record.get("stdout"))
    if image is None:
        errors.append("Docker image inspection output is invalid")
        return
    if image.get("Architecture") != "amd64" or image.get("Os") != "linux":
        errors.append("Docker image platform is not linux/amd64")
    image_id = image.get("Id")
    if not isinstance(image_id, str) or not IMAGE_ID_PATTERN.fullmatch(image_id):
        errors.append("Docker image ID is invalid")
    image_config = _mapping(image.get("Config"))
    labels = (
        _mapping(image_config.get("Labels"))
        if image_config is not None
        else None
    )
    expected_labels = {
        "org.opencontainers.image.revision": source_commit,
        "org.opencontainers.image.source": SOURCE_LABEL,
        "org.opencontainers.image.base.name": container_contract["base_tag"],
        "org.opencontainers.image.base.digest": (
            container_contract["base_manifest_digest"]
        ),
    }
    if labels is None:
        errors.append("Docker image labels are missing")
    else:
        for key, expected in expected_labels.items():
            if labels.get(key) != expected:
                errors.append(f"Docker image label {key} does not match")
    environment = (
        image_config.get("Env") if image_config is not None else None
    )
    if (
        not isinstance(environment, list)
        or not all(isinstance(item, str) for item in environment)
        or f"COLPALI_SOURCE_COMMIT={source_commit}" not in environment
    ):
        errors.append("Docker image source-commit environment is missing")


def validate_host_evidence(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_instance_id: str,
    expected_instance_type: str,
    expected_region: str,
    expected_ami_id: str,
    expected_account_id: str,
) -> dict[str, Any]:
    """Validate all evidence needed to identify the exact paid GPU host."""

    errors: list[str] = []
    if not INSTANCE_PATTERN.fullmatch(expected_instance_id):
        errors.append("expected EC2 instance ID is invalid")
    if expected_instance_type not in REQUIRED_INSTANCE_TYPES:
        errors.append("expected EC2 instance type is outside scope")
    if not REGION_PATTERN.fullmatch(expected_region):
        errors.append("expected EC2 region is invalid")
    if not AMI_PATTERN.fullmatch(expected_ami_id):
        errors.append("expected EC2 AMI ID is invalid")
    if not ACCOUNT_PATTERN.fullmatch(expected_account_id):
        errors.append("expected EC2 account ID is invalid")
    source_commit = record.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or not COMMIT_PATTERN.fullmatch(source_commit)
    ):
        errors.append("source commit is not 40 lowercase hexadecimal digits")
        source_commit = ""

    ami_contract = _mapping(contract.get("ami"))
    container_contract = _mapping(contract.get("container"))
    allowed = _mapping(contract.get("allowed_instance_types"))
    if ami_contract is None or container_contract is None or allowed is None:
        return {
            "format": "colpali-triton-phase11-host-validation-v1",
            "status": "failed",
            "errors": ["environment contract is incomplete"],
            "expected": {},
        }

    host = _mapping(record.get("host"))
    if host is None or host.get("architecture") != ami_contract["architecture"]:
        errors.append("host architecture is not x86_64")

    ec2 = _mapping(record.get("ec2"))
    identity = (
        _mapping(ec2.get("identity_document"))
        if ec2 is not None and ec2.get("available") is True
        else None
    )
    if identity is None:
        errors.append("EC2 IMDSv2 identity evidence is unavailable")
    else:
        instance_id = identity.get("instanceId")
        image_id = identity.get("imageId")
        region = identity.get("region")
        account_id = identity.get("accountId")
        availability_zone = identity.get("availabilityZone")
        instance_type = identity.get("instanceType")
        if identity.get("architecture") != ami_contract["architecture"]:
            errors.append("EC2 identity architecture is not x86_64")
        if not isinstance(instance_type, str) or instance_type not in allowed:
            errors.append("EC2 instance type is outside the scoped G6 types")
        elif instance_type != expected_instance_type:
            errors.append(
                "EC2 instance type does not match the launch receipt"
            )
        if (
            not isinstance(instance_id, str)
            or not INSTANCE_PATTERN.fullmatch(instance_id)
        ):
            errors.append("EC2 instance ID is invalid")
        elif instance_id != expected_instance_id:
            errors.append("EC2 instance ID does not match the launch receipt")
        if (
            not isinstance(image_id, str)
            or not AMI_PATTERN.fullmatch(image_id)
        ):
            errors.append("EC2 AMI ID is invalid")
        elif image_id != expected_ami_id:
            errors.append("EC2 AMI ID does not match the launch receipt")
        if not isinstance(region, str) or not REGION_PATTERN.fullmatch(region):
            errors.append("EC2 region is invalid")
        elif region != expected_region:
            errors.append("EC2 region does not match the launch receipt")
        if (
            not isinstance(account_id, str)
            or not ACCOUNT_PATTERN.fullmatch(account_id)
        ):
            errors.append("EC2 account ID is invalid")
        elif account_id != expected_account_id:
            errors.append("EC2 account ID does not match the launch receipt")
        if (
            not isinstance(availability_zone, str)
            or not isinstance(region, str)
            or not availability_zone.startswith(region)
        ):
            errors.append("EC2 availability zone does not match its region")
        if not _valid_instance_ip(identity.get("privateIp")):
            errors.append("EC2 instance IP identity is invalid")
        version = identity.get("version")
        if not isinstance(version, str) or not DATE_PATTERN.fullmatch(version):
            errors.append("EC2 identity document version is invalid")

    _validate_nvidia_smi(record.get("nvidia_smi"), errors)
    _validate_bootstrap(
        record.get("bootstrap"),
        container_contract=container_contract,
        errors=errors,
    )
    _validate_docker(
        record,
        source_commit=source_commit,
        container_contract=container_contract,
        errors=errors,
    )
    return {
        "format": "colpali-triton-phase11-host-validation-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "expected": {
            "host_architecture": ami_contract["architecture"],
            "instance_types": sorted(allowed),
            "instance_id": expected_instance_id,
            "instance_type": expected_instance_type,
            "region": expected_region,
            "ami_id": expected_ami_id,
            "account_id": expected_account_id,
            "gpu": "NVIDIA L4",
            "compute_capability": "8.9",
            "container_platform": contract["platform"],
            "container_base_reference": container_contract["base_reference"],
            "torch_version": container_contract["torch_version"],
            "cuda_version": container_contract["cuda_version"],
            "triton_version": container_contract["triton_version"],
            "source_commit": source_commit,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--expected-instance-id", required=True)
    parser.add_argument("--expected-instance-type", required=True)
    parser.add_argument("--expected-region", required=True)
    parser.add_argument("--expected-ami-id", required=True)
    parser.add_argument("--expected-account-id", required=True)
    parser.add_argument(
        "--bootstrap-environment",
        type=Path,
        default=Path(
            "/var/lib/colpali-triton/bootstrap-environment.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract, contract_evidence = _load_environment_contract()
    record: dict[str, Any] = {
        "format": "colpali-triton-phase11-host-metadata-v1",
        "captured_at": _utc_now(),
        "source_commit": args.source_commit,
        "run_id": args.run_id,
        "container_image_name": args.image_name,
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "ec2": _imds_identity(),
        "nvidia_smi": _command(
            (
                "nvidia-smi",
                (
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version,"
                    "memory.total,compute_cap,pstate,"
                    "clocks.current.sm,clocks.current.memory,power.limit"
                ),
                "--format=csv,noheader",
            )
        ),
        "docker_version": _command(
            ("docker", "version", "--format", "{{json .}}")
        ),
        "docker_image": _command(
            ("docker", "image", "inspect", args.image_name)
        ),
        "bootstrap": _bootstrap_evidence(args.bootstrap_environment),
        "environment_contract": contract_evidence,
    }
    if contract is None:
        record["validation"] = {
            "format": "colpali-triton-phase11-host-validation-v1",
            "status": "failed",
            "errors": [
                "committed environment contract is unavailable or invalid"
            ],
            "expected": {},
        }
    else:
        record["validation"] = validate_host_evidence(
            record,
            contract,
            expected_instance_id=args.expected_instance_id,
            expected_instance_type=args.expected_instance_type,
            expected_region=args.expected_region,
            expected_ami_id=args.expected_ami_id,
            expected_account_id=args.expected_account_id,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if record["validation"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
