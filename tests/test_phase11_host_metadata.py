from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "infra" / "aws" / "phase11_host_metadata.py"
SOURCE_COMMIT = "a" * 40
IMAGE_NAME = "colpali-triton:phase11-test"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "phase11_host_metadata_tested",
        MODULE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host_metadata = _load_module()


def _contract() -> dict:
    contract, evidence = host_metadata._load_environment_contract()
    assert evidence["available"] is True
    assert contract is not None
    return contract


def _successful_command(stdout: str) -> dict:
    return {
        "command": ["example"],
        "available": True,
        "returncode": 0,
        "stdout": stdout,
        "stderr": "",
    }


def _valid_record() -> dict:
    container_contract = _contract()["container"]
    image = [
        {
            "Id": "sha256:" + "b" * 64,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {
                "Env": [
                    "PATH=/usr/local/bin:/usr/bin",
                    f"COLPALI_SOURCE_COMMIT={SOURCE_COMMIT}",
                ],
                "Labels": {
                    "org.opencontainers.image.revision": SOURCE_COMMIT,
                    "org.opencontainers.image.source": "local-worktree",
                    "org.opencontainers.image.base.name": (
                        container_contract["base_tag"]
                    ),
                    "org.opencontainers.image.base.digest": (
                        container_contract["base_manifest_digest"]
                    ),
                },
            },
        }
    ]
    bootstrap = {
        "base_image": container_contract["base_reference"],
        "container": {
            "architecture": "x86_64",
            "torch": "2.6.0+cu124",
            "cuda_build": "12.4",
            "triton": "3.2.0",
            "cuda_available": True,
            "gpu_name": "NVIDIA L4",
            "gpu_memory_bytes": 23 * 1024**3,
        },
        "docker_version": "29.5.3",
        "nvidia_container_toolkit": "NVIDIA Container Toolkit 1.19.1",
        "nvidia_smi": "NVIDIA L4, GPU-example",
    }
    return {
        "source_commit": SOURCE_COMMIT,
        "host": {"architecture": "x86_64"},
        "ec2": {
            "available": True,
            "identity_document": {
                "accountId": "123456789012",
                "architecture": "x86_64",
                "availabilityZone": "us-west-2a",
                "imageId": "ami-0123456789abcdef0",
                "instanceId": "i-0123456789abcdef0",
                "instanceType": "g6.xlarge",
                "privateIp": "10.0.0.5",
                "region": "us-west-2",
                "version": "2017-09-30",
            },
        },
        "nvidia_smi": _successful_command(
            "0, NVIDIA L4, GPU-12345678-abcd-1234-abcd-1234567890ab, "
            "00000000:00:1E.0, 580.159.04, 23034 MiB, 8.9, P8, "
            "300 MHz, 405 MHz, 72.00 W"
        ),
        "docker_version": _successful_command(
            json.dumps(
                {
                    "Client": {"Version": "29.5.3"},
                    "Server": {"Version": "29.5.3"},
                }
            )
        ),
        "docker_image": _successful_command(json.dumps(image)),
        "bootstrap": {
            "available": True,
            "path": "/var/lib/colpali-triton/bootstrap-environment.json",
            "sha256": "c" * 64,
            "record": bootstrap,
        },
    }


def test_committed_environment_contract_is_strict_and_loadable() -> None:
    contract, evidence = host_metadata._load_environment_contract()

    assert evidence["available"] is True
    assert len(evidence["sha256"]) == 64
    assert contract is not None
    assert set(contract["allowed_instance_types"]) == {
        "g6.xlarge",
        "g6.2xlarge",
    }
    assert contract["platform"] == "linux/amd64"


def test_exact_g6_l4_host_and_container_evidence_passes() -> None:
    result = host_metadata.validate_host_evidence(
        _valid_record(),
        _contract(),
    )

    assert result["status"] == "passed"
    assert result["errors"] == []
    assert result["expected"]["instance_types"] == [
        "g6.2xlarge",
        "g6.xlarge",
    ]
    assert result["expected"]["source_commit"] == SOURCE_COMMIT


def test_new_region_prefix_and_non_rfc1918_vpc_address_pass() -> None:
    record = _valid_record()
    identity = record["ec2"]["identity_document"]
    identity["region"] = "eusc-de-east-1"
    identity["availabilityZone"] = "eusc-de-east-1a"
    identity["privateIp"] = "100.64.0.10"

    result = host_metadata.validate_host_evidence(record, _contract())

    assert result["status"] == "passed"
    assert result["errors"] == []


@pytest.mark.parametrize("instance_type", ("g6.xlarge", "g6.2xlarge"))
def test_both_scoped_g6_instance_types_pass(instance_type: str) -> None:
    record = _valid_record()
    record["ec2"]["identity_document"]["instanceType"] = instance_type

    result = host_metadata.validate_host_evidence(record, _contract())

    assert result["status"] == "passed"


def _imds_missing(record: dict) -> None:
    record["ec2"] = {"available": False, "error": "IMDS unavailable"}


def _wrong_host_architecture(record: dict) -> None:
    record["host"]["architecture"] = "arm64"


def _wrong_instance_type(record: dict) -> None:
    record["ec2"]["identity_document"]["instanceType"] = "g5.xlarge"


def _invalid_instance_identity(record: dict) -> None:
    record["ec2"]["identity_document"]["instanceId"] = "not-an-instance"
    record["ec2"]["identity_document"]["imageId"] = "not-an-ami"
    record["ec2"]["identity_document"]["region"] = "west"


def _failed_nvidia_query(record: dict) -> None:
    record["nvidia_smi"] = {
        "available": False,
        "returncode": 1,
        "stdout": "",
        "stderr": "driver error",
    }


def _failed_docker_inspection(record: dict) -> None:
    record["docker_image"] = {
        "available": False,
        "returncode": 1,
        "stdout": "",
        "stderr": "image absent",
    }


def _wrong_bootstrap_base(record: dict) -> None:
    record["bootstrap"]["record"]["base_image"] = "unpinned:latest"


def _wrong_image_revision(record: dict) -> None:
    image = json.loads(record["docker_image"]["stdout"])
    image[0]["Config"]["Labels"][
        "org.opencontainers.image.revision"
    ] = "d" * 40
    record["docker_image"]["stdout"] = json.dumps(image)


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    (
        (_imds_missing, "EC2 IMDSv2 identity evidence is unavailable"),
        (_wrong_host_architecture, "host architecture is not x86_64"),
        (_wrong_instance_type, "outside the scoped G6 types"),
        (_invalid_instance_identity, "EC2 instance ID is invalid"),
        (_failed_nvidia_query, "nvidia-smi query did not complete"),
        (_failed_docker_inspection, "Docker image inspection did not succeed"),
        (_wrong_bootstrap_base, "bootstrap base image does not match"),
        (
            _wrong_image_revision,
            "org.opencontainers.image.revision does not match",
        ),
    ),
)
def test_host_gate_rejects_missing_or_mismatched_evidence(
    mutate,
    expected_error: str,
) -> None:
    record = deepcopy(_valid_record())
    mutate(record)

    result = host_metadata.validate_host_evidence(record, _contract())

    assert result["status"] == "failed"
    assert any(expected_error in error for error in result["errors"])


@pytest.mark.parametrize(
    "label",
    (
        "org.opencontainers.image.revision",
        "org.opencontainers.image.source",
        "org.opencontainers.image.base.name",
        "org.opencontainers.image.base.digest",
    ),
)
def test_every_required_docker_provenance_label_is_enforced(
    label: str,
) -> None:
    record = _valid_record()
    image = json.loads(record["docker_image"]["stdout"])
    image[0]["Config"]["Labels"][label] = "mismatch"
    record["docker_image"]["stdout"] = json.dumps(image)

    result = host_metadata.validate_host_evidence(record, _contract())

    assert result["status"] == "failed"
    assert f"Docker image label {label} does not match" in result["errors"]


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("architecture", "aarch64", "container architecture"),
        ("torch", "2.5.0+cu124", "PyTorch version"),
        ("cuda_build", "12.6", "CUDA build"),
        ("triton", "3.1.0", "Triton version"),
        ("cuda_available", False, "did not have CUDA"),
        ("gpu_name", "NVIDIA A10G", "GPU is not NVIDIA L4"),
        ("gpu_memory_bytes", 8 * 1024**3, "outside the full-L4 range"),
    ),
)
def test_every_pinned_bootstrap_container_fact_is_enforced(
    field: str,
    value: object,
    expected_error: str,
) -> None:
    record = _valid_record()
    record["bootstrap"]["record"]["container"][field] = value

    result = host_metadata.validate_host_evidence(record, _contract())

    assert result["status"] == "failed"
    assert any(expected_error in error for error in result["errors"])


def test_invalid_bootstrap_json_is_preserved_as_failed_evidence(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text('{"base_image": "first", "base_image": "second"}')

    evidence = host_metadata._bootstrap_evidence(bootstrap)

    assert evidence["available"] is False
    assert len(evidence["sha256"]) == 64
    assert "duplicate JSON key" in evidence["error"]


def test_main_always_writes_failed_evidence_before_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "host_metadata.json"
    missing_bootstrap = tmp_path / "missing-bootstrap.json"
    monkeypatch.setattr(
        host_metadata,
        "_parse_args",
        lambda: SimpleNamespace(
            output=output,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            image_name=IMAGE_NAME,
            bootstrap_environment=missing_bootstrap,
        ),
    )
    monkeypatch.setattr(
        host_metadata,
        "_imds_identity",
        lambda: {"available": False, "error": "IMDS unavailable"},
    )
    monkeypatch.setattr(
        host_metadata,
        "_command",
        lambda command: {
            "command": list(command),
            "available": False,
            "error": "command unavailable",
        },
    )
    monkeypatch.setattr(host_metadata.platform, "machine", lambda: "x86_64")

    exit_code = host_metadata.main()

    assert exit_code == 1
    artifact = json.loads(output.read_text())
    assert artifact["validation"]["status"] == "failed"
    assert artifact["validation"]["errors"]
    assert artifact["bootstrap"]["available"] is False
    assert artifact["environment_contract"]["available"] is True
