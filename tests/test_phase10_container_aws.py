from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE_PATH = PROJECT_ROOT / "infra" / "aws" / "l4_lifecycle.py"
CONFIG_PATH = PROJECT_ROOT / "infra" / "aws" / "l4_environment.json"
DOCKERFILE_PATH = PROJECT_ROOT / "docker" / "Dockerfile.cuda"
EXPECTED_BASE_DIGEST = (
    "sha256:"
    "0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56"
)
EXPECTED_CONFIG_FINGERPRINT = (
    "b06d77f646da2d73eed72f171908f2232aae9b1e3c1ec9c3badf4b124817c6be"
)


def _load_lifecycle_module():
    spec = importlib.util.spec_from_file_location(
        "phase10_l4_lifecycle",
        LIFECYCLE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lifecycle = _load_lifecycle_module()


def _launch_spec(**overrides):
    values = {
        "region": "us-west-2",
        "subnet_id": "subnet-0123456789abcdef0",
        "security_group_id": "sg-0123456789abcdef0",
        "key_name": "colpali-benchmark",
        "instance_type": "g6.xlarge",
        "root_volume_gib": 100,
        "associate_public_ip": False,
        "name": "colpali-triton-phase10",
        "profile": None,
        "client_token": None,
    }
    values.update(overrides)
    return lifecycle.LaunchSpec(**values)


def _offline_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LIFECYCLE_PATH), *arguments],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "",
        },
    )


def test_environment_config_is_pinned_and_canonical() -> None:
    config = lifecycle.load_environment_config()

    assert lifecycle.canonical_fingerprint(config) == (
        EXPECTED_CONFIG_FINGERPRINT
    )
    assert set(config["allowed_instance_types"]) == {
        "g6.xlarge",
        "g6.2xlarge",
    }
    assert config["platform"] == "linux/amd64"
    assert config["container"]["base_manifest_digest"] == (
        EXPECTED_BASE_DIGEST
    )
    assert config["ami"]["ssm_parameter"] == (
        "/aws/service/deeplearning/ami/x86_64/"
        "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
    )
    assert config["tags"] == lifecycle.REQUIRED_TAGS


def test_dockerfile_uses_exact_amd64_base_and_asserts_stack() -> None:
    dockerfile = DOCKERFILE_PATH.read_text()
    config = json.loads(CONFIG_PATH.read_text())

    assert "FROM --platform=linux/amd64" in dockerfile
    assert config["container"]["base_reference"] in dockerfile
    assert f'base.digest="{EXPECTED_BASE_DIGEST}"' in dockerfile
    assert 'torch.version.cuda == "12.4"' in dockerfile
    assert 'triton.__version__ == "3.2.0"' in dockerfile
    assert "--no-build-isolation ." in dockerfile
    assert "COPY . ." in dockerfile


def test_bootstrap_and_container_scripts_preserve_platform_and_digest() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    user_data = (
        PROJECT_ROOT / config["bootstrap"]["user_data_path"]
    ).read_text()
    build_script = (PROJECT_ROOT / "docker" / "build_cuda.sh").read_text()
    check_script = (
        PROJECT_ROOT / "docker" / "run_cuda_checks.sh"
    ).read_text()

    assert config["container"]["base_reference"] in user_data
    assert "bootstrap.ready" in user_data
    assert "nvidia-ctk runtime configure --runtime=docker" in user_data
    assert "torch.cuda.is_available()" in user_data
    assert "--platform linux/amd64" in build_script
    assert "linux/amd64" in check_script
    assert "--gpus" in check_script


def test_offline_plan_never_needs_aws_or_credentials() -> None:
    completed = _offline_command(
        "plan",
        "--region",
        "us-west-2",
        "--subnet-id",
        "subnet-0123456789abcdef0",
        "--security-group-id",
        "sg-0123456789abcdef0",
        "--key-name",
        "colpali-benchmark",
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["operation"] == "offline-plan"
    assert plan["mutates_aws"] is False
    assert plan["requires_aws_credentials"] is False
    assert plan["network_defaults_used"] is False
    assert plan["public_ip_requested"] is False
    assert "<AMI_RESOLVED_FROM_SSM_AT_EXECUTION>" in (
        plan["launch_command"]
    )
    assert "--no-associate-public-ip-address" in plan["launch_command"]
    assert "--no-dry-run" in plan["launch_command"]


def test_launch_without_execute_is_the_same_offline_safety_boundary() -> None:
    completed = _offline_command(
        "launch",
        "--region",
        "us-west-2",
        "--subnet-id",
        "subnet-0123456789abcdef0",
        "--security-group-id",
        "sg-0123456789abcdef0",
        "--key-name",
        "colpali-benchmark",
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["mutates_aws"] is False


def test_execute_requires_exact_cost_acknowledgement() -> None:
    completed = _offline_command(
        "launch",
        "--region",
        "us-west-2",
        "--subnet-id",
        "subnet-0123456789abcdef0",
        "--security-group-id",
        "sg-0123456789abcdef0",
        "--key-name",
        "colpali-benchmark",
        "--execute",
        "--acknowledge-cost",
        "yes",
    )

    assert completed.returncode == 2
    assert "--acknowledge-cost must exactly equal" in completed.stderr


def test_launch_command_has_explicit_security_and_cost_controls() -> None:
    config = lifecycle.load_environment_config()
    spec = _launch_spec()
    command = lifecycle.build_launch_command(
        spec,
        config,
        ami_id="ami-0123456789abcdef0",
        dry_run=False,
    )

    assert command.count("--count") == 1
    assert command[command.index("--count") + 1] == "1"
    assert command[command.index("--instance-type") + 1] == "g6.xlarge"
    assert command[command.index("--subnet-id") + 1] == spec.subnet_id
    assert (
        command[command.index("--security-group-ids") + 1]
        == spec.security_group_id
    )
    assert "--no-associate-public-ip-address" in command
    assert "--enable-api-termination" in command
    assert "--no-dry-run" in command
    assert "HttpTokens=required" in (
        command[command.index("--metadata-options") + 1]
    )
    token = command[command.index("--client-token") + 1]
    assert token.startswith("colpali-p10-")
    assert len(token) <= 64

    mappings = json.loads(
        command[command.index("--block-device-mappings") + 1]
    )
    assert mappings == [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "DeleteOnTermination": True,
                "Encrypted": True,
                "VolumeSize": 100,
                "VolumeType": "gp3",
            },
        }
    ]
    tag_index = command.index("--tag-specifications")
    instance_tags = json.loads(command[tag_index + 1])
    volume_tags = json.loads(command[tag_index + 2])
    assert instance_tags["ResourceType"] == "instance"
    assert volume_tags["ResourceType"] == "volume"
    assert instance_tags["Tags"] == volume_tags["Tags"]


def test_client_token_is_stable_and_changes_with_plan() -> None:
    config = lifecycle.load_environment_config()
    base = lifecycle.build_launch_command(
        _launch_spec(),
        config,
        ami_id="ami-0123456789abcdef0",
        dry_run=False,
    )
    same = lifecycle.build_launch_command(
        _launch_spec(),
        config,
        ami_id="ami-0123456789abcdef0",
        dry_run=False,
    )
    changed = lifecycle.build_launch_command(
        _launch_spec(root_volume_gib=120),
        config,
        ami_id="ami-0123456789abcdef0",
        dry_run=False,
    )

    def token(command: list[str]) -> str:
        return command[command.index("--client-token") + 1]

    assert token(base) == token(same)
    assert token(base) != token(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("region", "west", "AWS region"),
        ("subnet_id", "default", "subnet ID"),
        ("security_group_id", "default", "security group ID"),
        ("instance_type", "g5.xlarge", "scoped L4"),
        ("root_volume_gib", 49, "between 50 and 1000"),
        ("key_name", "not a key", "key name"),
    ),
)
def test_launch_spec_rejects_implicit_or_out_of_scope_values(
    field: str,
    value: object,
    message: str,
) -> None:
    config = lifecycle.load_environment_config()
    values = {
        "region": "us-west-2",
        "subnet_id": "subnet-0123456789abcdef0",
        "security_group_id": "sg-0123456789abcdef0",
        "key_name": "colpali-benchmark",
        "instance_type": "g6.xlarge",
        "root_volume_gib": 100,
        "associate_public_ip": False,
        "name": "colpali-triton-phase10",
        "profile": None,
        "client_token": None,
    }
    values[field] = value
    args = type("Args", (), values)

    with pytest.raises(lifecycle.LifecycleError, match=message):
        lifecycle.launch_spec_from_args(args, config)


def test_termination_default_is_offline_and_exact_id_is_required() -> None:
    planned = _offline_command(
        "terminate",
        "--region",
        "us-west-2",
        "--instance-id",
        "i-0123456789abcdef0",
    )
    mismatch = _offline_command(
        "terminate",
        "--region",
        "us-west-2",
        "--instance-id",
        "i-0123456789abcdef0",
        "--execute",
        "--confirm-instance-id",
        "i-11111111111111111",
    )

    assert planned.returncode == 0, planned.stderr
    plan = json.loads(planned.stdout)
    assert plan["mutates_aws"] is False
    assert plan["requires_aws_credentials"] is False
    assert "--no-dry-run" in plan["command"]
    assert mismatch.returncode == 2
    assert "--confirm-instance-id must exactly match" in mismatch.stderr


def test_receipt_write_is_non_overwriting(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    lifecycle._write_json_atomic(receipt, {"first": 1})

    with pytest.raises(lifecycle.LifecycleError, match="not overwritten"):
        lifecycle._write_json_atomic(receipt, {"second": 2})

    assert json.loads(receipt.read_text()) == {"first": 1}


def test_source_deployer_refuses_uncommitted_input_by_contract() -> None:
    deployer = (
        PROJECT_ROOT / "infra" / "aws" / "deploy_committed_source.sh"
    ).read_text()

    assert 'git -C "${PROJECT_ROOT}" status --porcelain' in deployer
    assert 'git -C "${PROJECT_ROOT}" archive --format=tar HEAD' in deployer
    assert "test ! -e" in deployer
    assert "bootstrap.ready" in deployer
