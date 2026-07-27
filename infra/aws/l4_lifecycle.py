#!/usr/bin/env python3
"""Guarded EC2 lifecycle commands for the Phase 10 NVIDIA L4 host."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("l4_environment.json")
LAUNCH_ACKNOWLEDGEMENT = (
    "I UNDERSTAND THIS CREATES A BILLABLE EC2 INSTANCE"
)
REQUIRED_TAGS = {
    "Project": "colpali-triton",
    "Phase": "10",
    "ManagedBy": "infra-aws-l4-lifecycle",
}
AMI_PATTERN = re.compile(r"^ami-[0-9a-f]{8,17}$")
SUBNET_PATTERN = re.compile(r"^subnet-[0-9a-f]{8,17}$")
SECURITY_GROUP_PATTERN = re.compile(r"^sg-[0-9a-f]{8,17}$")
INSTANCE_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")
REGION_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+-\d+$"
)
KEY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_+=,.@-]{1,255}$")
CLIENT_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
TAG_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9 _.:/=+@-]{1,128}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LOCAL_GIT_TIMEOUT_SECONDS = 30


class LifecycleError(RuntimeError):
    """A user-facing lifecycle validation or AWS command failure."""


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LifecycleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_environment_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot load environment config: {exc}") from exc

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
    if set(raw) != required:
        raise LifecycleError(
            "environment config keys differ from the Phase 10 contract"
        )
    if raw["schema_version"] != 1:
        raise LifecycleError("unsupported environment schema_version")
    if raw["platform"] != "linux/amd64":
        raise LifecycleError("Phase 10 requires platform linux/amd64")
    if raw["tags"] != REQUIRED_TAGS:
        raise LifecycleError("managed resource tags differ from the contract")
    if raw["recommended_instance_type"] not in raw[
        "allowed_instance_types"
    ]:
        raise LifecycleError("recommended instance type is not allowed")
    return raw


def canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LaunchSpec:
    region: str
    subnet_id: str
    security_group_id: str
    key_name: str
    instance_type: str
    root_volume_gib: int
    associate_public_ip: bool
    name: str
    profile: str | None
    client_token: str | None


def _validate_identifier(
    name: str,
    value: str,
    pattern: re.Pattern[str],
) -> str:
    if not pattern.fullmatch(value):
        raise LifecycleError(f"invalid {name}: {value!r}")
    return value


def launch_spec_from_args(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> LaunchSpec:
    region = _validate_identifier("AWS region", args.region, REGION_PATTERN)
    subnet_id = _validate_identifier(
        "subnet ID", args.subnet_id, SUBNET_PATTERN
    )
    security_group_id = _validate_identifier(
        "security group ID",
        args.security_group_id,
        SECURITY_GROUP_PATTERN,
    )
    if not KEY_NAME_PATTERN.fullmatch(args.key_name):
        raise LifecycleError(f"invalid EC2 key name: {args.key_name!r}")
    if args.instance_type not in config["allowed_instance_types"]:
        choices = ", ".join(config["allowed_instance_types"])
        raise LifecycleError(
            f"instance type must be one of the scoped L4 types: {choices}"
        )
    if not 50 <= args.root_volume_gib <= 1000:
        raise LifecycleError("root volume must be between 50 and 1000 GiB")
    if not TAG_VALUE_PATTERN.fullmatch(args.name):
        raise LifecycleError(
            "instance name contains unsupported characters or is too long"
        )
    if args.client_token and not CLIENT_TOKEN_PATTERN.fullmatch(
        args.client_token
    ):
        raise LifecycleError(
            "client token must be 1-64 safe ASCII characters"
        )
    return LaunchSpec(
        region=region,
        subnet_id=subnet_id,
        security_group_id=security_group_id,
        key_name=args.key_name,
        instance_type=args.instance_type,
        root_volume_gib=args.root_volume_gib,
        associate_public_ip=args.associate_public_ip,
        name=args.name,
        profile=args.profile,
        client_token=args.client_token,
    )


def _aws_prefix(profile: str | None) -> list[str]:
    command = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    return command


def _aws_suffix(region: str) -> list[str]:
    return ["--region", region, "--output", "json"]


def build_resolve_ami_command(
    spec: LaunchSpec,
    config: Mapping[str, Any],
) -> list[str]:
    return [
        *_aws_prefix(spec.profile),
        "ssm",
        "get-parameter",
        "--name",
        config["ami"]["ssm_parameter"],
        "--query",
        "Parameter.Value",
        "--output",
        "text",
        "--region",
        spec.region,
    ]


def build_describe_ami_command(
    spec: LaunchSpec,
    ami_id: str,
) -> list[str]:
    return [
        *_aws_prefix(spec.profile),
        "ec2",
        "describe-images",
        "--image-ids",
        ami_id,
        "--owners",
        "amazon",
        *_aws_suffix(spec.region),
    ]


def _default_client_token(
    spec: LaunchSpec,
    ami_id: str,
    config_fingerprint: str,
) -> str:
    stable = {
        "region": spec.region,
        "subnet_id": spec.subnet_id,
        "security_group_id": spec.security_group_id,
        "key_name": spec.key_name,
        "instance_type": spec.instance_type,
        "root_volume_gib": spec.root_volume_gib,
        "associate_public_ip": spec.associate_public_ip,
        "name": spec.name,
        "ami_id": ami_id,
        "config_fingerprint": config_fingerprint,
    }
    return "colpali-p10-" + canonical_fingerprint(stable)[:32]


def build_launch_command(
    spec: LaunchSpec,
    config: Mapping[str, Any],
    *,
    ami_id: str,
    dry_run: bool,
) -> list[str]:
    _validate_identifier("AMI ID", ami_id, AMI_PATTERN)
    config_fingerprint = canonical_fingerprint(config)
    client_token = spec.client_token or _default_client_token(
        spec,
        ami_id,
        config_fingerprint,
    )
    root = config["root_volume"]
    block_device_mappings = [
        {
            "DeviceName": root["device_name"],
            "Ebs": {
                "DeleteOnTermination": root["delete_on_termination"],
                "Encrypted": root["encrypted"],
                "VolumeSize": spec.root_volume_gib,
                "VolumeType": root["type"],
            },
        }
    ]
    tags = [
        {"Key": key, "Value": value}
        for key, value in sorted(
            {**config["tags"], "Name": spec.name}.items()
        )
    ]
    tag_specifications = [
        {"ResourceType": resource_type, "Tags": tags}
        for resource_type in ("instance", "volume")
    ]
    user_data = (PROJECT_ROOT / config["bootstrap"]["user_data_path"]).resolve()
    command = [
        *_aws_prefix(spec.profile),
        "ec2",
        "run-instances",
        "--image-id",
        ami_id,
        "--count",
        "1",
        "--instance-type",
        spec.instance_type,
        "--key-name",
        spec.key_name,
        "--subnet-id",
        spec.subnet_id,
        "--security-group-ids",
        spec.security_group_id,
        "--block-device-mappings",
        json.dumps(block_device_mappings, separators=(",", ":")),
        "--tag-specifications",
        *[
            json.dumps(value, separators=(",", ":"))
            for value in tag_specifications
        ],
        "--metadata-options",
        (
            "HttpTokens=required,HttpPutResponseHopLimit=1,"
            "HttpEndpoint=enabled,InstanceMetadataTags=disabled"
        ),
        "--instance-initiated-shutdown-behavior",
        "stop",
        "--enable-api-termination",
        "--ebs-optimized",
        "--client-token",
        client_token,
        "--user-data",
        f"file://{user_data}",
        (
            "--associate-public-ip-address"
            if spec.associate_public_ip
            else "--no-associate-public-ip-address"
        ),
        *_aws_suffix(spec.region),
        "--dry-run" if dry_run else "--no-dry-run",
    ]
    return command


def build_status_command(
    *,
    region: str,
    instance_id: str,
    profile: str | None,
) -> list[str]:
    return [
        *_aws_prefix(profile),
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
        *_aws_suffix(region),
    ]


def build_terminate_command(
    *,
    region: str,
    instance_id: str,
    profile: str | None,
    dry_run: bool,
) -> list[str]:
    return [
        *_aws_prefix(profile),
        "ec2",
        "terminate-instances",
        "--instance-ids",
        instance_id,
        *_aws_suffix(region),
        "--dry-run" if dry_run else "--no-dry-run",
    ]


def build_wait_terminated_command(
    *,
    region: str,
    instance_id: str,
    profile: str | None,
) -> list[str]:
    return [
        *_aws_prefix(profile),
        "ec2",
        "wait",
        "instance-terminated",
        "--instance-ids",
        instance_id,
        "--region",
        region,
    ]


def _run_command(
    command: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["AWS_PAGER"] = ""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError as exc:
        raise LifecycleError(f"cannot execute {command[0]!r}: {exc}") from exc
    if check and completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise LifecycleError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}\n{message}"
        )
    return completed


def _parse_json_output(
    completed: subprocess.CompletedProcess[str],
    *,
    operation: str,
) -> dict[str, Any]:
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LifecycleError(
            f"{operation} returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{operation} returned a non-object JSON value")
    return value


def _resolve_ami(
    spec: LaunchSpec,
    config: Mapping[str, Any],
) -> str:
    completed = _run_command(build_resolve_ami_command(spec, config))
    ami_id = completed.stdout.strip()
    return _validate_identifier("resolved AMI ID", ami_id, AMI_PATTERN)


def _describe_and_validate_ami(
    spec: LaunchSpec,
    config: Mapping[str, Any],
    ami_id: str,
) -> dict[str, Any]:
    completed = _run_command(build_describe_ami_command(spec, ami_id))
    response = _parse_json_output(completed, operation="describe-images")
    images = response.get("Images")
    if not isinstance(images, list) or len(images) != 1:
        raise LifecycleError(
            "AMI validation did not return exactly one Amazon-owned image"
        )
    image = images[0]
    expected = config["ami"]
    if image.get("ImageId") != ami_id:
        raise LifecycleError("AMI validation returned the wrong image ID")
    if image.get("Architecture") != expected["architecture"]:
        raise LifecycleError("AMI architecture is not x86_64")
    if image.get("State") != "available":
        raise LifecycleError("AMI is not available")
    if not str(image.get("Name", "")).startswith(expected["name_prefix"]):
        raise LifecycleError(
            "AMI is not the approved Ubuntu 22.04 GPU Base DLAMI"
        )
    if image.get("RootDeviceName") != config["root_volume"]["device_name"]:
        raise LifecycleError(
            "AMI root device differs from the pinned block-device contract"
        )
    return {
        "image_id": image["ImageId"],
        "name": image["Name"],
        "creation_date": image.get("CreationDate"),
        "architecture": image["Architecture"],
        "root_device_name": image["RootDeviceName"],
        "owner_id": image.get("OwnerId"),
    }


def _validate_aws_dry_run(
    command: Sequence[str],
) -> None:
    completed = _run_command(command, check=False)
    combined = f"{completed.stdout}\n{completed.stderr}"
    if "DryRunOperation" not in combined:
        raise LifecycleError(
            "AWS permission dry-run did not return DryRunOperation: "
            f"{combined.strip()}"
        )


def _instance_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    reservations = response.get("Reservations")
    if not isinstance(reservations, list) or len(reservations) != 1:
        raise LifecycleError("describe-instances returned no single reservation")
    instances = reservations[0].get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise LifecycleError("describe-instances returned no single instance")
    instance = instances[0]
    if not isinstance(instance, dict):
        raise LifecycleError("describe-instances returned an invalid instance")
    return instance


def _tags_from_instance(instance: Mapping[str, Any]) -> dict[str, str]:
    tags = instance.get("Tags", [])
    if not isinstance(tags, list):
        return {}
    return {
        str(tag.get("Key")): str(tag.get("Value"))
        for tag in tags
        if isinstance(tag, dict) and "Key" in tag and "Value" in tag
    }


def _ensure_receipt_target(path: Path) -> None:
    if path.exists():
        raise LifecycleError(f"receipt already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise LifecycleError(
            f"receipt parent is not a directory: {path.parent}"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise LifecycleError(
                f"receipt appeared concurrently and was not overwritten: {path}"
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _require_clean_source_commit() -> str:
    """Return HEAD only when the exact launch source is committed and clean."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=LOCAL_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise LifecycleError(
            "launch source Git inspection exceeded its 30-second bound"
        ) from exc
    except OSError as exc:
        raise LifecycleError(
            f"cannot inspect the launch source with Git: {exc}"
        ) from exc
    commit = revision.stdout.strip()
    if revision.returncode != 0 or not COMMIT_PATTERN.fullmatch(commit):
        raise LifecycleError(
            "launch requires a resolvable 40-character Git HEAD"
        )
    if status.returncode != 0:
        raise LifecycleError("launch cannot inspect the Git worktree")
    if status.stdout:
        raise LifecycleError(
            "launch requires a clean worktree with every source change "
            "committed"
        )
    return commit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _offline_plan(
    spec: LaunchSpec,
    config: Mapping[str, Any],
    *,
    ami_id: str | None,
) -> dict[str, Any]:
    display_ami = ami_id or "ami-00000000000000000"
    launch_command = build_launch_command(
        spec,
        config,
        ami_id=display_ami,
        dry_run=False,
    )
    if ami_id is None:
        launch_command[launch_command.index(display_ami)] = (
            "<AMI_RESOLVED_FROM_SSM_AT_EXECUTION>"
        )
    return {
        "schema_version": 1,
        "operation": "offline-plan",
        "mutates_aws": False,
        "requires_aws_credentials": False,
        "config_fingerprint": canonical_fingerprint(config),
        "ami_id": ami_id,
        "ami_resolution_command": (
            None
            if ami_id
            else build_resolve_ami_command(spec, config)
        ),
        "launch_command": launch_command,
        "launch_command_shell": shlex.join(launch_command),
        "network_defaults_used": False,
        "public_ip_requested": spec.associate_public_ip,
        "instance_type_contract": config["allowed_instance_types"][
            spec.instance_type
        ],
        "warning": (
            "This plan made no AWS calls. Launching requires --execute and "
            "the exact cost acknowledgement."
        ),
    }


def _handle_plan_or_launch(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    spec = launch_spec_from_args(args, config)
    ami_id = (
        _validate_identifier("AMI ID", args.ami_id, AMI_PATTERN)
        if args.ami_id
        else None
    )
    if args.command == "plan" or not args.execute:
        print(
            json.dumps(
                _offline_plan(spec, config, ami_id=ami_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.acknowledge_cost != LAUNCH_ACKNOWLEDGEMENT:
        raise LifecycleError(
            "launch refused: --acknowledge-cost must exactly equal "
            f"{LAUNCH_ACKNOWLEDGEMENT!r}"
        )
    source_commit = _require_clean_source_commit()
    user_data_path = (
        PROJECT_ROOT / config["bootstrap"]["user_data_path"]
    )
    user_data_sha256 = _file_sha256(user_data_path)
    if shutil.which("aws") is None:
        raise LifecycleError("AWS CLI is not installed")
    receipt_path = args.receipt.resolve()
    _ensure_receipt_target(receipt_path)

    resolved_ami = ami_id or _resolve_ami(spec, config)
    image_record = _describe_and_validate_ami(
        spec,
        config,
        resolved_ami,
    )
    dry_run_command = build_launch_command(
        spec,
        config,
        ami_id=resolved_ami,
        dry_run=True,
    )
    _validate_aws_dry_run(dry_run_command)
    if (
        _require_clean_source_commit() != source_commit
        or _file_sha256(user_data_path) != user_data_sha256
    ):
        raise LifecycleError(
            "launch source changed during AWS preparation; rerun the plan"
        )
    launch_command = build_launch_command(
        spec,
        config,
        ami_id=resolved_ami,
        dry_run=False,
    )
    response = _parse_json_output(
        _run_command(launch_command),
        operation="run-instances",
    )
    instances = response.get("Instances")
    if not isinstance(instances, list) or len(instances) != 1:
        raise LifecycleError(
            "run-instances succeeded but returned no single instance"
        )
    instance_id = str(instances[0].get("InstanceId", ""))
    _validate_identifier("launched instance ID", instance_id, INSTANCE_PATTERN)

    receipt = {
        "schema_version": 1,
        "operation": "launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": canonical_fingerprint(config),
        "source_commit": source_commit,
        "user_data_sha256": user_data_sha256,
        "resolved_ami": image_record,
        "instance_id": instance_id,
        "instance_type": spec.instance_type,
        "region": spec.region,
        "subnet_id": spec.subnet_id,
        "security_group_id": spec.security_group_id,
        "public_ip_requested": spec.associate_public_ip,
        "root_volume_gib": spec.root_volume_gib,
        "client_token": launch_command[
            launch_command.index("--client-token") + 1
        ],
        "container": config["container"],
        "bootstrap": config["bootstrap"],
        "tags": {**config["tags"], "Name": spec.name},
        "launch_response": response,
    }
    _write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _handle_status(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    region = _validate_identifier("AWS region", args.region, REGION_PATTERN)
    instance_id = _validate_identifier(
        "instance ID", args.instance_id, INSTANCE_PATTERN
    )
    response = _parse_json_output(
        _run_command(
            build_status_command(
                region=region,
                instance_id=instance_id,
                profile=args.profile,
            )
        ),
        operation="describe-instances",
    )
    instance = _instance_from_response(response)
    payload = {
        "instance_id": instance_id,
        "state": instance.get("State", {}).get("Name"),
        "instance_type": instance.get("InstanceType"),
        "image_id": instance.get("ImageId"),
        "public_ip_address": instance.get("PublicIpAddress"),
        "private_ip_address": instance.get("PrivateIpAddress"),
        "launch_time": instance.get("LaunchTime"),
        "tags": _tags_from_instance(instance),
        "bootstrap_ready_marker": config["bootstrap"]["ready_marker"],
        "bootstrap_environment_record": config["bootstrap"][
            "environment_record"
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _handle_terminate(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    region = _validate_identifier("AWS region", args.region, REGION_PATTERN)
    instance_id = _validate_identifier(
        "instance ID", args.instance_id, INSTANCE_PATTERN
    )
    planned_command = build_terminate_command(
        region=region,
        instance_id=instance_id,
        profile=args.profile,
        dry_run=False,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "operation": "offline-termination-plan",
                    "mutates_aws": False,
                    "requires_aws_credentials": False,
                    "instance_id": instance_id,
                    "command": planned_command,
                    "command_shell": shlex.join(planned_command),
                    "warning": (
                        "No AWS call was made. Execution first verifies the "
                        "exact instance and all managed tags."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.confirm_instance_id != instance_id:
        raise LifecycleError(
            "termination refused: --confirm-instance-id must exactly match "
            f"{instance_id}"
        )
    if shutil.which("aws") is None:
        raise LifecycleError("AWS CLI is not installed")
    receipt_path = args.receipt.resolve()
    _ensure_receipt_target(receipt_path)

    status_response = _parse_json_output(
        _run_command(
            build_status_command(
                region=region,
                instance_id=instance_id,
                profile=args.profile,
            )
        ),
        operation="describe-instances",
    )
    instance = _instance_from_response(status_response)
    if instance.get("InstanceId") != instance_id:
        raise LifecycleError("described instance ID does not match target")
    actual_tags = _tags_from_instance(instance)
    missing_or_changed = {
        key: {"expected": value, "actual": actual_tags.get(key)}
        for key, value in config["tags"].items()
        if actual_tags.get(key) != value
    }
    if missing_or_changed:
        raise LifecycleError(
            "termination refused: managed tags did not match: "
            f"{json.dumps(missing_or_changed, sort_keys=True)}"
        )

    dry_run_command = build_terminate_command(
        region=region,
        instance_id=instance_id,
        profile=args.profile,
        dry_run=True,
    )
    _validate_aws_dry_run(dry_run_command)
    response = _parse_json_output(
        _run_command(planned_command),
        operation="terminate-instances",
    )
    wait_command = build_wait_terminated_command(
        region=region,
        instance_id=instance_id,
        profile=args.profile,
    )
    wait_result = _run_command(wait_command, check=False)
    final_status_command = build_status_command(
        region=region,
        instance_id=instance_id,
        profile=args.profile,
    )
    final_status_result = _run_command(
        final_status_command,
        check=False,
    )
    final_status_response: dict[str, Any] | None = None
    final_state: str | None = None
    final_status_error: str | None = None
    if final_status_result.returncode == 0:
        try:
            final_status_response = _parse_json_output(
                final_status_result,
                operation="post-termination describe-instances",
            )
            final_instance = _instance_from_response(
                final_status_response
            )
            if final_instance.get("InstanceId") != instance_id:
                raise LifecycleError(
                    "post-termination instance ID does not match target"
                )
            final_state = final_instance.get("State", {}).get("Name")
        except LifecycleError as exc:
            final_status_error = str(exc)
    else:
        final_status_error = (
            final_status_result.stderr.strip()
            or final_status_result.stdout.strip()
            or "post-termination describe-instances failed"
        )
    confirmed_terminated = final_state == "terminated" or (
        wait_result.returncode == 0
        and final_status_response is None
    )
    receipt = {
        "schema_version": 1,
        "operation": "terminate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_id": instance_id,
        "region": region,
        "verified_tags": {
            key: actual_tags[key] for key in sorted(config["tags"])
        },
        "termination_request_response": response,
        "wait": {
            "command": wait_command,
            "returncode": wait_result.returncode,
            "stdout": wait_result.stdout.strip(),
            "stderr": wait_result.stderr.strip(),
        },
        "post_termination_status": {
            "command": final_status_command,
            "state": final_state,
            "error": final_status_error,
            "response": final_status_response,
        },
        "confirmed_terminated": confirmed_terminated,
    }
    _write_json_atomic(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not confirmed_terminated:
        raise LifecycleError(
            "AWS accepted the termination request, but termination was not "
            f"confirmed; inspect receipt {receipt_path}"
        )
    return 0


def _add_aws_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--profile",
        help="optional named AWS CLI profile; the default provider chain is used otherwise",
    )


def _add_launch_arguments(
    parser: argparse.ArgumentParser,
    config: Mapping[str, Any],
) -> None:
    _add_aws_identity_arguments(parser)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--key-name", required=True)
    parser.add_argument(
        "--instance-type",
        default=config["recommended_instance_type"],
        choices=sorted(config["allowed_instance_types"]),
    )
    parser.add_argument(
        "--root-volume-gib",
        type=int,
        default=config["root_volume"]["default_size_gib"],
    )
    parser.add_argument(
        "--ami-id",
        help=(
            "immutable Amazon-owned DLAMI ID; if omitted during execution, "
            "resolve the documented public SSM parameter and record the result"
        ),
    )
    parser.add_argument(
        "--associate-public-ip",
        action="store_true",
        help=(
            "explicitly request a public IPv4 address; the default explicitly "
            "disables one"
        ),
    )
    parser.add_argument(
        "--name",
        default="colpali-triton-phase10",
    )
    parser.add_argument(
        "--client-token",
        help=(
            "optional EC2 idempotency token; a deterministic token is derived "
            "from the exact launch plan by default"
        ),
    )


def build_parser(
    config: Mapping[str, Any],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and manage one scoped AWS NVIDIA L4 benchmark host. "
            "Planning is entirely offline."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="print a complete offline launch plan and make no AWS calls",
    )
    _add_launch_arguments(plan, config)

    launch = subparsers.add_parser(
        "launch",
        help="print a plan by default; mutate AWS only with both safety flags",
    )
    _add_launch_arguments(launch, config)
    launch.add_argument("--execute", action="store_true")
    launch.add_argument(
        "--acknowledge-cost",
        default="",
        help=(
            "must exactly equal: "
            f"{LAUNCH_ACKNOWLEDGEMENT}"
        ),
    )
    launch.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "phase10"
        / "aws_launch_receipt.json",
    )

    status = subparsers.add_parser(
        "status",
        help="read current state for one exact instance ID",
    )
    _add_aws_identity_arguments(status)
    status.add_argument("--instance-id", required=True)

    terminate = subparsers.add_parser(
        "terminate",
        help=(
            "print an offline plan by default; execute only after tag and "
            "exact-ID verification"
        ),
    )
    _add_aws_identity_arguments(terminate)
    terminate.add_argument("--instance-id", required=True)
    terminate.add_argument("--execute", action="store_true")
    terminate.add_argument("--confirm-instance-id", default="")
    terminate.add_argument(
        "--receipt",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "phase10"
        / "aws_termination_receipt.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = load_environment_config()
        parser = build_parser(config)
        args = parser.parse_args(argv)
        if args.command in {"plan", "launch"}:
            if args.command == "plan":
                args.execute = False
            return _handle_plan_or_launch(args, config)
        if args.command == "status":
            return _handle_status(args, config)
        if args.command == "terminate":
            return _handle_terminate(args, config)
        raise LifecycleError(f"unsupported command: {args.command}")
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
