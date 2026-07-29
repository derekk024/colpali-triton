#!/usr/bin/env python3
"""Deploy, run, and retrieve one immutable Phase 11 NVIDIA result bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Mapping, Sequence

AWS_INFRA_DIRECTORY = Path(__file__).resolve().parent
if str(AWS_INFRA_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(AWS_INFRA_DIRECTORY))
from phase11_result_contract import validate_result_contract
from phase11_host_metadata import (
    _load_environment_contract as _load_host_environment_contract,
    validate_host_evidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAUNCH_RECEIPT = (
    PROJECT_ROOT / "artifacts" / "phase10" / "aws_launch_receipt.json"
)
RUN_ACKNOWLEDGEMENT = "RUN PHASE 11 ON THIS EXISTING INSTANCE"
HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)$"
)
USER_PATTERN = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RESERVED_RUN_ID_SUFFIXES = (
    ".control",
    ".tar.gz",
    ".tar.gz.sha256",
)
IMAGE_PATTERN = re.compile(
    r"^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$"
)
SAFE_ABSOLUTE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
AMI_PATTERN = re.compile(r"^ami-[0-9a-f]{8,17}$")
INSTANCE_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")
SUBNET_PATTERN = re.compile(r"^subnet-[0-9a-f]{8,17}$")
SECURITY_GROUP_PATTERN = re.compile(r"^sg-[0-9a-f]{8,17}$")
REGION_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+-\d+$"
)
ACCOUNT_PATTERN = re.compile(r"^\d{12}$")
MAX_BUNDLE_UNCOMPRESSED_BYTES = 10 * 1024**3
LOCAL_COMMAND_TIMEOUT_SECONDS = 60
SSH_COMMAND_TIMEOUT_SECONDS = 120
SOURCE_ARCHIVE_TIMEOUT_SECONDS = 120
SOURCE_UPLOAD_TIMEOUT_SECONDS = 20 * 60
ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS = 2 * 60 * 60
REMOTE_BUNDLE_TIMEOUT_SECONDS = 2 * 60 * 60
REMOTE_FINALIZER_TIMEOUT_SECONDS = 240
REMOTE_JOB_TIMEOUT_SECONDS = 4 * 60 * 60
REMOTE_JOB_OUTER_TIMEOUT_SECONDS = REMOTE_JOB_TIMEOUT_SECONDS + 10 * 60
REMOTE_POLL_TIMEOUT_SECONDS = REMOTE_JOB_OUTER_TIMEOUT_SECONDS + 10 * 60
REMOTE_POLL_INTERVAL_SECONDS = 10
REMOTE_RECONNECT_TIMEOUT_SECONDS = 15 * 60
SOURCE_MARKER_NAMES = {
    ".source_archive_sha256",
    ".source_commit",
    ".source_complete",
    ".source_content_sha256",
}
SOURCE_CONTENT_HASH_SCRIPT = r"""
from hashlib import sha256
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
markers = {
    ".source_archive_sha256",
    ".source_commit",
    ".source_complete",
    ".source_content_sha256",
}
digest = sha256()
paths = []
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    if relative in markers:
        continue
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise SystemExit(3)
    if path.is_file():
        paths.append((relative, path))
for relative, path in sorted(paths):
    content = path.read_bytes()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(content)).encode("ascii"))
    digest.update(b"\0")
    digest.update(content)
    digest.update(b"\0")
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(4)
print(actual)
""".strip()
MANDATORY_PHASE11_STEPS = (
    "command_contract",
    "container_build",
    "hardware_gate",
    "host_metadata",
    "cuda_tests",
    "cuda_assertion",
    "tuning_preflight",
    "tuning",
    "derive_winner",
    "benchmark",
    "profiling_status",
    "result_contract",
)
REQUIRED_CUDA_COUNTS = {
    "test_cuda_kernel_matches_vectorized_reference": 3,
    "test_cuda_kernel_handles_noncontiguous_inputs_and_empty_masks": 1,
    "test_cuda_kernel_propagates_active_nan_across_document_tiles": 1,
}


class WorkflowError(RuntimeError):
    """A safe, user-facing Phase 11 workflow failure."""


class SSHTransportError(WorkflowError):
    """An ambiguous SSH transport failure (OpenSSH exit status 255)."""


@dataclass(frozen=True)
class LaunchReceipt:
    path: Path
    sha256: str
    config_fingerprint: str
    instance_id: str
    instance_type: str
    region: str
    ami_id: str
    account_id: str

    @property
    def expected_host(self) -> dict[str, str]:
        return {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "region": self.region,
            "ami_id": self.ami_id,
            "account_id": self.account_id,
            "launch_receipt_sha256": self.sha256,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True)
class RemoteSpec:
    host: str
    user: str
    port: int
    identity_file: Path
    remote_source_base: str
    remote_result_base: str
    local_result_base: Path
    run_id: str
    source_commit: str
    image_name: str
    launch_receipt_path: Path
    launch_receipt_sha256: str | None
    launch_config_fingerprint: str | None
    expected_instance_id: str | None
    expected_instance_type: str | None
    expected_region: str | None
    expected_ami_id: str | None
    expected_account_id: str | None

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def remote_source_directory(self) -> str:
        return f"{self.remote_source_base}/{self.source_commit}"

    @property
    def remote_result_directory(self) -> str:
        return (
            f"{self.remote_result_base}/{self.source_commit}/{self.run_id}"
        )

    @property
    def local_result_directory(self) -> Path:
        return (
            self.local_result_base
            / self.source_commit
            / self.run_id
        )

    @property
    def remote_control_directory(self) -> str:
        return f"{self.remote_result_directory}.control"

    @property
    def remote_unit_name(self) -> str:
        run_digest = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()
        return (
            f"colpali-phase11-{self.source_commit[:12]}-{run_digest[:16]}"
        )

    @property
    def expected_host(self) -> dict[str, str]:
        values = {
            "instance_id": self.expected_instance_id,
            "instance_type": self.expected_instance_type,
            "region": self.expected_region,
            "ami_id": self.expected_ami_id,
            "account_id": self.expected_account_id,
            "launch_receipt_sha256": self.launch_receipt_sha256,
            "config_fingerprint": self.launch_config_fingerprint,
        }
        if not all(isinstance(value, str) and value for value in values.values()):
            raise WorkflowError(
                "a valid Phase 10 launch receipt is required for execution"
            )
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class SourceArchive:
    path: Path
    archive_sha256: str
    content_sha256: str


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_string(
    record: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise WorkflowError(f"launch receipt {label} is missing or invalid")
    return value


def _load_launch_receipt(
    path: Path,
    *,
    source_commit: str,
) -> LaunchReceipt:
    """Strictly bind a lifecycle launch receipt to the current source."""

    resolved_path = path.expanduser().resolve()
    try:
        raw = resolved_path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise WorkflowError(
            f"cannot read a valid launch receipt at {resolved_path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise WorkflowError("launch receipt must be a JSON object")
    required_keys = {
        "schema_version",
        "operation",
        "created_at_utc",
        "config_fingerprint",
        "source_commit",
        "user_data_sha256",
        "resolved_ami",
        "instance_id",
        "instance_type",
        "region",
        "subnet_id",
        "security_group_id",
        "public_ip_requested",
        "root_volume_gib",
        "client_token",
        "container",
        "bootstrap",
        "tags",
        "launch_response",
    }
    if set(value) != required_keys:
        raise WorkflowError("launch receipt keys differ from schema version 1")
    if (
        value.get("schema_version") != 1
        or isinstance(value.get("schema_version"), bool)
        or value.get("operation") != "launch"
    ):
        raise WorkflowError("launch receipt operation/schema is unsupported")
    _required_string(value, "created_at_utc", label="creation time")

    contract, evidence = _load_host_environment_contract()
    if contract is None or evidence.get("available") is not True:
        raise WorkflowError(
            "committed L4 environment contract is unavailable or invalid"
        )
    expected_fingerprint = _canonical_fingerprint(contract)
    config_fingerprint = _required_string(
        value,
        "config_fingerprint",
        label="configuration fingerprint",
    )
    if (
        not HASH_PATTERN.fullmatch(config_fingerprint)
        or config_fingerprint != expected_fingerprint
    ):
        raise WorkflowError(
            "launch receipt configuration fingerprint does not match"
        )
    receipt_commit = _required_string(
        value,
        "source_commit",
        label="source commit",
    )
    if (
        not COMMIT_PATTERN.fullmatch(receipt_commit)
        or receipt_commit != source_commit
    ):
        raise WorkflowError("launch receipt source commit does not match HEAD")

    container = value.get("container")
    if container != contract["container"]:
        raise WorkflowError(
            "launch receipt container pins do not match the committed contract"
        )
    if value.get("bootstrap") != contract["bootstrap"]:
        raise WorkflowError(
            "launch receipt bootstrap contract does not match"
        )
    user_data_path = PROJECT_ROOT / contract["bootstrap"]["user_data_path"]
    user_data_sha256 = _required_string(
        value,
        "user_data_sha256",
        label="user-data digest",
    )
    if (
        not HASH_PATTERN.fullmatch(user_data_sha256)
        or user_data_sha256 != _sha256_file(user_data_path)
    ):
        raise WorkflowError("launch receipt user-data digest does not match")

    instance_id = _required_string(
        value,
        "instance_id",
        label="instance ID",
    )
    instance_type = _required_string(
        value,
        "instance_type",
        label="instance type",
    )
    region = _required_string(value, "region", label="region")
    subnet_id = _required_string(value, "subnet_id", label="subnet ID")
    security_group_id = _required_string(
        value,
        "security_group_id",
        label="security group ID",
    )
    if not INSTANCE_PATTERN.fullmatch(instance_id):
        raise WorkflowError("launch receipt instance ID is invalid")
    if instance_type not in contract["allowed_instance_types"]:
        raise WorkflowError("launch receipt instance type is outside scope")
    if not REGION_PATTERN.fullmatch(region):
        raise WorkflowError("launch receipt region is invalid")
    if not SUBNET_PATTERN.fullmatch(subnet_id):
        raise WorkflowError("launch receipt subnet ID is invalid")
    if not SECURITY_GROUP_PATTERN.fullmatch(security_group_id):
        raise WorkflowError("launch receipt security group ID is invalid")
    if not isinstance(value.get("public_ip_requested"), bool):
        raise WorkflowError("launch receipt public-IP choice is invalid")
    root_volume_gib = value.get("root_volume_gib")
    if (
        not isinstance(root_volume_gib, int)
        or isinstance(root_volume_gib, bool)
        or not 50 <= root_volume_gib <= 1000
    ):
        raise WorkflowError("launch receipt root volume size is invalid")
    _required_string(value, "client_token", label="client token")

    tags = value.get("tags")
    expected_tags = contract["tags"]
    if (
        not isinstance(tags, dict)
        or any(tags.get(key) != expected for key, expected in expected_tags.items())
        or not isinstance(tags.get("Name"), str)
        or not tags["Name"]
    ):
        raise WorkflowError("launch receipt managed tags do not match")

    resolved_ami = value.get("resolved_ami")
    if not isinstance(resolved_ami, dict) or set(resolved_ami) != {
        "image_id",
        "name",
        "creation_date",
        "architecture",
        "root_device_name",
        "owner_id",
    }:
        raise WorkflowError("launch receipt resolved AMI record is invalid")
    ami_id = _required_string(
        resolved_ami,
        "image_id",
        label="resolved AMI ID",
    )
    if (
        not AMI_PATTERN.fullmatch(ami_id)
        or resolved_ami.get("architecture") != contract["ami"]["architecture"]
        or resolved_ami.get("root_device_name")
        != contract["root_volume"]["device_name"]
        or not isinstance(resolved_ami.get("name"), str)
        or not resolved_ami["name"].startswith(contract["ami"]["name_prefix"])
        or not isinstance(resolved_ami.get("creation_date"), str)
        or not resolved_ami["creation_date"]
        or not isinstance(resolved_ami.get("owner_id"), str)
        or not ACCOUNT_PATTERN.fullmatch(resolved_ami["owner_id"])
    ):
        raise WorkflowError(
            "launch receipt resolved AMI does not match the approved DLAMI"
        )

    launch_response = value.get("launch_response")
    account_id = (
        _required_string(
            launch_response,
            "OwnerId",
            label="launch-response owner account ID",
        )
        if isinstance(launch_response, dict)
        else ""
    )
    instances = (
        launch_response.get("Instances")
        if isinstance(launch_response, dict)
        else None
    )
    if (
        not isinstance(instances, list)
        or len(instances) != 1
        or not isinstance(instances[0], dict)
    ):
        raise WorkflowError(
            "launch receipt response does not contain one instance"
        )
    launched = instances[0]
    placement = launched.get("Placement")
    availability_zone = (
        placement.get("AvailabilityZone")
        if isinstance(placement, dict)
        else None
    )
    security_groups = launched.get("SecurityGroups")
    launched_security_group_ids = (
        {
            item.get("GroupId")
            for item in security_groups
            if isinstance(item, dict)
        }
        if isinstance(security_groups, list)
        else set()
    )
    if (
        not ACCOUNT_PATTERN.fullmatch(account_id)
        or launched.get("InstanceId") != instance_id
        or launched.get("InstanceType") != instance_type
        or launched.get("ImageId") != ami_id
        or launched.get("SubnetId") != subnet_id
        or launched_security_group_ids != {security_group_id}
        or launched.get("Architecture") != contract["ami"]["architecture"]
        or not isinstance(availability_zone, str)
        or not availability_zone.startswith(region)
    ):
        raise WorkflowError(
            "launch receipt response is inconsistent with its launch fields"
        )

    return LaunchReceipt(
        path=resolved_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        config_fingerprint=config_fingerprint,
        instance_id=instance_id,
        instance_type=instance_type,
        region=region,
        ami_id=ami_id,
        account_id=account_id,
    )


def _run_local(
    command: Sequence[str],
    *,
    capture_output: bool = True,
    timeout_seconds: int = LOCAL_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=capture_output,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(
            f"local command {command[0]!r} exceeded its "
            f"{timeout_seconds}-second bound"
        ) from exc
    except OSError as exc:
        raise WorkflowError(
            f"cannot execute local command {command[0]!r}: {exc}"
        ) from exc


def _source_state() -> tuple[str | None, list[str]]:
    blockers: list[str] = []
    revision = _run_local(
        ("git", "rev-parse", "--verify", "HEAD")
    )
    source_commit = revision.stdout.strip()
    if revision.returncode != 0 or not COMMIT_PATTERN.fullmatch(
        source_commit
    ):
        blockers.append("cannot resolve a 40-character Git HEAD")
        source_commit = None

    status = _run_local(
        ("git", "status", "--porcelain=v1", "--untracked-files=all")
    )
    if status.returncode != 0:
        blockers.append("cannot inspect the Git worktree")
    elif status.stdout:
        blockers.append(
            "worktree is not clean; commit or remove every change first"
        )
    return source_commit, blockers


def _validate_remote_path(name: str, value: str) -> str:
    if (
        not SAFE_ABSOLUTE_REMOTE_PATH.fullmatch(value)
        or "//" in value
        or "/../" in f"{value}/"
        or value in {"/", "/home", "/home/ubuntu"}
    ):
        raise WorkflowError(
            f"{name} must be a scoped safe absolute remote path"
        )
    return value.rstrip("/")


def _spec_from_args(
    args: argparse.Namespace,
    source_commit: str,
    launch_receipt: LaunchReceipt | None,
) -> RemoteSpec:
    if not HOST_PATTERN.fullmatch(args.host):
        raise WorkflowError("--host must be a safe IPv4 address or DNS name")
    if not USER_PATTERN.fullmatch(args.user):
        raise WorkflowError("--user is not a safe Linux account name")
    if not 1 <= args.port <= 65535:
        raise WorkflowError("--port must be between 1 and 65535")
    if not args.identity_file.is_absolute():
        raise WorkflowError("--identity-file must be an absolute path")
    launch_receipt_path = args.launch_receipt.expanduser().resolve()
    if not args.local_result_base.is_absolute():
        raise WorkflowError("--local-result-base must be an absolute path")
    local_base = args.local_result_base.resolve()
    broad_local_bases = {
        Path("/"),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
    }
    if local_base in broad_local_bases:
        raise WorkflowError("--local-result-base is too broad")
    remote_source_base = _validate_remote_path(
        "--remote-source-base", args.remote_source_base
    )
    remote_result_base = _validate_remote_path(
        "--remote-result-base", args.remote_result_base
    )
    if (
        remote_source_base == remote_result_base
        or remote_source_base.startswith(f"{remote_result_base}/")
        or remote_result_base.startswith(f"{remote_source_base}/")
    ):
        raise WorkflowError(
            "remote source and result bases must be disjoint"
        )
    run_id = args.run_id or (
        f"phase11-{source_commit[:12]}-{_utc_compact()}"
    )
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise WorkflowError("--run-id must contain 1-80 safe characters")
    if run_id.endswith(RESERVED_RUN_ID_SUFFIXES):
        raise WorkflowError(
            "--run-id uses a suffix reserved for remote workflow artifacts"
        )
    image_name = args.image_name or (
        f"colpali-triton:phase11-{source_commit[:12]}"
    )
    if not IMAGE_PATTERN.fullmatch(image_name):
        raise WorkflowError("--image-name must be a safe explicit name:tag")
    return RemoteSpec(
        host=args.host,
        user=args.user,
        port=args.port,
        identity_file=args.identity_file,
        remote_source_base=remote_source_base,
        remote_result_base=remote_result_base,
        local_result_base=local_base,
        run_id=run_id,
        source_commit=source_commit,
        image_name=image_name,
        launch_receipt_path=launch_receipt_path,
        launch_receipt_sha256=(
            launch_receipt.sha256 if launch_receipt is not None else None
        ),
        launch_config_fingerprint=(
            launch_receipt.config_fingerprint
            if launch_receipt is not None
            else None
        ),
        expected_instance_id=(
            launch_receipt.instance_id if launch_receipt is not None else None
        ),
        expected_instance_type=(
            launch_receipt.instance_type if launch_receipt is not None else None
        ),
        expected_region=(
            launch_receipt.region if launch_receipt is not None else None
        ),
        expected_ami_id=(
            launch_receipt.ami_id if launch_receipt is not None else None
        ),
        expected_account_id=(
            launch_receipt.account_id if launch_receipt is not None else None
        ),
    )


def _local_preflight(spec: RemoteSpec) -> list[str]:
    blockers: list[str] = []
    if not spec.identity_file.is_file():
        blockers.append("--identity-file is not an existing regular file")
    elif not os.access(spec.identity_file, os.R_OK):
        blockers.append("--identity-file is not readable")
    elif spec.identity_file.stat().st_mode & 0o077:
        blockers.append(
            "--identity-file must not be readable or writable by group/other"
        )
    for executable in ("git", "ssh", "scp"):
        if shutil.which(executable) is None:
            blockers.append(
                f"required local executable is absent: {executable}"
            )
    if spec.local_result_directory.exists():
        try:
            _reuse_local_result(spec)
        except WorkflowError as exc:
            blockers.append(
                "local result directory exists but is not an exact verified "
                f"reusable result: {exc}"
            )
    return blockers


def build_plan(
    spec: RemoteSpec,
    *,
    blockers: Sequence[str],
) -> dict[str, Any]:
    expected_host = (
        spec.expected_host
        if spec.launch_receipt_sha256 is not None
        else None
    )
    return {
        "format": "colpali-triton-phase11-remote-plan-v1",
        "status": "ready" if not blockers else "blocked",
        "blockers": list(blockers),
        "source_commit": spec.source_commit,
        "run_id": spec.run_id,
        "host": spec.host,
        "ssh_user": spec.user,
        "ssh_port": spec.port,
        "container_image": spec.image_name,
        "launch_receipt": {
            "path": str(spec.launch_receipt_path),
            "sha256": spec.launch_receipt_sha256,
            "config_fingerprint": spec.launch_config_fingerprint,
            "valid": expected_host is not None,
        },
        "expected_host": expected_host,
        "remote_source_directory": spec.remote_source_directory,
        "remote_result_directory": spec.remote_result_directory,
        "remote_control_directory": spec.remote_control_directory,
        "remote_unit_name": spec.remote_unit_name,
        "remote_job_timeout_seconds": REMOTE_JOB_TIMEOUT_SECONDS,
        "local_result_directory": str(spec.local_result_directory),
        "steps": [
            "verify the existing host bootstrap ready marker",
            "archive the exact source commit and verify both archive and content",
            "publish or reverify a read-only commit-specific remote source",
            "start or resume a detached bounded host-side service",
            "poll through reconnectable short SSH requests",
            "build the digest-pinned linux/amd64 CUDA image",
            "require cuda:0 to match the exact NVIDIA L4 hardware gate",
            "run the explicit CUDA test suite and zero-skip parity gate",
            "run the fixed Triton tuning matrix",
            "derive the winner config and run all canonical benchmark cases",
            "profile that winner with bounded ncu then nsys fallback",
            "seal every remote result with SHA-256",
            "retrieve and verify a non-overwriting local bundle",
        ],
        "safety": {
            "plan_invokes_ssh": False,
            "plan_invokes_aws": False,
            "execution_creates_ec2_instances": False,
            "execution_terminates_or_stops_instance": False,
            "remote_results_overwritten": False,
            "local_results_overwritten": False,
            "remote_job_survives_client_disconnect": True,
            "same_run_id_is_resumable": True,
            "instance_left_running_after_completion": True,
        },
    }


def _ssh_options(spec: RemoteSpec) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-i",
        str(spec.identity_file),
        "-p",
        str(spec.port),
    ]


def _ssh_command(spec: RemoteSpec, remote_command: str) -> list[str]:
    return [
        "ssh",
        *_ssh_options(spec),
        spec.ssh_target,
        remote_command,
    ]


def _run_ssh(
    spec: RemoteSpec,
    remote_command: str,
    *,
    capture_output: bool = True,
    timeout_seconds: int = SSH_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    completed = _run_local(
        _ssh_command(spec, remote_command),
        capture_output=capture_output,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        message = (
            completed.stderr.strip()
            if capture_output
            else "remote command failed"
        )
        error_type = (
            SSHTransportError
            if completed.returncode == 255
            else WorkflowError
        )
        raise error_type(message)
    return completed


def _prepare_source_archive(spec: RemoteSpec, output: Path) -> SourceArchive:
    try:
        with output.open("xb") as handle:
            completed = subprocess.run(
                (
                    "git",
                    "archive",
                    "--format=tar",
                    spec.source_commit,
                ),
                cwd=PROJECT_ROOT,
                check=False,
                stdout=handle,
                stderr=subprocess.PIPE,
                timeout=SOURCE_ARCHIVE_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(
            "committed-source archive creation exceeded its "
            f"{SOURCE_ARCHIVE_TIMEOUT_SECONDS}-second bound"
        ) from exc
    except OSError as exc:
        raise WorkflowError(f"cannot create committed source archive: {exc}") from exc
    if completed.returncode != 0:
        raise WorkflowError(
            "git archive failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    if not output.is_file() or output.stat().st_size == 0:
        raise WorkflowError("git archive produced an empty source archive")

    digest = hashlib.sha256()
    names: set[str] = set()
    total = 0
    try:
        with tarfile.open(output, "r:") as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                canonical = path.as_posix()
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "." in path.parts
                    or not path.parts
                    or member.name.rstrip("/") != canonical
                    or canonical in names
                    or not (member.isfile() or member.isdir())
                    or canonical in SOURCE_MARKER_NAMES
                ):
                    raise WorkflowError(
                        "unsafe member in committed source archive: "
                        f"{member.name!r}"
                    )
                names.add(canonical)
                if not member.isfile():
                    continue
                total += member.size
                if total > MAX_BUNDLE_UNCOMPRESSED_BYTES:
                    raise WorkflowError(
                        "committed source archive exceeds the 10 GiB "
                        "safety limit"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise WorkflowError(
                        f"cannot read source archive member: {member.name!r}"
                    )
                with source:
                    content = source.read()
                if len(content) != member.size:
                    raise WorkflowError(
                        f"truncated source archive member: {member.name!r}"
                    )
                digest.update(canonical.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(len(content)).encode("ascii"))
                digest.update(b"\0")
                digest.update(content)
                digest.update(b"\0")
    except tarfile.TarError as exc:
        raise WorkflowError(f"cannot inspect committed source archive: {exc}") from exc
    if not names:
        raise WorkflowError("committed source archive has no members")
    return SourceArchive(
        path=output,
        archive_sha256=_sha256_file(output),
        content_sha256=digest.hexdigest(),
    )


def _source_content_check(
    directory: str,
    content_sha256: str,
) -> str:
    return " ".join(
        (
            "python3",
            "-c",
            shlex.quote(SOURCE_CONTENT_HASH_SCRIPT),
            shlex.quote(directory),
            shlex.quote(content_sha256),
        )
    )


def _remote_source_state(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> str:
    directory = shlex.quote(spec.remote_source_directory)
    commit = shlex.quote(spec.source_commit)
    archive_sha = shlex.quote(source_archive.archive_sha256)
    content_sha = shlex.quote(source_archive.content_sha256)
    content_check = _source_content_check(
        spec.remote_source_directory,
        source_archive.content_sha256,
    )
    command = (
        "set -eu; "
        f"if test ! -e {directory}; then printf missing; "
        f"elif test -d {directory} "
        f"&& test -f {directory}/.source_complete "
        f"&& test -f {directory}/.source_commit "
        f"&& test -f {directory}/.source_archive_sha256 "
        f"&& test -f {directory}/.source_content_sha256 "
        f"&& test \"$(cat {directory}/.source_commit)\" = {commit} "
        f"&& test \"$(cat {directory}/.source_archive_sha256)\" "
        f"= {archive_sha} "
        f"&& test \"$(cat {directory}/.source_content_sha256)\" "
        f"= {content_sha} "
        f"&& test -z \"$(find {directory} -perm /222 -print -quit)\" "
        f"&& {content_check} >/dev/null 2>&1; "
        "then printf ready; else printf invalid; fi"
    )
    result = _run_ssh(spec, command).stdout.strip()
    if result not in {"missing", "ready", "invalid"}:
        raise WorkflowError(f"unexpected remote source state: {result!r}")
    return result


def _upload_archive(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> None:
    directory = shlex.quote(spec.remote_source_directory)
    parent_value = str(PurePosixPath(spec.remote_source_directory).parent)
    parent = shlex.quote(parent_value)
    commit = shlex.quote(spec.source_commit)
    archive_sha = shlex.quote(source_archive.archive_sha256)
    content_sha = shlex.quote(source_archive.content_sha256)
    stage_template = shlex.quote(
        f"{parent_value}/.source-{spec.source_commit[:12]}.upload.XXXXXX"
    )
    stage_content_check = " ".join(
        (
            "python3",
            "-c",
            shlex.quote(SOURCE_CONTENT_HASH_SCRIPT),
            '"${stage}"',
            content_sha,
        )
    )
    remote_command = (
        "set -eu; umask 022; "
        f"mkdir -p {parent}; test ! -e {directory}; "
        f"stage=$(mktemp -d {stage_template}); "
        "archive_path=\"${stage}.tar\"; "
        "cleanup() { chmod -R u+w \"${stage}\" 2>/dev/null || true; "
        "rm -rf -- \"${stage}\" \"${archive_path}\"; }; "
        "trap cleanup EXIT HUP INT TERM; "
        "cat > \"${archive_path}\"; "
        f"printf '%s  %s\\n' {archive_sha} \"${{archive_path}}\" "
        "| sha256sum --check - >/dev/null; "
        "tar --no-same-owner -xf \"${archive_path}\" -C \"${stage}\"; "
        f"{stage_content_check} >/dev/null; "
        f"printf '%s\\n' {commit} > \"${{stage}}/.source_commit\"; "
        f"printf '%s\\n' {archive_sha} "
        "> \"${stage}/.source_archive_sha256\"; "
        f"printf '%s\\n' {content_sha} "
        "> \"${stage}/.source_content_sha256\"; "
        "touch \"${stage}/.source_complete\"; "
        "chmod -R a-w \"${stage}\"; chmod -R a+rX \"${stage}\"; "
        f"{stage_content_check} >/dev/null; "
        "test -z \"$(find \"${stage}\" -perm /222 -print -quit)\"; "
        f"mv -T \"${{stage}}\" {directory}; "
        "rm -f -- \"${archive_path}\"; trap - EXIT HUP INT TERM; "
        "printf ready"
    )
    try:
        with source_archive.path.open("rb") as archive_stream:
            upload = subprocess.run(
                _ssh_command(spec, remote_command),
                cwd=PROJECT_ROOT,
                check=False,
                stdin=archive_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=SOURCE_UPLOAD_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(
            "committed-source upload exceeded its "
            f"{SOURCE_UPLOAD_TIMEOUT_SECONDS}-second transfer bound; "
            "the immutable target was not accepted as published"
        ) from exc
    except OSError as exc:
        raise WorkflowError(
            f"cannot start committed-source upload: {exc}"
        ) from exc
    if upload.returncode != 0:
        raise WorkflowError(
            "remote source upload failed: "
            + upload.stderr.decode("utf-8", errors="replace").strip()
        )
    if upload.stdout.strip() != b"ready":
        raise WorkflowError("remote source upload returned unexpected output")
    if _remote_source_state(spec, source_archive) != "ready":
        raise WorkflowError("uploaded source failed immutable re-verification")


def _remote_job_command(spec: RemoteSpec) -> str:
    source = shlex.quote(spec.remote_source_directory)
    expected_host = spec.expected_host
    return " ".join(
        (
            f"{source}/infra/aws/run_phase11_job.sh",
            "--source-directory",
            source,
            "--result-directory",
            shlex.quote(spec.remote_result_directory),
            "--source-commit",
            shlex.quote(spec.source_commit),
            "--run-id",
            shlex.quote(spec.run_id),
            "--image-name",
            shlex.quote(spec.image_name),
            "--expected-instance-id",
            shlex.quote(expected_host["instance_id"]),
            "--expected-instance-type",
            shlex.quote(expected_host["instance_type"]),
            "--expected-region",
            shlex.quote(expected_host["region"]),
            "--expected-ami-id",
            shlex.quote(expected_host["ami_id"]),
            "--expected-account-id",
            shlex.quote(expected_host["account_id"]),
            "--launch-receipt-sha256",
            shlex.quote(expected_host["launch_receipt_sha256"]),
            "--launch-config-fingerprint",
            shlex.quote(expected_host["config_fingerprint"]),
        )
    )


def _remote_finalizer_command(
    spec: RemoteSpec,
    *,
    recovery_source: str,
    job_exit_code_word: str,
) -> str:
    if recovery_source not in {"detached-wrapper", "poll-fallback"}:
        raise WorkflowError("invalid remote finalizer recovery source")
    expected_host = spec.expected_host
    source = shlex.quote(spec.remote_source_directory)
    return " ".join(
        (
            "python3",
            f"{source}/infra/aws/phase11_finalize_artifacts.py",
            "--source-directory",
            source,
            "--result-directory",
            shlex.quote(spec.remote_result_directory),
            "--control-directory",
            shlex.quote(spec.remote_control_directory),
            "--source-commit",
            shlex.quote(spec.source_commit),
            "--run-id",
            shlex.quote(spec.run_id),
            "--expected-instance-id",
            shlex.quote(expected_host["instance_id"]),
            "--expected-instance-type",
            shlex.quote(expected_host["instance_type"]),
            "--expected-region",
            shlex.quote(expected_host["region"]),
            "--expected-ami-id",
            shlex.quote(expected_host["ami_id"]),
            "--expected-account-id",
            shlex.quote(expected_host["account_id"]),
            "--launch-receipt-sha256",
            shlex.quote(expected_host["launch_receipt_sha256"]),
            "--launch-config-fingerprint",
            shlex.quote(expected_host["config_fingerprint"]),
            "--job-exit-code",
            job_exit_code_word,
            "--recovery-source",
            recovery_source,
        )
    )


def _control_contract_checks(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> str:
    control = shlex.quote(spec.remote_control_directory)
    expected_host = spec.expected_host
    expected = {
        ".source_commit": spec.source_commit,
        ".source_archive_sha256": source_archive.archive_sha256,
        ".source_content_sha256": source_archive.content_sha256,
        ".run_id": spec.run_id,
        ".image_name": spec.image_name,
        ".unit_name": spec.remote_unit_name,
        ".timeout_seconds": str(REMOTE_JOB_TIMEOUT_SECONDS),
        ".expected_instance_id": expected_host["instance_id"],
        ".expected_instance_type": expected_host["instance_type"],
        ".expected_region": expected_host["region"],
        ".expected_ami_id": expected_host["ami_id"],
        ".expected_account_id": expected_host["account_id"],
        ".launch_receipt_sha256": expected_host["launch_receipt_sha256"],
        ".launch_config_fingerprint": expected_host["config_fingerprint"],
    }
    checks = [f"test -f {control}/.control_complete"]
    for name, value in expected.items():
        checks.extend(
            (
                f"test -f {control}/{name}",
                (
                    f"test \"$(cat {control}/{name})\" = "
                    f"{shlex.quote(value)}"
                ),
            )
        )
    return " && ".join(checks)


def _ensure_remote_control(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> None:
    control_value = spec.remote_control_directory
    control = shlex.quote(control_value)
    parent_value = str(PurePosixPath(control_value).parent)
    parent = shlex.quote(parent_value)
    template = shlex.quote(
        f"{parent_value}/.{spec.run_id}.control.XXXXXX"
    )
    expected_host = spec.expected_host
    values = {
        ".source_commit": spec.source_commit,
        ".source_archive_sha256": source_archive.archive_sha256,
        ".source_content_sha256": source_archive.content_sha256,
        ".run_id": spec.run_id,
        ".image_name": spec.image_name,
        ".unit_name": spec.remote_unit_name,
        ".timeout_seconds": str(REMOTE_JOB_TIMEOUT_SECONDS),
        ".expected_instance_id": expected_host["instance_id"],
        ".expected_instance_type": expected_host["instance_type"],
        ".expected_region": expected_host["region"],
        ".expected_ami_id": expected_host["ami_id"],
        ".expected_account_id": expected_host["account_id"],
        ".launch_receipt_sha256": expected_host["launch_receipt_sha256"],
        ".launch_config_fingerprint": expected_host["config_fingerprint"],
    }
    writes = " ".join(
        (
            f"printf '%s\\n' {shlex.quote(value)} "
            f"> \"${{stage}}/{name}\";"
        )
        for name, value in values.items()
    )
    marker_paths = " ".join(
        f"\"${{stage}}/{name}\""
        for name in (*values.keys(), ".control_complete")
    )
    checks = _control_contract_checks(spec, source_archive)
    command = (
        "set -eu; umask 022; "
        f"mkdir -p {parent}; "
        f"if test ! -e {control}; then "
        f"stage=$(mktemp -d {template}); "
        "cleanup() { rm -rf -- \"${stage}\"; }; "
        "trap cleanup EXIT HUP INT TERM; "
        f"{writes} "
        "touch \"${stage}/.control_complete\"; "
        f"chmod 0444 {marker_paths}; "
        f"mv -T \"${{stage}}\" {control}; "
        "trap - EXIT HUP INT TERM; "
        "fi; "
        f"if {checks}; then printf ready; else printf invalid; fi"
    )
    result = _run_ssh(spec, command).stdout.strip()
    if result != "ready":
        raise WorkflowError(
            "remote run control path exists with a different contract"
        )


def _remote_run_state(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> str:
    control = shlex.quote(spec.remote_control_directory)
    result = shlex.quote(spec.remote_result_directory)
    bundle = shlex.quote(f"{spec.remote_result_directory}.tar.gz")
    unit = shlex.quote(spec.remote_unit_name)
    checks = _control_contract_checks(spec, source_archive)
    command = (
        "set -eu; "
        f"unit_state=$(systemctl is-active {unit} 2>/dev/null || true); "
        f"if test ! -e {control}; then "
        f"if test -e {result} || test -e {bundle}; "
        "then printf invalid; else printf missing; fi; "
        f"elif ! ({checks}); then printf invalid; "
        f"elif test -f {bundle}; then printf bundled; "
        f"elif test -f {result}/SEALED.sha256 "
        f"&& (cd {result} "
        "&& sha256sum --check SEALED.sha256 >/dev/null 2>&1); "
        "then printf sealed; "
        "elif test \"${unit_state}\" = active "
        "|| test \"${unit_state}\" = activating "
        "|| test \"${unit_state}\" = reloading "
        "|| test \"${unit_state}\" = deactivating; "
        "then printf running; "
        f"elif test -f {control}/job.done "
        f"|| test -e {result} "
        f"|| systemctl is-failed --quiet {unit}; "
        "then printf needs_finalization; "
        "else printf prepared; fi"
    )
    state = _run_ssh(spec, command).stdout.strip()
    allowed = {
        "missing",
        "prepared",
        "running",
        "sealed",
        "bundled",
        "needs_finalization",
        "invalid",
    }
    if state not in allowed:
        raise WorkflowError(f"unexpected remote run state: {state!r}")
    return state


def _start_remote_job(spec: RemoteSpec) -> None:
    control = shlex.quote(spec.remote_control_directory)
    result = shlex.quote(spec.remote_result_directory)
    bundle = shlex.quote(f"{spec.remote_result_directory}.tar.gz")
    unit = shlex.quote(spec.remote_unit_name)
    log = shlex.quote(f"{spec.remote_control_directory}/job.log")
    exit_tmp = shlex.quote(
        f"{spec.remote_control_directory}/job.exit_code.tmp"
    )
    exit_path = shlex.quote(
        f"{spec.remote_control_directory}/job.exit_code"
    )
    done = shlex.quote(f"{spec.remote_control_directory}/job.done")
    job = _remote_job_command(spec)
    finalizer = _remote_finalizer_command(
        spec,
        recovery_source="detached-wrapper",
        job_exit_code_word='"${job_code}"',
    )
    finalizer_exit_tmp = shlex.quote(
        f"{spec.remote_control_directory}/finalizer.exit_code.tmp"
    )
    finalizer_exit_path = shlex.quote(
        f"{spec.remote_control_directory}/finalizer.exit_code"
    )
    wrapper = (
        "set +e; "
        f"/usr/bin/timeout --signal=TERM --kill-after=60s "
        f"{REMOTE_JOB_TIMEOUT_SECONDS}s {job} > {log} 2>&1; "
        "job_code=$?; "
        f"/usr/bin/timeout --signal=TERM --kill-after=30s "
        f"{REMOTE_FINALIZER_TIMEOUT_SECONDS}s {finalizer} >> {log} 2>&1; "
        "finalizer_code=$?; "
        f"printf '%s\\n' \"${{finalizer_code}}\" > {finalizer_exit_tmp}; "
        f"mv -T {finalizer_exit_tmp} {finalizer_exit_path}; "
        f"printf '%s\\n' \"${{job_code}}\" > {exit_tmp}; "
        f"mv -T {exit_tmp} {exit_path}; touch {done}; "
        "if test \"${finalizer_code}\" -ne 0; then "
        "exit \"${finalizer_code}\"; fi; exit \"${job_code}\""
    )
    command = (
        "set -eu; "
        f"test -d {control}; "
        f"unit_state=$(systemctl is-active {unit} 2>/dev/null || true); "
        f"if test -f {bundle} || test -f {result}/SEALED.sha256; "
        "then printf complete; "
        "elif test \"${unit_state}\" = active "
        "|| test \"${unit_state}\" = activating "
        "|| test \"${unit_state}\" = reloading "
        "|| test \"${unit_state}\" = deactivating; "
        "then printf running; "
        f"elif test -f {done} || systemctl is-failed --quiet {unit}; "
        "then printf failed; exit 4; "
        "else "
        "sudo systemd-run --quiet "
        f"--unit={unit} --uid={shlex.quote(spec.user)} "
        "--property=Type=exec "
        f"--property=RuntimeMaxSec={REMOTE_JOB_OUTER_TIMEOUT_SECONDS}s "
        "--property=TimeoutStopSec=60s --property=KillMode=mixed "
        f"/bin/bash -c {shlex.quote(wrapper)}; "
        "printf started; fi"
    )
    output = _run_ssh(spec, command).stdout.strip()
    if output not in {"complete", "running", "started"}:
        raise WorkflowError(
            f"unexpected detached-job launch response: {output!r}"
        )


def _finalize_remote_run(spec: RemoteSpec) -> None:
    control = shlex.quote(spec.remote_control_directory)
    exit_path = shlex.quote(
        f"{spec.remote_control_directory}/job.exit_code"
    )
    log = shlex.quote(f"{spec.remote_control_directory}/job.log")
    finalizer = _remote_finalizer_command(
        spec,
        recovery_source="poll-fallback",
        job_exit_code_word='"${job_code}"',
    )
    command = (
        "set -eu; "
        f"test -d {control}; "
        "job_code=125; "
        f"if test -f {exit_path}; then "
        f"candidate=$(cat {exit_path}); "
        "case \"${candidate}\" in "
        "''|*[!0-9]*) ;; "
        "*) if test \"${candidate}\" -le 255; "
        "then job_code=\"${candidate}\"; fi ;; "
        "esac; fi; "
        f"/usr/bin/timeout --signal=TERM --kill-after=30s "
        f"{REMOTE_FINALIZER_TIMEOUT_SECONDS}s {finalizer} >> {log} 2>&1"
    )
    _run_ssh(
        spec,
        command,
        timeout_seconds=REMOTE_FINALIZER_TIMEOUT_SECONDS + 60,
    )


def _wait_for_remote_run(
    spec: RemoteSpec,
    source_archive: SourceArchive,
) -> str:
    deadline = time.monotonic() + REMOTE_POLL_TIMEOUT_SECONDS
    last_connection_error: str | None = None
    connection_error_started: float | None = None
    while True:
        try:
            state = _remote_run_state(spec, source_archive)
            if state in {"sealed", "bundled"}:
                return state
            if state == "invalid":
                raise WorkflowError(
                    "detached Phase 11 job ended without a sealed result "
                    f"(state={state})"
                )
            if state == "needs_finalization":
                _finalize_remote_run(spec)
                finalized_state = _remote_run_state(spec, source_archive)
                if finalized_state not in {"sealed", "bundled"}:
                    raise WorkflowError(
                        "recovery finalizer did not publish a sealed "
                        f"result (state={finalized_state})"
                    )
                return finalized_state
            if state in {"missing", "prepared"}:
                _ensure_remote_control(spec, source_archive)
                _start_remote_job(spec)
        except SSHTransportError as exc:
            message = str(exc)
            last_connection_error = message
            now = time.monotonic()
            if connection_error_started is None:
                connection_error_started = now
            elif (
                now - connection_error_started
                >= REMOTE_RECONNECT_TIMEOUT_SECONDS
            ):
                raise WorkflowError(
                    "SSH remained unavailable for 15 minutes; the detached "
                    "job was not stopped and the same run ID can be resumed"
                ) from exc
        else:
            last_connection_error = None
            connection_error_started = None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = (
                f"; last SSH error: {last_connection_error}"
                if last_connection_error
                else ""
            )
            raise WorkflowError(
                "timed out waiting for the bounded detached Phase 11 job"
                + detail
            )
        time.sleep(min(REMOTE_POLL_INTERVAL_SECONDS, remaining))


def _seal_remote_bundle(spec: RemoteSpec) -> tuple[str, str]:
    result = PurePosixPath(spec.remote_result_directory)
    parent = shlex.quote(str(result.parent))
    run_id = shlex.quote(spec.run_id)
    bundle = f"{spec.remote_result_directory}.tar.gz"
    bundle_sha = f"{bundle}.sha256"
    bundle_quoted = shlex.quote(bundle)
    bundle_sha_quoted = shlex.quote(bundle_sha)
    command = (
        "set -eu; "
        f"if test -e {bundle_sha_quoted} && test ! -f {bundle_quoted}; "
        "then echo 'bundle checksum exists without bundle' >&2; exit 3; fi; "
        f"if test ! -f {bundle_quoted}; then "
        f"test -f {shlex.quote(str(result / 'SEALED.sha256'))}; "
        f"cd {shlex.quote(str(result))}; "
        "sha256sum --check SEALED.sha256; "
        f"bundle_tmp={bundle_quoted}.tmp.$$; "
        f"tar --sort=name --mtime='UTC 1970-01-01' "
        "--owner=0 --group=0 --numeric-owner "
        f"-czf \"${{bundle_tmp}}\" -C {parent} {run_id}; "
        f"mv -T \"${{bundle_tmp}}\" {bundle_quoted}; fi; "
        f"sha_tmp={bundle_sha_quoted}.tmp.$$; "
        f"sha256sum {bundle_quoted} > \"${{sha_tmp}}\"; "
        f"if test -f {bundle_sha_quoted}; then "
        f"cmp -s \"${{sha_tmp}}\" {bundle_sha_quoted} "
        "|| { echo 'existing bundle checksum disagrees' >&2; "
        "rm -f -- \"${sha_tmp}\"; exit 4; }; "
        "rm -f -- \"${sha_tmp}\"; "
        f"else mv -T \"${{sha_tmp}}\" {bundle_sha_quoted}; fi; "
        f"sha256sum --check {bundle_sha_quoted}"
    )
    _run_ssh(
        spec,
        command,
        timeout_seconds=REMOTE_BUNDLE_TIMEOUT_SECONDS,
    )
    return bundle, bundle_sha


def _safe_extract(bundle: Path, destination: Path, run_id: str) -> Path:
    destination.mkdir()
    total = 0
    names: set[str] = set()
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "." in path.parts
                    or not path.parts
                    or path.parts[0] != run_id
                    or member.name in names
                    or member.name.rstrip("/") != path.as_posix()
                    or not (member.isfile() or member.isdir())
                ):
                    raise WorkflowError(
                        "unsafe member in remote result bundle: "
                        f"{member.name!r}"
                    )
                names.add(member.name)
                total += member.size
                if total > MAX_BUNDLE_UNCOMPRESSED_BYTES:
                    raise WorkflowError(
                        "remote result bundle exceeds the 10 GiB safety limit"
                    )
            for member in members:
                member_path = PurePosixPath(member.name)
                output = destination.joinpath(*member_path.parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True, mode=0o755)
                    output.chmod(0o755)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(member)
                if source is None:
                    raise WorkflowError(
                        f"cannot read bundle member: {member.name!r}"
                    )
                with source, output.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
                output.chmod(0o644)
    except tarfile.TarError as exc:
        raise WorkflowError(
            f"cannot read remote result bundle: {exc}"
        ) from exc
    root = destination / run_id
    if not root.is_dir():
        raise WorkflowError("remote result bundle lacks its expected root")
    return root


def _normalized_expected_host(
    expected_host: Mapping[str, Any],
) -> dict[str, str]:
    required = {
        "instance_id",
        "instance_type",
        "region",
        "ami_id",
        "account_id",
        "launch_receipt_sha256",
        "config_fingerprint",
    }
    if set(expected_host) != required or not all(
        isinstance(expected_host.get(key), str) and expected_host[key]
        for key in required
    ):
        raise WorkflowError("expected launch host contract is invalid")
    normalized = {key: str(expected_host[key]) for key in required}
    if (
        not INSTANCE_PATTERN.fullmatch(normalized["instance_id"])
        or normalized["instance_type"] not in {"g6.xlarge", "g6.2xlarge"}
        or not REGION_PATTERN.fullmatch(normalized["region"])
        or not AMI_PATTERN.fullmatch(normalized["ami_id"])
        or not ACCOUNT_PATTERN.fullmatch(normalized["account_id"])
        or not HASH_PATTERN.fullmatch(normalized["launch_receipt_sha256"])
        or not HASH_PATTERN.fullmatch(normalized["config_fingerprint"])
    ):
        raise WorkflowError("expected launch host values are invalid")
    return normalized


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise WorkflowError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} is not a JSON object")
    return value


def _verify_artifact_manifest(
    root: Path,
    *,
    source_commit: str,
    run_id: str,
    expected_host: Mapping[str, Any],
    allow_local_retrieval: bool = False,
) -> tuple[str, list[dict[str, Any]], str, list[str]]:
    manifest_path = root / "artifact_manifest.json"
    sealed_path = root / "SEALED.sha256"
    try:
        manifest = json.loads(manifest_path.read_text())
        sealed = sealed_path.read_text().strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"cannot read remote artifact manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise WorkflowError("remote artifact manifest is not an object")
    if manifest.get("format") != (
        "colpali-triton-phase11-artifact-manifest-v2"
    ):
        raise WorkflowError("unexpected remote artifact manifest format")
    if manifest.get("source_commit") != source_commit:
        raise WorkflowError("remote artifact source commit does not match")
    if manifest.get("run_id") != run_id:
        raise WorkflowError("remote artifact run ID does not match")
    normalized_expected_host = _normalized_expected_host(expected_host)
    if manifest.get("expected_host") != normalized_expected_host:
        raise WorkflowError(
            "remote artifact expected host differs from the launch receipt"
        )
    manifest_sha = _sha256_file(manifest_path)
    if sealed != f"{manifest_sha}  artifact_manifest.json":
        raise WorkflowError("remote SEALED marker does not match manifest")
    workflow_status = manifest.get("status")
    if workflow_status not in {"complete", "incomplete"}:
        raise WorkflowError("remote workflow status is invalid")
    reasons = manifest.get("failure_reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, str) for reason in reasons
    ):
        raise WorkflowError("remote workflow failure reasons are invalid")
    profiling = manifest.get("profiling")
    if not isinstance(profiling, dict):
        raise WorkflowError("remote profiling status is invalid")
    profile_report = profiling.get("report")
    profile_metadata = profiling.get("metadata")
    winner_record_name = profiling.get("winner_record")

    def safe_relative_file(value: object) -> bool:
        return (
            isinstance(value, str)
            and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts
            and (root / value).is_file()
            and (root / value).stat().st_size > 0
        )

    profile_report_is_safe = safe_relative_file(profile_report)
    profile_metadata_is_safe = safe_relative_file(profile_metadata)
    winner_record_is_safe = safe_relative_file(winner_record_name)
    winner_matches_metadata = False
    if profile_metadata_is_safe and winner_record_is_safe:
        try:
            metadata_record = json.loads(
                (root / profile_metadata).read_text()
            )
            winner_record = json.loads(
                (root / winner_record_name).read_text()
            )
        except (OSError, json.JSONDecodeError):
            metadata_record = None
            winner_record = None
        winner_sha = _sha256_file(root / winner_record_name)
        winner_matches_metadata = (
            isinstance(metadata_record, dict)
            and isinstance(winner_record, dict)
            and metadata_record.get("format")
            == "colpali-triton-phase11-profile-probe-v2"
            and winner_record.get("format")
            == "colpali-triton-phase11-tuning-winner-v1"
            and metadata_record.get("source_commit")
            == winner_record.get("source_commit")
            and metadata_record.get("winner_candidate_id")
            == winner_record.get("candidate_id")
            and metadata_record.get("launch_configuration")
            == winner_record.get("launch_configuration")
            and metadata_record.get("winner_record_sha256")
            == winner_sha
            and profiling.get("winner_record_sha256") == winner_sha
        )
    profile_succeeded = (
        profiling.get("succeeded") is True
        and profile_report_is_safe
        and profile_metadata_is_safe
        and winner_record_is_safe
        and winner_matches_metadata
    )
    steps = manifest.get("steps")
    try:
        step_status = json.loads((root / "step_status.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"cannot read mandatory-step artifact: {exc}"
        ) from exc
    stored_steps = (
        step_status.get("steps")
        if isinstance(step_status, dict)
        and step_status.get("format")
        == "colpali-triton-phase11-step-status-v1"
        else None
    )
    if (
        not isinstance(steps, dict)
        or set(steps) != set(MANDATORY_PHASE11_STEPS)
        or stored_steps != steps
    ):
        raise WorkflowError("remote mandatory-step status is invalid")
    failed_steps = []
    for name, record in steps.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"exit_code", "succeeded"}
            or not isinstance(record.get("exit_code"), int)
            or isinstance(record.get("exit_code"), bool)
            or not isinstance(record.get("succeeded"), bool)
            or record["succeeded"] != (record["exit_code"] == 0)
        ):
            raise WorkflowError("remote mandatory-step record is invalid")
        if not record["succeeded"]:
            failed_steps.append(name)
    if workflow_status == "complete" and (
        reasons or failed_steps or not profile_succeeded
    ):
        raise WorkflowError(
            "remote manifest claims completion without a real profile"
        )
    if workflow_status == "incomplete" and not reasons:
        raise WorkflowError(
            "remote incomplete manifest lacks failure reasons"
        )
    if workflow_status == "complete":
        try:
            hardware_gate = json.loads(
                (root / "hardware_gate.json").read_text()
            )
            cuda_assertion = json.loads(
                (root / "cuda_test_assertion.json").read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(
                f"cannot read GPU gate artifacts: {exc}"
            ) from exc
        hardware_device = (
            hardware_gate.get("device")
            if isinstance(hardware_gate, dict)
            else None
        )
        if (
            not isinstance(hardware_gate, dict)
            or hardware_gate.get("status") != "passed"
            or not isinstance(hardware_device, dict)
            or hardware_device.get("name") != "NVIDIA L4"
            or hardware_device.get("compute_capability") != [8, 9]
            or not isinstance(
                hardware_device.get("total_memory_bytes"), int
            )
            or not 21 * 1024**3
            <= hardware_device["total_memory_bytes"]
            <= 25 * 1024**3
        ):
            raise WorkflowError(
                "remote manifest claims completion without the L4 gate"
            )
        observed_counts = (
            cuda_assertion.get("observed_counts")
            if isinstance(cuda_assertion, dict)
            else None
        )
        if (
            not isinstance(cuda_assertion, dict)
            or cuda_assertion.get("format")
            != "colpali-triton-phase11-cuda-test-assertion-v1"
            or cuda_assertion.get("status") != "passed"
            or cuda_assertion.get("required_counts")
            != REQUIRED_CUDA_COUNTS
            or observed_counts != REQUIRED_CUDA_COUNTS
            or cuda_assertion.get("skipped") != []
            or cuda_assertion.get("failed") != []
            or cuda_assertion.get("errors") != []
        ):
            raise WorkflowError(
                "remote manifest claims completion without five GPU cases"
            )
        host_metadata = _read_json_object(
            root / "host_metadata.json",
            label="remote host metadata",
        )
        if (
            host_metadata.get("format")
            != "colpali-triton-phase11-host-metadata-v1"
            or host_metadata.get("source_commit") != source_commit
            or host_metadata.get("run_id") != run_id
        ):
            raise WorkflowError(
                "remote host metadata does not identify this run"
            )
        contract, contract_evidence = _load_host_environment_contract()
        if contract is None or contract_evidence.get("available") is not True:
            raise WorkflowError(
                "cannot load the committed host environment contract"
            )
        recomputed_host_validation = validate_host_evidence(
            host_metadata,
            contract,
            expected_instance_id=normalized_expected_host["instance_id"],
            expected_instance_type=normalized_expected_host["instance_type"],
            expected_region=normalized_expected_host["region"],
            expected_ami_id=normalized_expected_host["ami_id"],
            expected_account_id=normalized_expected_host["account_id"],
        )
        if host_metadata.get("validation") != recomputed_host_validation:
            raise WorkflowError(
                "stored host validation differs from local recomputation"
            )
        if recomputed_host_validation.get("status") != "passed":
            raise WorkflowError(
                "remote manifest claims completion on the wrong EC2 host"
            )
    computed_result_contract = validate_result_contract(
        root,
        source_commit=source_commit,
    )
    try:
        stored_result_contract = json.loads(
            (root / "result_contract_validation.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"cannot read result cross-check artifact: {exc}"
        ) from exc
    if stored_result_contract != computed_result_contract:
        raise WorkflowError(
            "stored result cross-check differs from local recomputation"
        )
    if workflow_status == "complete" and (
        computed_result_contract.get("status") != "passed"
    ):
        raise WorkflowError(
            "remote manifest claims completion with inconsistent results"
        )
    records = manifest.get("files")
    if not isinstance(records, list):
        raise WorkflowError("remote artifact file list is invalid")

    declared: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise WorkflowError("remote artifact file record is invalid")
        relative = record["path"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in declared
            or not isinstance(digest, str)
            or not HASH_PATTERN.fullmatch(digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise WorkflowError("remote artifact file record is unsafe")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise WorkflowError(
                f"remote artifact failed verification: {relative}"
            )
        declared.add(relative)
        normalized.append(record)
    actual = {
        relative
        for path in root.rglob("*")
        if path.is_file()
        for relative in (path.relative_to(root).as_posix(),)
        if relative not in {"artifact_manifest.json", "SEALED.sha256"}
        and not (
            allow_local_retrieval and relative.startswith("_retrieval/")
        )
    }
    if actual != declared:
        raise WorkflowError(
            "remote artifact bundle has missing or undeclared files"
        )
    if profile_succeeded:
        required_profile_files = {
            profile_report,
            profile_metadata,
            winner_record_name,
        }
        if not required_profile_files.issubset(declared):
            raise WorkflowError(
                "remote profiler/winner records are not all in the file "
                "manifest"
            )
    return manifest_sha, normalized, workflow_status, reasons


def _retrieve_bundle(
    spec: RemoteSpec,
    bundle_remote: str,
    bundle_sha_remote: str,
    source_archive: SourceArchive,
) -> dict[str, Any]:
    target = spec.local_result_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise WorkflowError(f"refusing to overwrite local results: {target}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.run_id}.",
            dir=target.parent,
        )
    )
    try:
        bundle = staging / f"{spec.run_id}.tar.gz"
        bundle_sha_file = staging / f"{spec.run_id}.tar.gz.sha256"
        scp_options = [
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ConnectionAttempts=3",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-i",
            str(spec.identity_file),
            "-P",
            str(spec.port),
        ]
        for remote, local, timeout_seconds in (
            (
                bundle_remote,
                bundle,
                ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS,
            ),
            (
                bundle_sha_remote,
                bundle_sha_file,
                LOCAL_COMMAND_TIMEOUT_SECONDS,
            ),
        ):
            completed = _run_local(
                (
                    "scp",
                    *scp_options,
                    f"{spec.ssh_target}:{remote}",
                    str(local),
                ),
                timeout_seconds=timeout_seconds,
            )
            if completed.returncode != 0:
                raise WorkflowError(
                    "artifact retrieval failed: "
                    + completed.stderr.strip()
                )
        expected_parts = bundle_sha_file.read_text().strip().split()
        bundle_sha = _sha256_file(bundle)
        if (
            len(expected_parts) != 2
            or expected_parts[0] != bundle_sha
            or PurePosixPath(expected_parts[1]).name
            != PurePosixPath(bundle_remote).name
        ):
            raise WorkflowError("retrieved bundle SHA-256 does not match")
        extracted = staging / "extracted"
        root = _safe_extract(bundle, extracted, spec.run_id)
        (
            manifest_sha,
            files,
            workflow_status,
            failure_reasons,
        ) = _verify_artifact_manifest(
            root,
            source_commit=spec.source_commit,
            run_id=spec.run_id,
            expected_host=spec.expected_host,
        )
        retrieval = root / "_retrieval"
        retrieval.mkdir()
        shutil.copy2(bundle, retrieval / "remote_bundle.tar.gz")
        shutil.copy2(
            bundle_sha_file,
            retrieval / "remote_bundle.tar.gz.sha256",
        )
        receipt = {
            "format": "colpali-triton-phase11-retrieval-receipt-v1",
            "retrieved_at": _utc_now(),
            "source_commit": spec.source_commit,
            "source_archive_sha256": source_archive.archive_sha256,
            "source_content_sha256": source_archive.content_sha256,
            "run_id": spec.run_id,
            "remote_host": spec.host,
            "remote_result_directory": spec.remote_result_directory,
            "remote_bundle_sha256": bundle_sha,
            "remote_artifact_manifest_sha256": manifest_sha,
            "remote_artifact_file_count": len(files),
            "remote_workflow_status": workflow_status,
            "remote_failure_reasons": failure_reasons,
            "remote_unit_name": spec.remote_unit_name,
            "remote_job_timeout_seconds": REMOTE_JOB_TIMEOUT_SECONDS,
            "launch_receipt_sha256": spec.launch_receipt_sha256,
            "expected_host": spec.expected_host,
            "instance_was_stopped_or_terminated": False,
        }
        with (retrieval / "receipt.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            root.rename(target)
        except FileExistsError as exc:
            raise WorkflowError(
                f"local result directory appeared concurrently: {target}"
            ) from exc
        return receipt
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _reuse_local_result(
    spec: RemoteSpec,
    source_archive: SourceArchive | None = None,
) -> dict[str, Any]:
    root = spec.local_result_directory
    retrieval = root / "_retrieval"
    receipt_path = retrieval / "receipt.json"
    bundle = retrieval / "remote_bundle.tar.gz"
    bundle_sha_path = retrieval / "remote_bundle.tar.gz.sha256"
    try:
        receipt = json.loads(receipt_path.read_text())
        expected_parts = bundle_sha_path.read_text().strip().split()
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"cannot read reusable local result receipt: {exc}"
        ) from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("format")
        != "colpali-triton-phase11-retrieval-receipt-v1"
        or receipt.get("source_commit") != spec.source_commit
        or not isinstance(receipt.get("source_archive_sha256"), str)
        or not HASH_PATTERN.fullmatch(receipt["source_archive_sha256"])
        or not isinstance(receipt.get("source_content_sha256"), str)
        or not HASH_PATTERN.fullmatch(receipt["source_content_sha256"])
        or receipt.get("run_id") != spec.run_id
        or receipt.get("remote_host") != spec.host
        or receipt.get("remote_result_directory")
        != spec.remote_result_directory
        or receipt.get("remote_unit_name") != spec.remote_unit_name
        or receipt.get("remote_job_timeout_seconds")
        != REMOTE_JOB_TIMEOUT_SECONDS
        or receipt.get("launch_receipt_sha256")
        != spec.launch_receipt_sha256
        or receipt.get("expected_host") != spec.expected_host
    ):
        raise WorkflowError("local result receipt does not match this run")
    if source_archive is not None and (
        receipt["source_archive_sha256"] != source_archive.archive_sha256
        or receipt["source_content_sha256"] != source_archive.content_sha256
    ):
        raise WorkflowError(
            "local result source archive/content hashes do not match"
        )
    if not bundle.is_file() or len(expected_parts) != 2:
        raise WorkflowError("local result lacks its verified remote bundle")
    bundle_sha = _sha256_file(bundle)
    if (
        expected_parts[0] != bundle_sha
        or PurePosixPath(expected_parts[1]).name
        != f"{spec.run_id}.tar.gz"
        or receipt.get("remote_bundle_sha256") != bundle_sha
    ):
        raise WorkflowError("local reusable bundle SHA-256 does not match")
    manifest_sha, files, status, reasons = _verify_artifact_manifest(
        root,
        source_commit=spec.source_commit,
        run_id=spec.run_id,
        expected_host=spec.expected_host,
        allow_local_retrieval=True,
    )
    if (
        receipt.get("remote_artifact_manifest_sha256") != manifest_sha
        or receipt.get("remote_artifact_file_count") != len(files)
        or receipt.get("remote_workflow_status") != status
        or receipt.get("remote_failure_reasons") != reasons
    ):
        raise WorkflowError("local result receipt differs from its artifacts")
    return receipt


def execute(spec: RemoteSpec) -> dict[str, Any]:
    if spec.local_result_directory.exists():
        with tempfile.TemporaryDirectory(
            prefix=".phase11-source-archive-"
        ) as archive_directory:
            source_archive = _prepare_source_archive(
                spec,
                Path(archive_directory) / f"{spec.source_commit}.tar",
            )
            receipt = _reuse_local_result(spec, source_archive)
        return {
            "status": receipt["remote_workflow_status"],
            "failure_reasons": receipt["remote_failure_reasons"],
            "source_commit": spec.source_commit,
            "run_id": spec.run_id,
            "local_result_directory": str(spec.local_result_directory),
            "receipt": receipt,
            "reused_verified_local_result": True,
            "instance_left_running": True,
            "termination_was_not_invoked": True,
        }
    _run_ssh(
        spec,
        "sudo test -f /var/lib/colpali-triton/bootstrap.ready",
    )
    with tempfile.TemporaryDirectory(
        prefix=".phase11-source-archive-"
    ) as archive_directory:
        source_archive = _prepare_source_archive(
            spec,
            Path(archive_directory) / f"{spec.source_commit}.tar",
        )
        remote_source_state = _remote_source_state(spec, source_archive)
        if remote_source_state == "invalid":
            raise WorkflowError(
                "remote source path failed exact archive/content "
                "re-verification; no files were replaced"
            )
        if remote_source_state == "missing":
            _upload_archive(spec, source_archive)
        _ensure_remote_control(spec, source_archive)
        run_state = _remote_run_state(spec, source_archive)
        if run_state == "invalid":
            raise WorkflowError(
                "existing Phase 11 run cannot be resumed safely "
                f"(state={run_state})"
            )
        if run_state == "needs_finalization":
            _finalize_remote_run(spec)
            run_state = _remote_run_state(spec, source_archive)
            if run_state not in {"sealed", "bundled"}:
                raise WorkflowError(
                    "recovery finalizer did not publish a sealed "
                    f"result (state={run_state})"
                )
        if run_state in {"missing", "prepared"}:
            try:
                _start_remote_job(spec)
            except SSHTransportError:
                # The SSH response can be lost after systemd accepted the
                # unit. Short reconnecting polls determine authoritative state.
                pass
        if run_state not in {"sealed", "bundled"}:
            run_state = _wait_for_remote_run(spec, source_archive)
        bundle, bundle_sha = _seal_remote_bundle(spec)
    receipt = _retrieve_bundle(
        spec,
        bundle,
        bundle_sha,
        source_archive,
    )
    return {
        "status": receipt["remote_workflow_status"],
        "failure_reasons": receipt["remote_failure_reasons"],
        "source_commit": spec.source_commit,
        "run_id": spec.run_id,
        "local_result_directory": str(spec.local_result_directory),
        "receipt": receipt,
        "remote_resume_state": run_state,
        "reused_verified_local_result": False,
        "instance_left_running": True,
        "termination_was_not_invoked": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument(
        "--launch-receipt",
        type=Path,
        default=DEFAULT_LAUNCH_RECEIPT,
        help=(
            "Phase 10 lifecycle launch receipt; execution requires an exact "
            "schema/source/environment match"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--remote-source-base",
        default="/home/ubuntu/colpali-triton",
    )
    parser.add_argument(
        "--remote-result-base",
        default="/home/ubuntu/colpali-phase11-results",
    )
    parser.add_argument(
        "--local-result-base",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase11" / "remote",
    )
    parser.add_argument("--image-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the printed plan against the existing remote host.",
    )
    parser.add_argument(
        "--acknowledge-run",
        help=(
            "Execution safety phrase; ignored for an offline plan."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        source_commit, blockers = _source_state()
        if source_commit is None:
            source_commit = "0" * 40
        launch_receipt = None
        try:
            launch_receipt = _load_launch_receipt(
                args.launch_receipt,
                source_commit=source_commit,
            )
        except WorkflowError as exc:
            blockers.append(f"launch receipt: {exc}")
        spec = _spec_from_args(args, source_commit, launch_receipt)
        blockers.extend(_local_preflight(spec))
        plan = build_plan(spec, blockers=blockers)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.acknowledge_run != RUN_ACKNOWLEDGEMENT:
            raise WorkflowError(
                "--acknowledge-run must exactly equal "
                f"{RUN_ACKNOWLEDGEMENT!r}"
            )
        if blockers:
            raise WorkflowError(
                "local preflight is blocked: " + "; ".join(blockers)
            )
        result = execute(spec)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "complete" else 3
    except (OSError, WorkflowError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
