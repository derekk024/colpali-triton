#!/usr/bin/env python3
"""Fail-closed recovery finalizer for an unsealed Phase 11 host run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

AWS_INFRA_DIRECTORY = Path(__file__).resolve().parent
if str(AWS_INFRA_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(AWS_INFRA_DIRECTORY))
from phase11_result_contract import validate_result_contract


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INSTANCE_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")
AMI_PATTERN = re.compile(r"^ami-[0-9a-f]{8,17}$")
REGION_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+-\d+$")
ACCOUNT_PATTERN = re.compile(r"^\d{12}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SAFE_ABSOLUTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{12,64}$")
MANDATORY_STEPS = (
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
GENERATED_CONTRACT_FILES = (
    "artifact_manifest.json",
    "SEALED.sha256",
    "step_status.json",
    "profiling_status.json",
    "result_contract_validation.json",
    "emergency_container_cleanup.json",
)
DOCKER_COMMAND_TIMEOUT_SECONDS = 30


class FinalizationError(RuntimeError):
    """An unrecoverable recovery-finalization error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".phase11-finalize-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _existing_seal_is_valid(
    root: Path,
    *,
    source_commit: str,
    run_id: str,
    expected_host: dict[str, str],
) -> bool:
    manifest = root / "artifact_manifest.json"
    sealed = root / "SEALED.sha256"
    if not manifest.is_file() or not sealed.is_file():
        return False
    try:
        marker = sealed.read_text(encoding="ascii").strip()
        record = json.loads(
            manifest.read_bytes(),
            object_pairs_hook=_reject_duplicate_keys,
        )
        manifest_sha256 = _sha256(manifest)
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        marker == f"{manifest_sha256}  artifact_manifest.json"
        and isinstance(record, dict)
        and record.get("format")
        == "colpali-triton-phase11-artifact-manifest-v2"
        and record.get("status") in {"complete", "incomplete"}
        and record.get("source_commit") == source_commit
        and record.get("run_id") == run_id
        and record.get("expected_host") == expected_host
    )


def _quarantine_partial_contracts(root: Path) -> list[str]:
    candidates = [
        root / name
        for name in GENERATED_CONTRACT_FILES
        if (root / name).exists()
    ]
    candidates.extend(
        path
        for path in root.glob(".phase11-finalize-*.tmp")
        if path.is_file()
    )
    if not candidates:
        return []
    quarantine_root = root / ".finalizer-quarantine"
    quarantine_root.mkdir(exist_ok=True)
    attempt = quarantine_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{os.getpid()}"
    )
    attempt.mkdir()
    moved = []
    for path in candidates:
        destination = attempt / path.name
        os.replace(path, destination)
        moved.append(destination.relative_to(root).as_posix())
    return moved


def _docker(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["docker", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=DOCKER_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _container_labels(container_id: str) -> tuple[str | None, str | None]:
    completed = _docker(
        (
            "inspect",
            "--format",
            (
                "{{ index .Config.Labels "
                '"colpali.phase11.run_id" }}\t'
                "{{ index .Config.Labels "
                '"colpali.phase11.source_commit" }}'
            ),
            container_id,
        )
    )
    if completed is None or completed.returncode != 0:
        return None, None
    parts = completed.stdout.strip().split("\t")
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _cleanup_containers(
    root: Path,
    *,
    run_id: str,
    source_commit: str,
) -> dict[str, Any]:
    label_filters = (
        "ps",
        "-aq",
        "--filter",
        f"label=colpali.phase11.run_id={run_id}",
        "--filter",
        f"label=colpali.phase11.source_commit={source_commit}",
    )
    candidates: set[str] = set()
    errors: list[str] = []
    listed = _docker(label_filters)
    if listed is None:
        errors.append("bounded docker label query failed")
    elif listed.returncode != 0:
        errors.append("docker label query returned a nonzero status")
    else:
        candidates.update(
            item
            for item in listed.stdout.split()
            if CONTAINER_ID_PATTERN.fullmatch(item)
        )

    cid_directory = root / ".docker-cids"
    if cid_directory.is_dir():
        for cid_file in sorted(cid_directory.glob("*.cid")):
            try:
                container_id = cid_file.read_text().strip()
            except OSError:
                errors.append(f"cannot read cidfile: {cid_file.name}")
                continue
            if CONTAINER_ID_PATTERN.fullmatch(container_id):
                candidates.add(container_id)
            else:
                errors.append(f"invalid cidfile: {cid_file.name}")

    removed: list[str] = []
    skipped: list[str] = []
    for container_id in sorted(candidates):
        labels = _container_labels(container_id)
        if labels != (run_id, source_commit):
            skipped.append(container_id)
            continue
        removal = _docker(("rm", "--force", container_id))
        if removal is not None and removal.returncode == 0:
            removed.append(container_id)
        else:
            errors.append(f"failed to remove labeled container: {container_id}")

    remaining: list[str] = []
    verified = _docker(label_filters)
    if verified is None:
        errors.append("bounded final docker label query failed")
    elif verified.returncode != 0:
        errors.append("final docker label query returned a nonzero status")
    else:
        verified_tokens = verified.stdout.split()
        remaining = sorted(
            item
            for item in verified_tokens
            if CONTAINER_ID_PATTERN.fullmatch(item)
        )
        if len(remaining) != len(verified_tokens):
            errors.append("final docker label query returned an invalid ID")
        if remaining:
            errors.append("labeled containers remain after cleanup")
    safe_to_seal = (
        verified is not None
        and verified.returncode == 0
        and len(remaining) == len(verified.stdout.split())
        and not remaining
    )

    if safe_to_seal and cid_directory.is_dir():
        for cid_file in cid_directory.glob("*.cid"):
            cid_file.unlink(missing_ok=True)
        try:
            cid_directory.rmdir()
        except OSError:
            pass
    return {
        "format": "colpali-triton-phase11-emergency-cleanup-v1",
        "recorded_at": _utc_now(),
        "scope": {
            "run_id": run_id,
            "source_commit": source_commit,
            "required_labels": {
                "colpali.phase11.run_id": run_id,
                "colpali.phase11.source_commit": source_commit,
            },
        },
        "removed_container_ids": removed,
        "skipped_unmatched_container_ids": skipped,
        "remaining_labeled_container_ids": remaining,
        "safe_to_seal": safe_to_seal,
        "errors": errors,
    }


def _profiling_status() -> dict[str, Any]:
    return {
        "format": "colpali-triton-phase11-profiling-status-v1",
        "succeeded": False,
        "profiler": None,
        "report": None,
        "metadata": None,
        "winner_record": None,
        "winner_record_sha256": None,
        "winner_record_error": "job exited before normal artifact sealing",
        "timeout_seconds_per_attempt": 300,
        "container_capabilities": ["SYS_ADMIN"],
        "privileged_container": False,
        "attempts": [
            {
                "profiler": profiler,
                "attempted": False,
                "exit_code": None,
                "report": None,
                "report_valid": False,
                "metadata": None,
                "metadata_matches_winner": False,
                "succeeded": False,
            }
            for profiler in ("ncu", "nsys")
        ],
    }


def _safe_artifact_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        pure = PurePosixPath(relative)
        if (
            relative in {"artifact_manifest.json", "SEALED.sha256"}
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
        ):
            continue
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def _validate_args(args: argparse.Namespace) -> None:
    paths = {
        "--source-directory": args.source_directory,
        "--result-directory": args.result_directory,
        "--control-directory": args.control_directory,
    }
    for name, path in paths.items():
        text = str(path)
        if (
            not path.is_absolute()
            or not SAFE_ABSOLUTE_PATH.fullmatch(text)
            or "//" in text
            or "/../" in f"{text}/"
        ):
            raise FinalizationError(f"{name} must be a safe absolute path")
    if not args.source_directory.is_dir():
        raise FinalizationError("--source-directory does not exist")
    if not args.result_directory.is_dir():
        raise FinalizationError("--result-directory does not exist")
    if not args.control_directory.is_dir():
        raise FinalizationError("--control-directory does not exist")
    marker = args.source_directory / ".source_commit"
    try:
        marked_commit = marker.read_text().strip()
    except OSError as exc:
        raise FinalizationError(f"cannot read source marker: {exc}") from exc
    if (
        not COMMIT_PATTERN.fullmatch(args.source_commit)
        or marked_commit != args.source_commit
    ):
        raise FinalizationError("source commit does not match its marker")
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise FinalizationError("run ID is invalid")
    if not INSTANCE_PATTERN.fullmatch(args.expected_instance_id):
        raise FinalizationError("expected instance ID is invalid")
    if args.expected_instance_type not in {"g6.xlarge", "g6.2xlarge"}:
        raise FinalizationError("expected instance type is invalid")
    if not REGION_PATTERN.fullmatch(args.expected_region):
        raise FinalizationError("expected region is invalid")
    if not AMI_PATTERN.fullmatch(args.expected_ami_id):
        raise FinalizationError("expected AMI ID is invalid")
    if not ACCOUNT_PATTERN.fullmatch(args.expected_account_id):
        raise FinalizationError("expected account ID is invalid")
    if not HASH_PATTERN.fullmatch(args.launch_receipt_sha256):
        raise FinalizationError("launch receipt SHA-256 is invalid")
    if not HASH_PATTERN.fullmatch(args.launch_config_fingerprint):
        raise FinalizationError("launch config fingerprint is invalid")
    if not 0 <= args.job_exit_code <= 255:
        raise FinalizationError("job exit code is invalid")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    expected_host = {
        "instance_id": args.expected_instance_id,
        "instance_type": args.expected_instance_type,
        "region": args.expected_region,
        "ami_id": args.expected_ami_id,
        "account_id": args.expected_account_id,
        "launch_receipt_sha256": args.launch_receipt_sha256,
        "config_fingerprint": args.launch_config_fingerprint,
    }
    lock_path = args.control_directory / "artifact-finalizer.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _existing_seal_is_valid(
            args.result_directory,
            source_commit=args.source_commit,
            run_id=args.run_id,
            expected_host=expected_host,
        ):
            return {"status": "already_sealed"}

        cleanup = _cleanup_containers(
            args.result_directory,
            run_id=args.run_id,
            source_commit=args.source_commit,
        )
        cleanup["recovery_source"] = args.recovery_source
        cleanup["job_exit_code"] = args.job_exit_code
        _atomic_json(
            args.result_directory / "emergency_container_cleanup.json",
            cleanup,
        )
        if cleanup.get("safe_to_seal") is not True:
            raise FinalizationError(
                "cannot seal while exact-run container cleanup is unverified"
            )

        quarantined = _quarantine_partial_contracts(args.result_directory)
        cleanup["quarantined_contract_files"] = quarantined
        _atomic_json(
            args.result_directory / "emergency_container_cleanup.json",
            cleanup,
        )

        steps = {
            name: {"exit_code": 125, "succeeded": False}
            for name in MANDATORY_STEPS
        }
        _atomic_json(
            args.result_directory / "step_status.json",
            {
                "format": "colpali-triton-phase11-step-status-v1",
                "steps": steps,
                "recovery_finalized": True,
            },
        )
        profiling = _profiling_status()
        _atomic_json(
            args.result_directory / "profiling_status.json",
            profiling,
        )
        result_contract = validate_result_contract(
            args.result_directory,
            source_commit=args.source_commit,
        )
        _atomic_json(
            args.result_directory / "result_contract_validation.json",
            result_contract,
        )

        failure_reasons = [
            (
                "recovery finalizer sealed an unsealed host job "
                f"(source={args.recovery_source}, "
                f"exit_code={args.job_exit_code})"
            ),
            *(
                f"mandatory step failed: {name}"
                for name in MANDATORY_STEPS
            ),
            "no profiler completed with exit code 0 and a nonempty report",
        ]
        manifest = {
            "format": "colpali-triton-phase11-artifact-manifest-v2",
            "sealed_at": _utc_now(),
            "status": "incomplete",
            "failure_reasons": failure_reasons,
            "source_commit": args.source_commit,
            "run_id": args.run_id,
            "expected_host": expected_host,
            "steps": steps,
            "profiling": profiling,
            "files": _safe_artifact_files(args.result_directory),
        }
        manifest_path = args.result_directory / "artifact_manifest.json"
        _atomic_json(manifest_path, manifest)
        marker = f"{_sha256(manifest_path)}  artifact_manifest.json\n"
        _atomic_write(
            args.result_directory / "SEALED.sha256",
            marker.encode("ascii"),
        )
        return {
            "status": "sealed_incomplete",
            "failure_reasons": failure_reasons,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-instance-id", required=True)
    parser.add_argument("--expected-instance-type", required=True)
    parser.add_argument("--expected-region", required=True)
    parser.add_argument("--expected-ami-id", required=True)
    parser.add_argument("--expected-account-id", required=True)
    parser.add_argument("--launch-receipt-sha256", required=True)
    parser.add_argument("--launch-config-fingerprint", required=True)
    parser.add_argument("--job-exit-code", required=True, type=int)
    parser.add_argument(
        "--recovery-source",
        choices=("detached-wrapper", "poll-fallback"),
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = finalize(_parse_args(argv))
    except FinalizationError as exc:
        print(f"Phase 11 recovery finalization failed: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Phase 11 recovery finalization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
