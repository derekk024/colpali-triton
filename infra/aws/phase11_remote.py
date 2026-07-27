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
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ACKNOWLEDGEMENT = "RUN PHASE 11 ON THIS EXISTING INSTANCE"
HOST_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)$"
)
USER_PATTERN = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
IMAGE_PATTERN = re.compile(
    r"^[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+$"
)
SAFE_ABSOLUTE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_BUNDLE_UNCOMPRESSED_BYTES = 10 * 1024**3


class WorkflowError(RuntimeError):
    """A safe, user-facing Phase 11 workflow failure."""


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


def _run_local(
    command: Sequence[str],
    *,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=capture_output,
            text=True,
        )
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
) -> RemoteSpec:
    if not HOST_PATTERN.fullmatch(args.host):
        raise WorkflowError("--host must be a safe IPv4 address or DNS name")
    if not USER_PATTERN.fullmatch(args.user):
        raise WorkflowError("--user is not a safe Linux account name")
    if not 1 <= args.port <= 65535:
        raise WorkflowError("--port must be between 1 and 65535")
    if not args.identity_file.is_absolute():
        raise WorkflowError("--identity-file must be an absolute path")
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
        blockers.append(
            "local result directory already exists and will not be overwritten"
        )
    return blockers


def build_plan(
    spec: RemoteSpec,
    *,
    blockers: Sequence[str],
) -> dict[str, Any]:
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
        "remote_source_directory": spec.remote_source_directory,
        "remote_result_directory": spec.remote_result_directory,
        "local_result_directory": str(spec.local_result_directory),
        "steps": [
            "verify the existing host bootstrap ready marker",
            "deploy the clean committed HEAD without replacing prior source",
            "build the digest-pinned linux/amd64 CUDA image",
            "run the complete test suite on CUDA",
            "run the fixed Triton tuning matrix",
            "run the complete canonical PyTorch-versus-Triton benchmark",
            "attempt one profiler launch with a 300-second hard timeout",
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
) -> subprocess.CompletedProcess[str]:
    completed = _run_local(
        _ssh_command(spec, remote_command),
        capture_output=capture_output,
    )
    if completed.returncode != 0:
        message = (
            completed.stderr.strip()
            if capture_output
            else "remote command failed"
        )
        raise WorkflowError(message)
    return completed


def _remote_source_state(spec: RemoteSpec) -> str:
    directory = shlex.quote(spec.remote_source_directory)
    commit = shlex.quote(spec.source_commit)
    command = (
        "set -eu; "
        f"if test ! -e {directory}; then printf missing; "
        f"elif test -f {directory}/.source_complete "
        f"&& test -f {directory}/.source_commit "
        f"&& test \"$(cat {directory}/.source_commit)\" = {commit}; "
        "then printf ready; else printf invalid; fi"
    )
    result = _run_ssh(spec, command).stdout.strip()
    if result not in {"missing", "ready", "invalid"}:
        raise WorkflowError(f"unexpected remote source state: {result!r}")
    return result


def _upload_archive(spec: RemoteSpec) -> None:
    directory = shlex.quote(spec.remote_source_directory)
    parent = shlex.quote(str(PurePosixPath(
        spec.remote_source_directory
    ).parent))
    commit = shlex.quote(spec.source_commit)
    remote_command = (
        "set -eu; umask 022; "
        f"mkdir -p {parent}; test ! -e {directory}; mkdir {directory}; "
        f"tar -xf - -C {directory}; "
        f"printf '%s\\n' {commit} > {directory}/.source_commit; "
        f"touch {directory}/.source_complete"
    )
    try:
        archive = subprocess.Popen(
            ("git", "archive", "--format=tar", "HEAD"),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert archive.stdout is not None
        upload = subprocess.Popen(
            _ssh_command(spec, remote_command),
            cwd=PROJECT_ROOT,
            stdin=archive.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise WorkflowError(
            f"cannot start committed-source upload: {exc}"
        ) from exc
    archive.stdout.close()
    upload_stdout, upload_stderr = upload.communicate()
    assert archive.stderr is not None
    archive_stderr = archive.stderr.read()
    archive_returncode = archive.wait()
    if archive_returncode != 0:
        raise WorkflowError(
            "git archive failed: "
            + archive_stderr.decode("utf-8", errors="replace").strip()
        )
    if upload.returncode != 0:
        raise WorkflowError(
            "remote source upload failed: "
            + upload_stderr.decode("utf-8", errors="replace").strip()
        )
    if upload_stdout.strip():
        raise WorkflowError("remote source upload returned unexpected output")


def _build_image(spec: RemoteSpec) -> None:
    source = shlex.quote(spec.remote_source_directory)
    commit = shlex.quote(spec.source_commit)
    image = shlex.quote(spec.image_name)
    _run_ssh(
        spec,
        (
            f"set -eu; cd {source}; "
            f"COLPALI_SOURCE_COMMIT={commit} "
            f"COLPALI_CUDA_IMAGE={image} ./docker/build_cuda.sh"
        ),
        capture_output=False,
    )


def _run_remote_job(spec: RemoteSpec) -> None:
    source = shlex.quote(spec.remote_source_directory)
    command = " ".join(
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
        )
    )
    _run_ssh(spec, f"set -eu; {command}", capture_output=False)


def _seal_remote_bundle(spec: RemoteSpec) -> tuple[str, str]:
    result = PurePosixPath(spec.remote_result_directory)
    parent = shlex.quote(str(result.parent))
    run_id = shlex.quote(spec.run_id)
    bundle = f"{spec.remote_result_directory}.tar.gz"
    bundle_sha = f"{bundle}.sha256"
    command = (
        "set -eu; "
        f"test -f {shlex.quote(str(result / 'COMPLETE.sha256'))}; "
        f"cd {shlex.quote(str(result))}; "
        "sha256sum --check COMPLETE.sha256; "
        f"test ! -e {shlex.quote(bundle)}; "
        f"test ! -e {shlex.quote(bundle_sha)}; "
        f"tar --sort=name --mtime='UTC 1970-01-01' "
        "--owner=0 --group=0 --numeric-owner "
        f"-czf {shlex.quote(bundle)} -C {parent} {run_id}; "
        f"sha256sum {shlex.quote(bundle)} > {shlex.quote(bundle_sha)}"
    )
    _run_ssh(spec, command)
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


def _verify_artifact_manifest(
    root: Path,
    *,
    source_commit: str,
    run_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.sha256"
    try:
        manifest = json.loads(manifest_path.read_text())
        complete = complete_path.read_text().strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"cannot read remote artifact manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise WorkflowError("remote artifact manifest is not an object")
    if manifest.get("format") != (
        "colpali-triton-phase11-artifact-manifest-v1"
    ):
        raise WorkflowError("unexpected remote artifact manifest format")
    if manifest.get("source_commit") != source_commit:
        raise WorkflowError("remote artifact source commit does not match")
    if manifest.get("run_id") != run_id:
        raise WorkflowError("remote artifact run ID does not match")
    manifest_sha = _sha256_file(manifest_path)
    if complete != f"{manifest_sha}  artifact_manifest.json":
        raise WorkflowError("remote COMPLETE marker does not match manifest")
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
        if relative not in {"artifact_manifest.json", "COMPLETE.sha256"}
    }
    if actual != declared:
        raise WorkflowError(
            "remote artifact bundle has missing or undeclared files"
        )
    return manifest_sha, normalized


def _retrieve_bundle(
    spec: RemoteSpec,
    bundle_remote: str,
    bundle_sha_remote: str,
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
            "-i",
            str(spec.identity_file),
            "-P",
            str(spec.port),
        ]
        for remote, local in (
            (bundle_remote, bundle),
            (bundle_sha_remote, bundle_sha_file),
        ):
            completed = _run_local(
                (
                    "scp",
                    *scp_options,
                    f"{spec.ssh_target}:{remote}",
                    str(local),
                )
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
        manifest_sha, files = _verify_artifact_manifest(
            root,
            source_commit=spec.source_commit,
            run_id=spec.run_id,
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
            "run_id": spec.run_id,
            "remote_host": spec.host,
            "remote_result_directory": spec.remote_result_directory,
            "remote_bundle_sha256": bundle_sha,
            "remote_artifact_manifest_sha256": manifest_sha,
            "remote_artifact_file_count": len(files),
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


def execute(spec: RemoteSpec) -> dict[str, Any]:
    _run_ssh(
        spec,
        "sudo test -f /var/lib/colpali-triton/bootstrap.ready",
    )
    remote_state = _remote_source_state(spec)
    if remote_state == "invalid":
        raise WorkflowError(
            "remote source path exists but lacks the exact complete commit "
            "marker; no files were replaced"
        )
    if remote_state == "missing":
        _upload_archive(spec)
    _build_image(spec)
    _run_remote_job(spec)
    bundle, bundle_sha = _seal_remote_bundle(spec)
    receipt = _retrieve_bundle(spec, bundle, bundle_sha)
    return {
        "status": "completed",
        "source_commit": spec.source_commit,
        "run_id": spec.run_id,
        "local_result_directory": str(spec.local_result_directory),
        "receipt": receipt,
        "instance_left_running": True,
        "termination_was_not_invoked": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", type=Path, required=True)
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
        spec = _spec_from_args(args, source_commit)
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
        return 0
    except (OSError, WorkflowError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
