from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / "infra" / "aws" / "phase11_remote.py"
JOB_PATH = PROJECT_ROOT / "infra" / "aws" / "run_phase11_job.sh"
PROFILE_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "profile_phase11_maxsim.py"
)
SOURCE_COMMIT = "a" * 40


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module("phase11_remote_workflow", WORKFLOW_PATH)


def _spec(tmp_path: Path, **overrides):
    values = {
        "host": "203.0.113.10",
        "user": "ubuntu",
        "port": 22,
        "identity_file": tmp_path / "benchmark.pem",
        "remote_source_base": "/home/ubuntu/colpali-triton",
        "remote_result_base": "/home/ubuntu/colpali-phase11-results",
        "local_result_base": tmp_path / "results",
        "run_id": "phase11-test",
        "source_commit": SOURCE_COMMIT,
        "image_name": "colpali-triton:phase11-test",
    }
    values.update(overrides)
    return workflow.RemoteSpec(**values)


def _write_manifest(root: Path) -> tuple[Path, str]:
    payload = root / "maxsim_cuda_benchmark.json"
    payload.write_text('{"status":"completed"}\n')
    payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "format": "colpali-triton-phase11-artifact-manifest-v1",
        "completed_at": "2026-07-26T00:00:00Z",
        "source_commit": SOURCE_COMMIT,
        "run_id": "phase11-test",
        "files": [
            {
                "path": payload.name,
                "sha256": payload_sha,
                "size_bytes": payload.stat().st_size,
            }
        ],
    }
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "COMPLETE.sha256").write_text(
        f"{manifest_sha}  artifact_manifest.json\n"
    )
    return manifest_path, manifest_sha


def test_offline_plan_states_every_remote_safety_boundary(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    plan = workflow.build_plan(spec, blockers=[])

    assert plan["status"] == "ready"
    assert plan["source_commit"] == SOURCE_COMMIT
    assert plan["remote_source_directory"].endswith(SOURCE_COMMIT)
    assert plan["remote_result_directory"].endswith(
        f"{SOURCE_COMMIT}/phase11-test"
    )
    assert plan["safety"] == {
        "plan_invokes_ssh": False,
        "plan_invokes_aws": False,
        "execution_creates_ec2_instances": False,
        "execution_terminates_or_stops_instance": False,
        "remote_results_overwritten": False,
        "local_results_overwritten": False,
        "instance_left_running_after_completion": True,
    }
    assert any("complete test suite" in step for step in plan["steps"])
    assert any("tuning matrix" in step for step in plan["steps"])
    assert any("canonical" in step for step in plan["steps"])
    assert any("profiler" in step for step in plan["steps"])


def test_plan_main_invokes_only_local_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("not-a-real-key")
    identity.chmod(0o600)
    commands: list[list[str]] = []

    def fake_run(command, *, capture_output=True):
        del capture_output
        command = list(command)
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "--verify"]:
            return subprocess.CompletedProcess(command, 0, SOURCE_COMMIT, "")
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(
            f"offline plan invoked unexpected command: {command}"
        )

    monkeypatch.setattr(workflow, "_run_local", fake_run)
    monkeypatch.setattr(
        workflow.shutil,
        "which",
        lambda executable: f"/mock/{executable}",
    )

    status = workflow.main(
        (
            "--host",
            "203.0.113.10",
            "--identity-file",
            str(identity),
            "--run-id",
            "phase11-test",
            "--local-result-base",
            str(tmp_path / "local"),
        )
    )

    assert status == 0
    assert [command[0] for command in commands] == ["git", "git"]
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready"
    assert output["safety"]["plan_invokes_ssh"] is False


def test_execution_requires_exact_acknowledgement_before_ssh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("not-a-real-key")
    identity.chmod(0o600)

    monkeypatch.setattr(
        workflow,
        "_source_state",
        lambda: (SOURCE_COMMIT, []),
    )
    monkeypatch.setattr(workflow, "_local_preflight", lambda spec: [])
    monkeypatch.setattr(
        workflow,
        "execute",
        lambda spec: pytest.fail("execute must not be called"),
    )

    status = workflow.main(
        (
            "--host",
            "203.0.113.10",
            "--identity-file",
            str(identity),
            "--run-id",
            "phase11-test",
            "--local-result-base",
            str(tmp_path / "local"),
            "--execute",
            "--acknowledge-run",
            "yes",
        )
    )

    assert status == 2
    assert workflow.RUN_ACKNOWLEDGEMENT in capsys.readouterr().err


def test_existing_local_result_is_a_preflight_blocker(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("key")
    identity.chmod(0o600)
    spec = _spec(tmp_path, identity_file=identity)
    spec.local_result_directory.mkdir(parents=True)

    blockers = workflow._local_preflight(spec)

    assert any("will not be overwritten" in item for item in blockers)


def test_manifest_verification_detects_payload_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    _, expected_sha = _write_manifest(root)

    actual_sha, records = workflow._verify_artifact_manifest(
        root,
        source_commit=SOURCE_COMMIT,
        run_id="phase11-test",
    )

    assert actual_sha == expected_sha
    assert [record["path"] for record in records] == [
        "maxsim_cuda_benchmark.json"
    ]
    (root / "maxsim_cuda_benchmark.json").write_text("tampered")
    with pytest.raises(workflow.WorkflowError, match="failed verification"):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
        )


def test_safe_extract_rejects_links_and_parent_traversal(
    tmp_path: Path,
) -> None:
    for name, member in (
        (
            "link",
            tarfile.TarInfo("phase11-test/result-link"),
        ),
        (
            "traversal",
            tarfile.TarInfo("phase11-test/../outside"),
        ),
    ):
        bundle = tmp_path / f"{name}.tar.gz"
        with tarfile.open(bundle, "w:gz") as archive:
            if name == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
            else:
                data = b"unsafe"
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
                continue
            archive.addfile(member)

        with pytest.raises(workflow.WorkflowError, match="unsafe member"):
            workflow._safe_extract(
                bundle,
                tmp_path / f"extract-{name}",
                "phase11-test",
            )


def test_safe_extract_copies_only_regular_files_with_safe_modes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "valid.tar.gz"
    payload = b"phase11-result"
    with tarfile.open(bundle, "w:gz") as archive:
        directory = tarfile.TarInfo("phase11-test/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o777
        archive.addfile(directory)
        member = tarfile.TarInfo("phase11-test/result.json")
        member.size = len(payload)
        member.mode = 0o6755
        archive.addfile(member, io.BytesIO(payload))

    root = workflow._safe_extract(
        bundle,
        tmp_path / "extract-valid",
        "phase11-test",
    )

    result = root / "result.json"
    assert result.read_bytes() == payload
    assert result.stat().st_mode & 0o7777 == 0o644
    assert root.stat().st_mode & 0o7777 == 0o755


def test_remote_job_has_bounded_complete_contract_and_no_termination() -> None:
    script = JOB_PATH.read_text()

    assert "python -m pytest" in script
    assert "benchmarks/tune_triton_maxsim.py --preflight" in script
    assert "--output /phase11-results/triton_tuning.json" in script
    assert "benchmarks/benchmark_maxsim.py" in script
    assert "--output /phase11-results/maxsim_cuda_benchmark.json" in script
    assert "timeout --signal=TERM --kill-after=30s 300s" in script
    assert "artifact_manifest.json" in script
    assert "COMPLETE.sha256" in script
    assert "terminate-instances" not in script
    assert "l4_lifecycle.py terminate" not in script


def test_profiler_probe_is_one_fixed_colpali_shape() -> None:
    source = PROFILE_PATH.read_text()

    assert "(1, 15, 128)" in source
    assert "(32, 1030, 128)" in source
    assert "torch.inference_mode()" in source
    assert source.count("maxsim_triton(") == 1
