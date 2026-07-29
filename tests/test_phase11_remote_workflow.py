from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / "infra" / "aws" / "phase11_remote.py"
FINALIZER_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "phase11_finalize_artifacts.py"
)
JOB_PATH = PROJECT_ROOT / "infra" / "aws" / "run_phase11_job.sh"
PROFILE_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "profile_phase11_maxsim.py"
)
CUDA_ASSERTION_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "assert_phase11_cuda_tests.py"
)
PROFILE_STATUS_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "phase11_profile_status.py"
)
DERIVE_WINNER_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "derive_phase11_winner.py"
)
L4_ASSERTION_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "assert_phase11_l4.py"
)
CONFIG_PATH = PROJECT_ROOT / "infra" / "aws" / "l4_environment.json"
SOURCE_COMMIT = "a" * 40
EXPECTED_INSTANCE_ID = "i-0123456789abcdef0"
EXPECTED_INSTANCE_TYPE = "g6.xlarge"
EXPECTED_REGION = "us-west-2"
EXPECTED_AMI_ID = "ami-0123456789abcdef0"
EXPECTED_ACCOUNT_ID = "123456789012"
LAUNCH_RECEIPT_SHA256 = "b" * 64
EXPECTED_CONFIG_FINGERPRINT = (
    "b06d77f646da2d73eed72f171908f2232aae9b1e3c1ec9c3badf4b124817c6be"
)


def _memory_record() -> dict[str, int]:
    return {
        "baseline_allocated_bytes": 1024,
        "baseline_reserved_bytes": 2048,
        "peak_allocated_bytes": 4096,
        "peak_reserved_bytes": 8192,
        "incremental_peak_allocated_bytes": 3072,
        "incremental_peak_reserved_bytes": 6144,
    }


def _timing_record(
    latency: float,
    *,
    trials: int,
    iterations: int,
    tuning: bool,
) -> dict:
    trial_records = [
        {
            **({"status": "completed"} if tuning else {}),
            "trial_index": trial_index,
            "cuda_event_latency_ms": [latency] * iterations,
            "memory": _memory_record(),
        }
        for trial_index in range(trials)
    ]
    samples = trials * iterations
    record = {
        "measured_iterations_per_trial": iterations,
        "summary": {
            "sample_count": samples,
            "latency_ms": {"p50": latency},
        },
        "trials": trial_records,
    }
    if tuning:
        record.update(
            {
                "trial_count_requested": trials,
                "trial_count_completed": trials,
                "total_cuda_event_latency_ms": latency * samples,
            }
        )
    else:
        record["trial_count"] = trials
    return record


def _write_complete_result_evidence(root: Path) -> tuple[dict, str]:
    tuning_config_path = PROJECT_ROOT / "configs" / "triton_tuning.json"
    benchmark_config_path = (
        PROJECT_ROOT / "configs" / "maxsim_benchmark.json"
    )
    tuning_config = json.loads(tuning_config_path.read_text())
    base_benchmark = json.loads(benchmark_config_path.read_text())
    candidates = tuning_config["candidates"]
    candidate_ids = [item["candidate_id"] for item in candidates]
    winner_config = candidates[0]
    winner_id = winner_config["candidate_id"]
    launch = {
        key: winner_config[key]
        for key in (
            "block_query_tokens",
            "block_document_tokens",
            "block_embedding_dimension",
            "num_warps",
            "num_stages",
        )
    }
    source_record = {
        "effective_commit": SOURCE_COMMIT,
        "consistent": True,
        "sources": {
            "git": None,
            "environment": SOURCE_COMMIT,
            "marker_file": SOURCE_COMMIT,
        },
    }
    tuning_results = []
    tuning_trials = tuning_config["timing"]["trials"]
    tuning_iterations = tuning_config["timing"]["measured_iterations"]
    for dtype_spec in tuning_config["dtypes"]:
        for case in tuning_config["cases"]:
            case_record = dict(case)
            case_record["query_document_pairs"] = (
                case["query_batch_size"] * case["document_batch_size"]
            )
            candidate_records = []
            for candidate_index, candidate in enumerate(candidates):
                latency = float(candidate_index + 1)
                candidate_launch = {
                    key: candidate[key] for key in launch
                }
                candidate_records.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "launch_configuration": candidate_launch,
                        "status": "completed",
                        "numerical_gate": {
                            "status": "passed",
                            "agreement": {
                                "reference_provider": "torch",
                                "shape_matches": True,
                                "finite": True,
                                "agreement": True,
                            },
                        },
                        "timing": _timing_record(
                            latency,
                            trials=tuning_trials,
                            iterations=tuning_iterations,
                            tuning=True,
                        ),
                    }
                )
            tuning_results.append(
                {
                    "case": case_record,
                    "dtype": dtype_spec["name"],
                    "candidates": candidate_records,
                }
            )
    tuning_config_sha = hashlib.sha256(
        tuning_config_path.read_bytes()
    ).hexdigest()
    tuning = {
        "status": "completed",
        "selection": {
            "canonical_full_matrix": True,
            "candidates": candidate_ids,
            "dtypes": [item["name"] for item in tuning_config["dtypes"]],
            "cases": [item["name"] for item in tuning_config["cases"]],
        },
        "provenance": {
            "config_file_sha256": tuning_config_sha,
            "source_sha256": {
                "configs/triton_tuning.json": tuning_config_sha,
            },
            "source_commit": source_record,
        },
        "protocol": {"config": tuning_config},
        "winner": {
            "candidate_id": winner_id,
            "configuration": winner_config,
            "overall": {
                "status": "selected",
                "winner_candidate_id": winner_id,
                "workload_count": len(tuning_results),
            },
        },
        "results": tuning_results,
    }
    tuning_path = root / "triton_tuning.json"
    tuning_path.write_text(json.dumps(tuning, sort_keys=True) + "\n")

    derived = json.loads(json.dumps(base_benchmark))
    derived["benchmark_id"] += f"-tuned-{winner_id}"
    derived["triton_launch"] = launch
    derived_path = root / "maxsim_benchmark_tuned_config.json"
    derived_path.write_text(json.dumps(derived, sort_keys=True) + "\n")
    winner = {
        "format": "colpali-triton-phase11-tuning-winner-v1",
        "source_commit": SOURCE_COMMIT,
        "candidate_id": winner_id,
        "launch_configuration": launch,
        "base_benchmark_id": base_benchmark["benchmark_id"],
        "derived_benchmark_id": derived["benchmark_id"],
        "canonical_case_count": len(base_benchmark["cases"]),
        "canonical_dtype_count": len(base_benchmark["dtypes"]),
        "tuning_result_sha256": hashlib.sha256(
            tuning_path.read_bytes()
        ).hexdigest(),
        "base_benchmark_config_sha256": hashlib.sha256(
            benchmark_config_path.read_bytes()
        ).hexdigest(),
        "derived_benchmark_config_sha256": hashlib.sha256(
            derived_path.read_bytes()
        ).hexdigest(),
    }
    winner_path = root / "tuning_winner.json"
    winner_path.write_text(json.dumps(winner, sort_keys=True) + "\n")
    winner_sha = hashlib.sha256(winner_path.read_bytes()).hexdigest()

    benchmark_results = []
    benchmark_trials = base_benchmark["timing"]["trials"]
    benchmark_iterations = base_benchmark["timing"]["measured_iterations"]
    for dtype_spec in base_benchmark["dtypes"]:
        for case in base_benchmark["cases"]:
            case_record = dict(case)
            case_record["query_document_pairs"] = (
                case["query_batch_size"] * case["document_batch_size"]
            )
            providers = []
            for provider_name, latency in (("torch", 2.0), ("triton", 1.0)):
                providers.append(
                    {
                        "provider": provider_name,
                        "launch_configuration": (
                            launch if provider_name == "triton" else None
                        ),
                        "correctness": {
                            "reference_provider": "torch",
                            "finite": True,
                            "agreement": True,
                            "absolute_tolerance": dtype_spec[
                                "absolute_tolerance"
                            ],
                            "relative_tolerance": dtype_spec[
                                "relative_tolerance"
                            ],
                        },
                        "timing": _timing_record(
                            latency,
                            trials=benchmark_trials,
                            iterations=benchmark_iterations,
                            tuning=False,
                        ),
                        "memory": {
                            "maximum_peak_allocated_bytes": 4096,
                            "maximum_peak_reserved_bytes": 8192,
                            "maximum_incremental_peak_allocated_bytes": 3072,
                            "maximum_incremental_peak_reserved_bytes": 6144,
                        },
                    }
                )
            benchmark_results.append(
                {
                    "case": case_record,
                    "dtype": dtype_spec["name"],
                    "providers": providers,
                }
            )
    benchmark = {
        "status": "completed",
        "selection": {
            "canonical_full_matrix": True,
            "providers": ["torch", "triton"],
            "dtypes": [item["name"] for item in base_benchmark["dtypes"]],
            "cases": [item["name"] for item in base_benchmark["cases"]],
        },
        "provenance": {
            "source_commit": source_record,
            "config_file_sha256": hashlib.sha256(
                derived_path.read_bytes()
            ).hexdigest(),
        },
        "protocol": {"config": derived},
        "results": benchmark_results,
    }
    (root / "maxsim_cuda_benchmark.json").write_text(
        json.dumps(benchmark, sort_keys=True) + "\n"
    )
    return winner, winner_sha


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module("phase11_remote_workflow", WORKFLOW_PATH)
finalizer = _load_module("phase11_recovery_finalizer", FINALIZER_PATH)
cuda_assertion = _load_module(
    "phase11_cuda_test_assertion", CUDA_ASSERTION_PATH
)
profile_status = _load_module(
    "phase11_profile_status", PROFILE_STATUS_PATH
)
derive_winner = _load_module(
    "phase11_derive_winner", DERIVE_WINNER_PATH
)
l4_assertion = _load_module(
    "phase11_l4_assertion", L4_ASSERTION_PATH
)


def _launch_receipt_record() -> dict:
    contract = json.loads(CONFIG_PATH.read_text())
    user_data = PROJECT_ROOT / contract["bootstrap"]["user_data_path"]
    return {
        "schema_version": 1,
        "operation": "launch",
        "created_at_utc": "2026-07-26T00:00:00+00:00",
        "config_fingerprint": workflow._canonical_fingerprint(contract),
        "source_commit": SOURCE_COMMIT,
        "user_data_sha256": hashlib.sha256(user_data.read_bytes()).hexdigest(),
        "resolved_ami": {
            "image_id": EXPECTED_AMI_ID,
            "name": (
                "Deep Learning Base OSS Nvidia Driver GPU AMI "
                "(Ubuntu 22.04) 20260726"
            ),
            "creation_date": "2026-07-26T00:00:00.000Z",
            "architecture": "x86_64",
            "root_device_name": "/dev/sda1",
            "owner_id": "898082745236",
        },
        "instance_id": EXPECTED_INSTANCE_ID,
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "region": EXPECTED_REGION,
        "subnet_id": "subnet-0123456789abcdef0",
        "security_group_id": "sg-0123456789abcdef0",
        "public_ip_requested": True,
        "root_volume_gib": 100,
        "client_token": "colpali-p10-example",
        "container": contract["container"],
        "bootstrap": contract["bootstrap"],
        "tags": {**contract["tags"], "Name": "colpali-triton-phase10"},
        "launch_response": {
            "OwnerId": EXPECTED_ACCOUNT_ID,
            "ReservationId": "r-0123456789abcdef0",
            "Groups": [],
            "Instances": [
                {
                    "InstanceId": EXPECTED_INSTANCE_ID,
                    "InstanceType": EXPECTED_INSTANCE_TYPE,
                    "ImageId": EXPECTED_AMI_ID,
                    "SubnetId": "subnet-0123456789abcdef0",
                    "SecurityGroups": [
                        {
                            "GroupId": "sg-0123456789abcdef0",
                            "GroupName": "colpali-benchmark",
                        }
                    ],
                    "Placement": {"AvailabilityZone": "us-west-2a"},
                    "Architecture": "x86_64",
                }
            ],
        },
    }


def _write_launch_receipt(path: Path, record: dict | None = None) -> Path:
    path.write_text(
        json.dumps(record or _launch_receipt_record(), sort_keys=True) + "\n"
    )
    return path


def _expected_host() -> dict[str, str]:
    return {
        "instance_id": EXPECTED_INSTANCE_ID,
        "instance_type": EXPECTED_INSTANCE_TYPE,
        "region": EXPECTED_REGION,
        "ami_id": EXPECTED_AMI_ID,
        "account_id": EXPECTED_ACCOUNT_ID,
        "launch_receipt_sha256": LAUNCH_RECEIPT_SHA256,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
    }


def _host_metadata_record() -> dict:
    contract = json.loads(CONFIG_PATH.read_text())
    image = [
        {
            "Id": "sha256:" + "c" * 64,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {
                "Env": [
                    f"COLPALI_SOURCE_COMMIT={SOURCE_COMMIT}",
                ],
                "Labels": {
                    "org.opencontainers.image.revision": SOURCE_COMMIT,
                    "org.opencontainers.image.source": "local-worktree",
                    "org.opencontainers.image.base.name": contract["container"][
                        "base_tag"
                    ],
                    "org.opencontainers.image.base.digest": contract[
                        "container"
                    ]["base_manifest_digest"],
                },
            },
        }
    ]
    bootstrap = {
        "base_image": contract["container"]["base_reference"],
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
    successful = lambda stdout: {
        "command": ["example"],
        "available": True,
        "returncode": 0,
        "stdout": stdout,
        "stderr": "",
    }
    record = {
        "format": "colpali-triton-phase11-host-metadata-v1",
        "captured_at": "2026-07-26T00:00:00Z",
        "source_commit": SOURCE_COMMIT,
        "run_id": "phase11-test",
        "container_image_name": "colpali-triton:phase11-test",
        "host": {"architecture": "x86_64"},
        "ec2": {
            "available": True,
            "identity_document": {
                "accountId": EXPECTED_ACCOUNT_ID,
                "architecture": "x86_64",
                "availabilityZone": "us-west-2a",
                "imageId": EXPECTED_AMI_ID,
                "instanceId": EXPECTED_INSTANCE_ID,
                "instanceType": EXPECTED_INSTANCE_TYPE,
                "privateIp": "10.0.0.5",
                "region": EXPECTED_REGION,
                "version": "2017-09-30",
            },
        },
        "nvidia_smi": successful(
            "0, NVIDIA L4, GPU-12345678-abcd-1234-abcd-1234567890ab, "
            "00000000:00:1E.0, 580.159.04, 23034 MiB, 8.9, P8, "
            "300 MHz, 405 MHz, 72.00 W"
        ),
        "docker_version": successful(
            json.dumps(
                {
                    "Client": {"Version": "29.5.3"},
                    "Server": {"Version": "29.5.3"},
                }
            )
        ),
        "docker_image": successful(json.dumps(image)),
        "bootstrap": {
            "available": True,
            "path": "/var/lib/colpali-triton/bootstrap-environment.json",
            "sha256": "d" * 64,
            "record": bootstrap,
        },
        "environment_contract": {"available": True},
    }
    record["validation"] = workflow.validate_host_evidence(
        record,
        contract,
        expected_instance_id=EXPECTED_INSTANCE_ID,
        expected_instance_type=EXPECTED_INSTANCE_TYPE,
        expected_region=EXPECTED_REGION,
        expected_ami_id=EXPECTED_AMI_ID,
        expected_account_id=EXPECTED_ACCOUNT_ID,
    )
    assert record["validation"]["status"] == "passed"
    return record


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
        "launch_receipt_path": tmp_path / "aws_launch_receipt.json",
        "launch_receipt_sha256": LAUNCH_RECEIPT_SHA256,
        "launch_config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "expected_instance_id": EXPECTED_INSTANCE_ID,
        "expected_instance_type": EXPECTED_INSTANCE_TYPE,
        "expected_region": EXPECTED_REGION,
        "expected_ami_id": EXPECTED_AMI_ID,
        "expected_account_id": EXPECTED_ACCOUNT_ID,
    }
    values.update(overrides)
    return workflow.RemoteSpec(**values)


def _finalizer_args(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "source"
    result = tmp_path / "remote-result"
    control = tmp_path / "remote-result.control"
    source.mkdir()
    result.mkdir()
    control.mkdir()
    (source / ".source_commit").write_text(SOURCE_COMMIT + "\n")
    return SimpleNamespace(
        source_directory=source,
        result_directory=result,
        control_directory=control,
        source_commit=SOURCE_COMMIT,
        run_id="phase11-test",
        expected_instance_id=EXPECTED_INSTANCE_ID,
        expected_instance_type=EXPECTED_INSTANCE_TYPE,
        expected_region=EXPECTED_REGION,
        expected_ami_id=EXPECTED_AMI_ID,
        expected_account_id=EXPECTED_ACCOUNT_ID,
        launch_receipt_sha256=LAUNCH_RECEIPT_SHA256,
        launch_config_fingerprint=EXPECTED_CONFIG_FINGERPRINT,
        job_exit_code=124,
        recovery_source="detached-wrapper",
    )


@pytest.mark.parametrize(
    "run_id",
    (
        "other-run.control",
        "other-run.tar.gz",
        "other-run.tar.gz.sha256",
    ),
)
def test_run_id_cannot_alias_another_runs_remote_artifacts(
    tmp_path: Path,
    run_id: str,
) -> None:
    args = SimpleNamespace(
        host="203.0.113.10",
        user="ubuntu",
        port=22,
        identity_file=(tmp_path / "benchmark.pem").resolve(),
        launch_receipt=(tmp_path / "launch.json").resolve(),
        local_result_base=(tmp_path / "local").resolve(),
        remote_source_base="/home/ubuntu/colpali-triton",
        remote_result_base="/home/ubuntu/colpali-phase11-results",
        run_id=run_id,
        image_name=None,
    )

    with pytest.raises(workflow.WorkflowError, match="reserved"):
        workflow._spec_from_args(args, SOURCE_COMMIT, None)


def _write_manifest(
    root: Path,
    *,
    status: str = "complete",
    with_profile: bool = True,
) -> tuple[Path, str]:
    winner, winner_sha = _write_complete_result_evidence(root)
    cross_check = workflow.validate_result_contract(
        root,
        source_commit=SOURCE_COMMIT,
    )
    assert cross_check["status"] == "passed"
    (root / "result_contract_validation.json").write_text(
        json.dumps(cross_check, sort_keys=True) + "\n"
    )
    (root / "hardware_gate.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "device": {
                    "name": "NVIDIA L4",
                    "compute_capability": [8, 9],
                    "total_memory_bytes": 23 * 1024**3,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "cuda_test_assertion.json").write_text(
        json.dumps(
            {
                "format": (
                    "colpali-triton-phase11-cuda-test-assertion-v1"
                ),
                "status": "passed",
                "required_counts": workflow.REQUIRED_CUDA_COUNTS,
                "observed_counts": workflow.REQUIRED_CUDA_COUNTS,
                "skipped": [],
                "failed": [],
                "errors": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "host_metadata.json").write_text(
        json.dumps(_host_metadata_record(), sort_keys=True) + "\n"
    )
    if with_profile:
        (root / "maxsim_profile.ncu-rep").write_bytes(b"real-profile")
        metadata = {
            "format": "colpali-triton-phase11-profile-probe-v2",
            "source_commit": SOURCE_COMMIT,
            "winner_candidate_id": winner["candidate_id"],
            "winner_record_sha256": winner_sha,
            "launch_configuration": winner["launch_configuration"],
        }
        (root / "profile_probe_ncu.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n"
        )
    profiling = {
        "succeeded": with_profile,
        "profiler": "ncu" if with_profile else None,
        "report": "maxsim_profile.ncu-rep" if with_profile else None,
        "metadata": "profile_probe_ncu.json" if with_profile else None,
        "winner_record": "tuning_winner.json",
        "winner_record_sha256": winner_sha,
        "attempts": [],
    }
    steps = {
        name: {"exit_code": 0, "succeeded": True}
        for name in workflow.MANDATORY_PHASE11_STEPS
    }
    (root / "step_status.json").write_text(
        json.dumps(
            {
                "format": "colpali-triton-phase11-step-status-v1",
                "steps": steps,
            },
            sort_keys=True,
        )
        + "\n"
    )
    records = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "format": "colpali-triton-phase11-artifact-manifest-v2",
        "sealed_at": "2026-07-26T00:00:00Z",
        "status": status,
        "failure_reasons": (
            [] if status == "complete" else ["profiling failed"]
        ),
        "profiling": profiling,
        "steps": steps,
        "source_commit": SOURCE_COMMIT,
        "run_id": "phase11-test",
        "expected_host": _expected_host(),
        "files": records,
    }
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "SEALED.sha256").write_text(
        f"{manifest_sha}  artifact_manifest.json\n"
    )
    return manifest_path, manifest_sha


def _reseal_manifest(root: Path, manifest: dict) -> None:
    records = []
    for path in sorted(root.iterdir()):
        if (
            not path.is_file()
            or path.name in {"artifact_manifest.json", "SEALED.sha256"}
        ):
            continue
        records.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest["files"] = records
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (root / "SEALED.sha256").write_text(
        f"{manifest_sha}  artifact_manifest.json\n"
    )


def _write_profile_winner(
    root: Path,
    *,
    profiler: str | None = None,
) -> Path:
    winner = {
        "format": "colpali-triton-phase11-tuning-winner-v1",
        "source_commit": SOURCE_COMMIT,
        "candidate_id": "winner",
        "launch_configuration": {
            "block_query_tokens": 16,
            "block_document_tokens": 32,
            "block_embedding_dimension": 128,
            "num_warps": 4,
            "num_stages": 1,
        },
    }
    winner_path = root / "tuning_winner.json"
    winner_path.write_text(json.dumps(winner, sort_keys=True) + "\n")
    if profiler is not None:
        winner_sha = hashlib.sha256(winner_path.read_bytes()).hexdigest()
        metadata = {
            "format": "colpali-triton-phase11-profile-probe-v2",
            "source_commit": SOURCE_COMMIT,
            "winner_candidate_id": "winner",
            "winner_record_sha256": winner_sha,
            "launch_configuration": winner["launch_configuration"],
        }
        (root / f"profile_probe_{profiler}.json").write_text(
            json.dumps(metadata, sort_keys=True) + "\n"
        )
    return winner_path


def test_offline_plan_states_every_remote_safety_boundary(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    plan = workflow.build_plan(spec, blockers=[])

    assert plan["status"] == "ready"
    assert plan["source_commit"] == SOURCE_COMMIT
    assert plan["expected_host"] == _expected_host()
    assert plan["launch_receipt"]["valid"] is True
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
        "remote_job_survives_client_disconnect": True,
        "same_run_id_is_resumable": True,
        "instance_left_running_after_completion": True,
    }
    assert any("zero-skip parity" in step for step in plan["steps"])
    assert any("tuning matrix" in step for step in plan["steps"])
    assert any("canonical" in step for step in plan["steps"])
    assert any("ncu then nsys" in step for step in plan["steps"])


def test_launch_receipt_binds_exact_lifecycle_and_response_fields(
    tmp_path: Path,
) -> None:
    path = _write_launch_receipt(tmp_path / "launch.json")

    receipt = workflow._load_launch_receipt(
        path,
        source_commit=SOURCE_COMMIT,
    )

    assert receipt.path == path.resolve()
    assert receipt.instance_id == EXPECTED_INSTANCE_ID
    assert receipt.instance_type == EXPECTED_INSTANCE_TYPE
    assert receipt.region == EXPECTED_REGION
    assert receipt.ami_id == EXPECTED_AMI_ID
    assert receipt.account_id == EXPECTED_ACCOUNT_ID
    assert receipt.config_fingerprint == EXPECTED_CONFIG_FINGERPRINT
    assert receipt.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "operation",
        "config_fingerprint",
        "source_commit",
        "container",
        "response_instance",
        "response_account_invalid",
    ),
)
def test_launch_receipt_rejects_contract_or_response_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt = _launch_receipt_record()
    if mutation == "schema":
        receipt["schema_version"] = 2
    elif mutation == "operation":
        receipt["operation"] = "status"
    elif mutation == "config_fingerprint":
        receipt["config_fingerprint"] = "0" * 64
    elif mutation == "source_commit":
        receipt["source_commit"] = "0" * 40
    elif mutation == "container":
        receipt["container"]["triton_version"] = "0.0.0"
    elif mutation == "response_instance":
        receipt["launch_response"]["Instances"][0]["InstanceId"] = (
            "i-11111111111111111"
        )
    else:
        receipt["launch_response"]["OwnerId"] = "not-an-account"
    path = _write_launch_receipt(tmp_path / "launch.json", receipt)

    with pytest.raises(workflow.WorkflowError, match="launch receipt"):
        workflow._load_launch_receipt(path, source_commit=SOURCE_COMMIT)


def test_launch_receipt_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "launch.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1, "operation": "launch"}'
    )

    with pytest.raises(workflow.WorkflowError, match="duplicate JSON key"):
        workflow._load_launch_receipt(path, source_commit=SOURCE_COMMIT)


def test_plan_main_invokes_only_local_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("not-a-real-key")
    identity.chmod(0o600)
    launch_receipt = _write_launch_receipt(tmp_path / "launch.json")
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
            "--launch-receipt",
            str(launch_receipt),
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


def test_missing_launch_receipt_blocks_plan_without_external_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("key")
    identity.chmod(0o600)
    monkeypatch.setattr(
        workflow,
        "_source_state",
        lambda: (SOURCE_COMMIT, []),
    )
    monkeypatch.setattr(workflow, "_local_preflight", lambda spec: [])

    status = workflow.main(
        (
            "--host",
            "203.0.113.10",
            "--identity-file",
            str(identity),
            "--launch-receipt",
            str(tmp_path / "missing.json"),
            "--run-id",
            "phase11-test",
            "--local-result-base",
            str(tmp_path / "local"),
        )
    )

    plan = json.loads(capsys.readouterr().out)
    assert status == 0
    assert plan["status"] == "blocked"
    assert plan["expected_host"] is None
    assert plan["launch_receipt"]["valid"] is False
    assert any("launch receipt" in blocker for blocker in plan["blockers"])


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


def test_execution_requires_a_valid_launch_receipt_before_ssh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("key")
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
            "--launch-receipt",
            str(tmp_path / "missing.json"),
            "--run-id",
            "phase11-test",
            "--local-result-base",
            str(tmp_path / "local"),
            "--execute",
            "--acknowledge-run",
            workflow.RUN_ACKNOWLEDGEMENT,
        )
    )

    assert status == 2
    assert "launch receipt" in capsys.readouterr().err


def test_existing_local_result_is_a_preflight_blocker(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("key")
    identity.chmod(0o600)
    spec = _spec(tmp_path, identity_file=identity)
    spec.local_result_directory.mkdir(parents=True)

    blockers = workflow._local_preflight(spec)

    assert any(
        "is not an exact verified reusable result" in item
        for item in blockers
    )


def test_ssh_connections_have_bounded_connect_and_keepalive_options(
    tmp_path: Path,
) -> None:
    options = workflow._ssh_options(_spec(tmp_path))

    assert "ConnectTimeout=15" in options
    assert "ConnectionAttempts=3" in options
    assert "ServerAliveInterval=30" in options
    assert "ServerAliveCountMax=3" in options


def test_source_archive_uses_exact_commit_and_hashes_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = []
    timeouts = []

    def fake_run(command, **kwargs):
        commands.append(tuple(command))
        timeouts.append(kwargs["timeout"])
        output = kwargs["stdout"]
        payload = b"committed source\n"
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("tracked.txt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    archive = workflow._prepare_source_archive(
        spec,
        tmp_path / "source.tar",
    )
    expected_content = hashlib.sha256()
    expected_content.update(b"tracked.txt\0")
    expected_content.update(str(len(b"committed source\n")).encode("ascii"))
    expected_content.update(b"\0committed source\n\0")

    assert commands == [
        ("git", "archive", "--format=tar", spec.source_commit)
    ]
    assert timeouts == [workflow.SOURCE_ARCHIVE_TIMEOUT_SECONDS]
    assert "HEAD" not in commands[0]
    assert archive.archive_sha256 == hashlib.sha256(
        archive.path.read_bytes()
    ).hexdigest()
    assert archive.content_sha256 == expected_content.hexdigest()


def test_remote_source_reuse_reverifies_hashes_and_read_only_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    commands = []

    def fake_ssh(spec_value, command, *, capture_output=True):
        del spec_value, capture_output
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "ready", "")

    monkeypatch.setattr(workflow, "_run_ssh", fake_ssh)

    assert workflow._remote_source_state(spec, archive) == "ready"
    assert archive.archive_sha256 in commands[0]
    assert archive.content_sha256 in commands[0]
    assert ".source_archive_sha256" in commands[0]
    assert ".source_content_sha256" in commands[0]
    assert "-perm /222" in commands[0]
    assert "python3 -c" in commands[0]


def test_remote_source_tamper_is_rejected_instead_of_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    monkeypatch.setattr(
        workflow,
        "_run_ssh",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "invalid", ""
        ),
    )

    assert workflow._remote_source_state(spec, archive) == "invalid"


def test_source_upload_has_a_separate_transfer_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "source.tar"
    archive_path.write_bytes(b"archive")
    archive = workflow.SourceArchive(
        path=archive_path,
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    observed = []

    def time_out(command, **kwargs):
        observed.append((command, kwargs["timeout"]))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(workflow.subprocess, "run", time_out)

    with pytest.raises(workflow.WorkflowError, match="transfer bound"):
        workflow._upload_archive(_spec(tmp_path), archive)

    assert observed[0][1] == workflow.SOURCE_UPLOAD_TIMEOUT_SECONDS
    assert "sha256sum --check - >/dev/null" in observed[0][0][-1]


def test_detached_job_is_systemd_managed_and_overall_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = []

    def fake_ssh(spec_value, command, *, capture_output=True):
        del spec_value, capture_output
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "started", "")

    monkeypatch.setattr(workflow, "_run_ssh", fake_ssh)

    workflow._start_remote_job(spec)

    command = commands[0]
    assert "sudo systemd-run" in command
    assert f"--unit={spec.remote_unit_name}" in command
    assert f"--uid={spec.user}" in command
    assert (
        f"RuntimeMaxSec={workflow.REMOTE_JOB_OUTER_TIMEOUT_SECONDS}s"
        in command
    )
    assert (
        f"/usr/bin/timeout --signal=TERM --kill-after=60s "
        f"{workflow.REMOTE_JOB_TIMEOUT_SECONDS}s"
    ) in command
    assert "phase11_finalize_artifacts.py" in command
    assert (
        f"/usr/bin/timeout --signal=TERM --kill-after=30s "
        f"{workflow.REMOTE_FINALIZER_TIMEOUT_SECONDS}s"
    ) in command
    assert "--recovery-source detached-wrapper" in command
    assert "finalizer.exit_code" in command
    for flag, value in (
        ("--expected-instance-id", EXPECTED_INSTANCE_ID),
        ("--expected-instance-type", EXPECTED_INSTANCE_TYPE),
        ("--expected-region", EXPECTED_REGION),
        ("--expected-ami-id", EXPECTED_AMI_ID),
        ("--expected-account-id", EXPECTED_ACCOUNT_ID),
        ("--launch-receipt-sha256", LAUNCH_RECEIPT_SHA256),
        ("--launch-config-fingerprint", EXPECTED_CONFIG_FINGERPRINT),
    ):
        assert flag in command
        assert value in command
    assert "terminate" not in command


def test_remote_state_treats_systemd_transitions_as_live_and_checks_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands = []
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="c" * 64,
        content_sha256="d" * 64,
    )

    def fake_ssh(spec_value, command, *, capture_output=True):
        del spec_value, capture_output
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "running", "")

    monkeypatch.setattr(workflow, "_run_ssh", fake_ssh)

    assert workflow._remote_run_state(_spec(tmp_path), archive) == "running"
    command = commands[0]
    for live_state in ("active", "activating", "reloading", "deactivating"):
        assert f'test "${{unit_state}}" = {live_state}' in command
    assert "sha256sum --check SEALED.sha256" in command
    assert "needs_finalization" in command


def test_poll_fallback_finalizes_then_requeries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="c" * 64,
        content_sha256="d" * 64,
    )
    states = iter(("needs_finalization", "sealed"))
    finalized = []
    monkeypatch.setattr(
        workflow,
        "_remote_run_state",
        lambda *args: next(states),
    )
    monkeypatch.setattr(
        workflow,
        "_finalize_remote_run",
        lambda *args: finalized.append(args),
    )

    assert workflow._wait_for_remote_run(_spec(tmp_path), archive) == "sealed"
    assert len(finalized) == 1


def test_poll_fallback_finalizer_is_separately_bounded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_ssh(
        spec_value,
        command,
        *,
        capture_output=True,
        timeout_seconds=workflow.SSH_COMMAND_TIMEOUT_SECONDS,
    ):
        del spec_value, capture_output
        calls.append((command, timeout_seconds))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(workflow, "_run_ssh", fake_ssh)

    workflow._finalize_remote_run(_spec(tmp_path))

    command, timeout_seconds = calls[0]
    assert "phase11_finalize_artifacts.py" in command
    assert "--recovery-source poll-fallback" in command
    assert (
        f"timeout --signal=TERM --kill-after=30s "
        f"{workflow.REMOTE_FINALIZER_TIMEOUT_SECONDS}s"
    ) in command
    assert timeout_seconds == workflow.REMOTE_FINALIZER_TIMEOUT_SECONDS + 60


def test_recovery_finalizer_seals_verifiable_incomplete_bundle_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _finalizer_args(tmp_path)
    partial = args.result_directory / "step_status.json"
    partial.write_text('{"partial": true}\n')
    (args.result_directory / "partial.log").write_text("timed out\n")
    cleanup_calls = []

    def fake_cleanup(root, *, run_id, source_commit):
        cleanup_calls.append((root, run_id, source_commit))
        return {
            "format": "colpali-triton-phase11-emergency-cleanup-v1",
            "recorded_at": "2026-07-26T00:00:00Z",
            "scope": {},
            "removed_container_ids": [],
            "skipped_unmatched_container_ids": [],
            "remaining_labeled_container_ids": [],
            "safe_to_seal": True,
            "errors": [],
        }

    monkeypatch.setattr(finalizer, "_cleanup_containers", fake_cleanup)

    result = finalizer.finalize(args)
    manifest_before = (
        args.result_directory / "artifact_manifest.json"
    ).read_bytes()
    repeated = finalizer.finalize(args)

    assert result["status"] == "sealed_incomplete"
    assert repeated == {"status": "already_sealed"}
    assert (
        args.result_directory / "artifact_manifest.json"
    ).read_bytes() == manifest_before
    assert len(cleanup_calls) == 1
    assert list(
        (args.result_directory / ".finalizer-quarantine").rglob(
            "step_status.json"
        )
    )
    _, _, status, reasons = workflow._verify_artifact_manifest(
        args.result_directory,
        source_commit=SOURCE_COMMIT,
        run_id="phase11-test",
        expected_host=_expected_host(),
    )
    assert status == "incomplete"
    assert any("exit_code=124" in reason for reason in reasons)
    assert set(
        json.loads(
            (args.result_directory / "step_status.json").read_text()
        )["steps"]
    ) == set(workflow.MANDATORY_PHASE11_STEPS)


def test_recovery_finalizer_does_not_preserve_wrong_identity_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _finalizer_args(tmp_path)
    wrong_manifest = {
        "format": "colpali-triton-phase11-artifact-manifest-v2",
        "status": "incomplete",
        "source_commit": SOURCE_COMMIT,
        "run_id": "phase11-test",
        "expected_host": {**_expected_host(), "instance_id": "i-deadbeef"},
    }
    manifest_path = args.result_directory / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(wrong_manifest, sort_keys=True) + "\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (args.result_directory / "SEALED.sha256").write_text(
        f"{digest}  artifact_manifest.json\n"
    )
    monkeypatch.setattr(
        finalizer,
        "_cleanup_containers",
        lambda *args, **kwargs: {
            "format": "colpali-triton-phase11-emergency-cleanup-v1",
            "recorded_at": "2026-07-26T00:00:00Z",
            "scope": {},
            "removed_container_ids": [],
            "skipped_unmatched_container_ids": [],
            "remaining_labeled_container_ids": [],
            "safe_to_seal": True,
            "errors": [],
        },
    )

    assert finalizer.finalize(args)["status"] == "sealed_incomplete"
    recovered = json.loads(manifest_path.read_text())
    assert recovered["expected_host"] == _expected_host()
    assert list(
        (args.result_directory / ".finalizer-quarantine").rglob(
            "artifact_manifest.json"
        )
    )


def test_emergency_cleanup_removes_only_exactly_labeled_containers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    cid_directory = root / ".docker-cids"
    cid_directory.mkdir(parents=True)
    matching = "a" * 64
    mismatched = "b" * 64
    (cid_directory / "matching.cid").write_text(matching)
    (cid_directory / "mismatched.cid").write_text(mismatched)
    calls = []
    removed = set()

    def fake_docker(arguments):
        arguments = tuple(arguments)
        calls.append(arguments)
        if arguments[:2] == ("ps", "-aq"):
            output = "" if matching in removed else matching + "\n"
            return subprocess.CompletedProcess(arguments, 0, output, "")
        if arguments[0] == "inspect":
            labels = (
                "phase11-test\t" + SOURCE_COMMIT
                if arguments[-1] == matching
                else "another-run\t" + SOURCE_COMMIT
            )
            return subprocess.CompletedProcess(arguments, 0, labels + "\n", "")
        if arguments[:2] == ("rm", "--force"):
            removed.add(arguments[-1])
            return subprocess.CompletedProcess(arguments, 0, "", "")
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_docker", fake_docker)

    cleanup = finalizer._cleanup_containers(
        root,
        run_id="phase11-test",
        source_commit=SOURCE_COMMIT,
    )

    assert cleanup["removed_container_ids"] == [matching]
    assert cleanup["skipped_unmatched_container_ids"] == [mismatched]
    assert cleanup["remaining_labeled_container_ids"] == []
    assert cleanup["safe_to_seal"] is True
    assert not cid_directory.exists()
    query = calls[0]
    assert "label=colpali.phase11.run_id=phase11-test" in query
    assert f"label=colpali.phase11.source_commit={SOURCE_COMMIT}" in query
    assert ("rm", "--force", mismatched) not in calls


def test_recovery_finalizer_refuses_unverified_container_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _finalizer_args(tmp_path)
    monkeypatch.setattr(
        finalizer,
        "_cleanup_containers",
        lambda *args, **kwargs: {
            "format": "colpali-triton-phase11-emergency-cleanup-v1",
            "recorded_at": "2026-07-26T00:00:00Z",
            "scope": {},
            "removed_container_ids": [],
            "skipped_unmatched_container_ids": [],
            "remaining_labeled_container_ids": ["a" * 64],
            "safe_to_seal": False,
            "errors": ["labeled containers remain after cleanup"],
        },
    )

    with pytest.raises(
        finalizer.FinalizationError,
        match="cleanup is unverified",
    ):
        finalizer.finalize(args)

    assert not (args.result_directory / "SEALED.sha256").exists()
    cleanup = json.loads(
        (
            args.result_directory / "emergency_container_cleanup.json"
        ).read_text()
    )
    assert cleanup["safe_to_seal"] is False


def test_cleanup_retains_cidfiles_when_final_label_query_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    cid_directory = root / ".docker-cids"
    cid_directory.mkdir(parents=True)
    container_id = "a" * 64
    cid_file = cid_directory / "tuning.cid"
    cid_file.write_text(container_id)
    query_count = 0

    def fake_docker(arguments):
        nonlocal query_count
        arguments = tuple(arguments)
        if arguments[:2] == ("ps", "-aq"):
            query_count += 1
            if query_count == 1:
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    container_id + "\n",
                    "",
                )
            return None
        if arguments[0] == "inspect":
            return subprocess.CompletedProcess(
                arguments,
                0,
                "phase11-test\t" + SOURCE_COMMIT + "\n",
                "",
            )
        if arguments[:2] == ("rm", "--force"):
            return subprocess.CompletedProcess(arguments, 0, "", "")
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_docker", fake_docker)

    cleanup = finalizer._cleanup_containers(
        root,
        run_id="phase11-test",
        source_commit=SOURCE_COMMIT,
    )

    assert cleanup["safe_to_seal"] is False
    assert "bounded final docker label query failed" in cleanup["errors"]
    assert cid_file.is_file()


def test_remote_control_contract_binds_launch_receipt_and_host(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="c" * 64,
        content_sha256="d" * 64,
    )

    checks = workflow._control_contract_checks(spec, archive)

    for marker, value in (
        (".expected_instance_id", EXPECTED_INSTANCE_ID),
        (".expected_instance_type", EXPECTED_INSTANCE_TYPE),
        (".expected_region", EXPECTED_REGION),
        (".expected_ami_id", EXPECTED_AMI_ID),
        (".expected_account_id", EXPECTED_ACCOUNT_ID),
        (".launch_receipt_sha256", LAUNCH_RECEIPT_SHA256),
        (".launch_config_fingerprint", EXPECTED_CONFIG_FINGERPRINT),
    ):
        assert marker in checks
        assert value in checks


def test_remote_wait_reconnects_after_transient_ssh_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    states: list[object] = [
        workflow.SSHTransportError("connection reset"),
        "running",
        "sealed",
    ]

    def fake_state(spec_value, archive_value):
        del spec_value, archive_value
        value = states.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(workflow, "_remote_run_state", fake_state)
    monkeypatch.setattr(workflow.time, "sleep", lambda seconds: None)

    assert workflow._wait_for_remote_run(spec, archive) == "sealed"
    assert states == []


@pytest.mark.parametrize(
    ("returncode", "error_type"),
    (
        (255, workflow.SSHTransportError),
        (4, workflow.WorkflowError),
    ),
)
def test_ssh_distinguishes_transport_from_remote_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    error_type: type[Exception],
) -> None:
    monkeypatch.setattr(
        workflow,
        "_run_local",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            returncode,
            "",
            "ssh failure",
        ),
    )

    with pytest.raises(error_type) as captured:
        workflow._run_ssh(_spec(tmp_path), "false")

    if returncode != 255:
        assert not isinstance(captured.value, workflow.SSHTransportError)


def test_ssh_subprocess_timeout_is_bounded_but_not_retried_as_rc255(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed = []

    def time_out(command, **kwargs):
        observed.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(workflow.subprocess, "run", time_out)

    with pytest.raises(workflow.WorkflowError, match="120-second bound") as exc:
        workflow._run_ssh(_spec(tmp_path), "printf ready")

    assert not isinstance(exc.value, workflow.SSHTransportError)
    assert observed == [workflow.SSH_COMMAND_TIMEOUT_SECONDS]


def test_remote_command_start_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="c" * 64,
        content_sha256="d" * 64,
    )
    monkeypatch.setattr(
        workflow,
        "_remote_run_state",
        lambda *args: "prepared",
    )
    monkeypatch.setattr(
        workflow,
        "_ensure_remote_control",
        lambda *args: None,
    )
    attempts = []

    def fail_start(*args):
        attempts.append(args)
        raise workflow.WorkflowError("sudo policy rejected systemd-run")

    monkeypatch.setattr(workflow, "_start_remote_job", fail_start)

    with pytest.raises(workflow.WorkflowError, match="sudo policy rejected"):
        workflow._wait_for_remote_run(_spec(tmp_path), archive)

    assert len(attempts) == 1


def test_repeated_ambiguous_start_failures_share_one_reconnect_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="c" * 64,
        content_sha256="d" * 64,
    )
    monotonic_values = iter((0.0, 1.0, 2.0, 7.0))
    monkeypatch.setattr(
        workflow,
        "_remote_run_state",
        lambda *args: "prepared",
    )
    monkeypatch.setattr(
        workflow,
        "_ensure_remote_control",
        lambda *args: None,
    )
    monkeypatch.setattr(
        workflow,
        "_start_remote_job",
        lambda *args: (_ for _ in ()).throw(
            workflow.SSHTransportError("connection reset")
        ),
    )
    monkeypatch.setattr(workflow.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(workflow.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(workflow, "REMOTE_RECONNECT_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(workflow, "REMOTE_POLL_TIMEOUT_SECONDS", 100)

    with pytest.raises(workflow.WorkflowError, match="remained unavailable"):
        workflow._wait_for_remote_run(_spec(tmp_path), archive)


@pytest.mark.parametrize("resume_state", ("sealed", "bundled"))
def test_execute_reuses_sealed_or_bundled_run_without_restarting_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume_state: str,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    monkeypatch.setattr(
        workflow,
        "_run_ssh",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, "", ""
        ),
    )
    monkeypatch.setattr(
        workflow, "_prepare_source_archive", lambda *args: archive
    )
    monkeypatch.setattr(
        workflow, "_remote_source_state", lambda *args: "ready"
    )
    monkeypatch.setattr(workflow, "_ensure_remote_control", lambda *args: None)
    monkeypatch.setattr(
        workflow, "_remote_run_state", lambda *args: resume_state
    )
    monkeypatch.setattr(
        workflow,
        "_start_remote_job",
        lambda *args: pytest.fail("sealed run must not restart"),
    )
    monkeypatch.setattr(
        workflow,
        "_wait_for_remote_run",
        lambda *args: pytest.fail("sealed run must not be polled"),
    )
    monkeypatch.setattr(
        workflow,
        "_seal_remote_bundle",
        lambda *args: ("/remote/run.tar.gz", "/remote/run.tar.gz.sha256"),
    )
    receipt = {
        "remote_workflow_status": "complete",
        "remote_failure_reasons": [],
    }
    monkeypatch.setattr(
        workflow, "_retrieve_bundle", lambda *args: receipt
    )

    result = workflow.execute(spec)

    assert result["status"] == "complete"
    assert result["remote_resume_state"] == resume_state
    assert result["reused_verified_local_result"] is False


def test_execute_reuses_already_verified_local_result_without_ssh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec.local_result_directory.mkdir(parents=True)
    receipt = {
        "remote_workflow_status": "complete",
        "remote_failure_reasons": [],
    }
    monkeypatch.setattr(
        workflow,
        "_prepare_source_archive",
        lambda *args: workflow.SourceArchive(
            path=tmp_path / "source.tar",
            archive_sha256="b" * 64,
            content_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        workflow, "_reuse_local_result", lambda *args: receipt
    )
    monkeypatch.setattr(
        workflow,
        "_run_ssh",
        lambda *args, **kwargs: pytest.fail(
            "verified local reuse must not invoke SSH"
        ),
    )

    result = workflow.execute(spec)

    assert result["status"] == "complete"
    assert result["reused_verified_local_result"] is True


def test_retrieval_receipt_retains_source_archive_and_content_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    archive = workflow.SourceArchive(
        path=tmp_path / "source.tar",
        archive_sha256="b" * 64,
        content_sha256="c" * 64,
    )
    bundle_remote = "/remote/phase11-test.tar.gz"
    bundle_sha_remote = f"{bundle_remote}.sha256"
    transfer_timeouts = []

    def fake_run(
        command,
        *,
        capture_output=True,
        timeout_seconds=workflow.LOCAL_COMMAND_TIMEOUT_SECONDS,
    ):
        del capture_output
        transfer_timeouts.append(timeout_seconds)
        destination = Path(command[-1])
        if destination.name.endswith(".sha256"):
            bundle = destination.with_suffix("")
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            destination.write_text(
                f"{digest}  {Path(bundle_remote).name}\n"
            )
        else:
            destination.write_bytes(b"verified bundle")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_extract(bundle, destination, run_id):
        del bundle
        root = destination / run_id
        root.mkdir(parents=True)
        return root

    monkeypatch.setattr(workflow, "_run_local", fake_run)
    monkeypatch.setattr(workflow, "_safe_extract", fake_extract)
    monkeypatch.setattr(
        workflow,
        "_verify_artifact_manifest",
        lambda *args, **kwargs: ("d" * 64, [], "complete", []),
    )

    receipt = workflow._retrieve_bundle(
        spec,
        bundle_remote,
        bundle_sha_remote,
        archive,
    )

    assert receipt["source_archive_sha256"] == "b" * 64
    assert receipt["source_content_sha256"] == "c" * 64
    assert receipt["remote_unit_name"] == spec.remote_unit_name
    assert (
        receipt["remote_job_timeout_seconds"]
        == workflow.REMOTE_JOB_TIMEOUT_SECONDS
    )
    assert transfer_timeouts == [
        workflow.ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS,
        workflow.LOCAL_COMMAND_TIMEOUT_SECONDS,
    ]


def test_manifest_verification_detects_payload_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    _, expected_sha = _write_manifest(root)

    actual_sha, records, status, reasons = (
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )
    )

    assert actual_sha == expected_sha
    assert status == "complete"
    assert reasons == []
    assert {
        "maxsim_cuda_benchmark.json",
        "maxsim_profile.ncu-rep",
        "profile_probe_ncu.json",
        "tuning_winner.json",
        "triton_tuning.json",
        "maxsim_benchmark_tuned_config.json",
        "result_contract_validation.json",
    }.issubset({record["path"] for record in records})
    (root / "maxsim_cuda_benchmark.json").write_text("tampered")
    with pytest.raises(
        workflow.WorkflowError,
        match="cross-check differs|failed verification",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )


def test_manifest_expected_host_must_equal_launch_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    manifest_path, _ = _write_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    manifest["expected_host"]["account_id"] = "999999999999"
    _reseal_manifest(root, manifest)

    with pytest.raises(
        workflow.WorkflowError,
        match="expected host differs from the launch receipt",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )


def test_complete_manifest_locally_recomputes_host_identity_gate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    manifest_path, _ = _write_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    host_path = root / "host_metadata.json"
    host = json.loads(host_path.read_text())
    host["ec2"]["identity_document"]["instanceId"] = "i-deadbeef"
    host_path.write_text(json.dumps(host, sort_keys=True) + "\n")
    _reseal_manifest(root, manifest)

    with pytest.raises(
        workflow.WorkflowError,
        match="stored host validation differs from local recomputation",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )


def test_result_contract_rejects_placeholder_benchmark_results(
    tmp_path: Path,
) -> None:
    _write_complete_result_evidence(tmp_path)
    benchmark_path = tmp_path / "maxsim_cuda_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["results"] = [{} for _ in range(12)]
    benchmark_path.write_text(json.dumps(benchmark, sort_keys=True) + "\n")

    result = workflow.validate_result_contract(
        tmp_path,
        source_commit=SOURCE_COMMIT,
    )

    assert result["status"] == "failed"
    assert any(
        "shape/dtype is invalid" in error for error in result["errors"]
    )


def test_result_contract_recomputes_the_measured_tuning_winner(
    tmp_path: Path,
) -> None:
    _write_complete_result_evidence(tmp_path)
    tuning_path = tmp_path / "triton_tuning.json"
    tuning = json.loads(tuning_path.read_text())
    faster_candidate_id = tuning["protocol"]["config"]["candidates"][1][
        "candidate_id"
    ]
    for workload in tuning["results"]:
        candidate = next(
            item
            for item in workload["candidates"]
            if item["candidate_id"] == faster_candidate_id
        )
        timing = candidate["timing"]
        for trial in timing["trials"]:
            trial["cuda_event_latency_ms"] = [0.5] * len(
                trial["cuda_event_latency_ms"]
            )
        timing["summary"]["latency_ms"]["p50"] = 0.5
        timing["total_cuda_event_latency_ms"] = 0.5 * sum(
            len(trial["cuda_event_latency_ms"])
            for trial in timing["trials"]
        )
    tuning_path.write_text(json.dumps(tuning, sort_keys=True) + "\n")

    result = workflow.validate_result_contract(
        tmp_path,
        source_commit=SOURCE_COMMIT,
    )

    assert result["status"] == "failed"
    assert "claimed tuning winner is not the measured winner" in (
        result["errors"]
    )


@pytest.mark.parametrize("mutation", ("missing_step", "status_disagreement"))
def test_manifest_requires_exact_steps_and_step_status_agreement(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    manifest_path, _ = _write_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    step_status_path = root / "step_status.json"
    step_status = json.loads(step_status_path.read_text())
    if mutation == "missing_step":
        manifest["steps"].pop("benchmark")
        step_status["steps"].pop("benchmark")
    else:
        step_status["steps"]["benchmark"] = {
            "exit_code": 7,
            "succeeded": False,
        }
    step_status_path.write_text(json.dumps(step_status, sort_keys=True) + "\n")
    _reseal_manifest(root, manifest)

    with pytest.raises(
        workflow.WorkflowError,
        match="mandatory-step status is invalid",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )


def test_manifest_rejects_unrelated_cuda_count_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    manifest_path, _ = _write_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    assertion_path = root / "cuda_test_assertion.json"
    assertion = json.loads(assertion_path.read_text())
    assertion["required_counts"] = {"parity": 5}
    assertion["observed_counts"] = {"parity": 5}
    assertion_path.write_text(json.dumps(assertion, sort_keys=True) + "\n")
    _reseal_manifest(root, manifest)

    with pytest.raises(
        workflow.WorkflowError,
        match="without five GPU cases",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
        )


def test_manifest_preserves_explicit_incomplete_profile_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    _write_manifest(root, status="incomplete", with_profile=False)

    _, _, status, reasons = workflow._verify_artifact_manifest(
        root,
        source_commit=SOURCE_COMMIT,
        run_id="phase11-test",
        expected_host=_expected_host(),
    )

    assert status == "incomplete"
    assert reasons == ["profiling failed"]


def test_manifest_cannot_claim_completion_without_real_profile(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase11-test"
    root.mkdir()
    _write_manifest(root, status="complete", with_profile=False)

    with pytest.raises(
        workflow.WorkflowError,
        match="claims completion without a real profile",
    ):
        workflow._verify_artifact_manifest(
            root,
            source_commit=SOURCE_COMMIT,
            run_id="phase11-test",
            expected_host=_expected_host(),
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

    assert "CUDA_TEST_FILES=(" in script
    assert "tests/test_triton_maxsim.py" in script
    assert "tests/test_tune_triton_maxsim_script.py" in script
    assert "assert_phase11_cuda_tests.py" in script
    assert "assert_phase11_l4.py" in script
    assert "benchmarks/tune_triton_maxsim.py --preflight" in script
    assert "--output /phase11-results/triton_tuning.json" in script
    assert (
        "run_docker_capture_timed \\\n"
        '        "derive_winner" \\\n'
        '        "${DERIVE_TIMEOUT_SECONDS}"'
    ) in script
    assert "--config /phase11-results/maxsim_benchmark_tuned_config.json" in (
        script
    )
    assert "benchmarks/benchmark_maxsim.py" in script
    assert "--output /phase11-results/maxsim_cuda_benchmark.json" in script
    assert script.count(
        "timeout --signal=TERM --kill-after=30s 300s"
    ) == 2
    assert "maxsim_profile.ncu-rep" in script
    assert "maxsim_profile.nsys-rep" in script
    assert "--winner-record /phase11-results/tuning_winner.json" in script
    assert "--cap-add" in script
    assert "SYS_ADMIN" in script
    assert "--privileged" not in script
    assert "phase11_result_contract.py" in script
    assert 'status = "complete" if not failure_reasons else "incomplete"' in (
        script
    )
    assert "artifact_manifest.json" in script
    assert "SEALED.sha256" in script
    assert "COMPLETE.sha256" not in script
    assert "terminate-instances" not in script
    assert "l4_lifecycle.py terminate" not in script


def test_profiler_probe_is_one_fixed_colpali_shape() -> None:
    source = PROFILE_PATH.read_text()

    assert "(1, 15, 128)" in source
    assert "(32, 1030, 128)" in source
    assert "torch.inference_mode()" in source
    assert source.count("maxsim_triton(") == 1


def test_cuda_assertion_requires_all_five_cases_without_skips(
    tmp_path: Path,
) -> None:
    cases = """
      <testcase classname="tests.test_triton_maxsim"
        name="test_cuda_kernel_matches_vectorized_reference[float16-0.002-0.002]"/>
      <testcase classname="tests.test_triton_maxsim"
        name="test_cuda_kernel_matches_vectorized_reference[bfloat16-0.02-0.02]"/>
      <testcase classname="tests.test_triton_maxsim"
        name="test_cuda_kernel_matches_vectorized_reference[float32-0.0002-0.0002]"/>
      <testcase classname="tests.test_triton_maxsim"
        name="test_cuda_kernel_handles_noncontiguous_inputs_and_empty_masks"/>
      <testcase classname="tests.test_triton_maxsim"
        name="test_cuda_kernel_propagates_active_nan_across_document_tiles"/>
    """
    junit = tmp_path / "cuda.xml"
    junit.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")

    result = cuda_assertion.inspect_junit(junit)

    assert result["status"] == "passed"
    assert sum(result["observed_counts"].values()) == 5


def test_cuda_assertion_rejects_a_skipped_parity_case(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "cuda.xml"
    junit.write_text(
        """
        <testsuites><testsuite>
          <testcase classname="tests.test_triton_maxsim"
            name="test_cuda_kernel_matches_vectorized_reference[float16]">
            <skipped message="CUDA absent"/>
          </testcase>
        </testsuite></testsuites>
        """
    )

    result = cuda_assertion.inspect_junit(junit)

    assert result["status"] == "failed"
    assert result["skipped"]
    assert any("expected 3" in error for error in result["errors"])


def test_incomplete_remote_result_returns_nonzero_after_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = tmp_path / "benchmark.pem"
    identity.write_text("key")
    identity.chmod(0o600)
    launch_receipt = _write_launch_receipt(tmp_path / "launch.json")
    called = []

    monkeypatch.setattr(
        workflow,
        "_source_state",
        lambda: (SOURCE_COMMIT, []),
    )
    monkeypatch.setattr(workflow, "_local_preflight", lambda spec: [])

    def fake_execute(spec):
        called.append(spec)
        return {
            "status": "incomplete",
            "failure_reasons": ["no profiler report"],
        }

    monkeypatch.setattr(workflow, "execute", fake_execute)
    status = workflow.main(
        (
            "--host",
            "203.0.113.10",
            "--identity-file",
            str(identity),
            "--launch-receipt",
            str(launch_receipt),
            "--run-id",
            "phase11-test",
            "--local-result-base",
            str(tmp_path / "local"),
            "--execute",
            "--acknowledge-run",
            workflow.RUN_ACKNOWLEDGEMENT,
        )
    )

    assert called
    assert status == 3
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_missing_profilers_are_incomplete(tmp_path: Path) -> None:
    winner = _write_profile_winner(tmp_path)
    result = profile_status.evaluate_profile_attempts(
        tmp_path,
        (
            {
                "profiler": "ncu",
                "attempted": False,
                "exit_code": -1,
                "report": "",
                "metadata": "",
            },
            {
                "profiler": "nsys",
                "attempted": False,
                "exit_code": -1,
                "report": "",
                "metadata": "",
            },
        ),
        winner_record=winner,
    )

    assert result["succeeded"] is False
    assert result["profiler"] is None
    assert result["report"] is None


def test_failed_ncu_falls_back_to_successful_nsys(
    tmp_path: Path,
) -> None:
    winner = _write_profile_winner(tmp_path, profiler="nsys")
    (tmp_path / "maxsim_profile.nsys-rep").write_bytes(b"nsys-report")
    result = profile_status.evaluate_profile_attempts(
        tmp_path,
        (
            {
                "profiler": "ncu",
                "attempted": True,
                "exit_code": 1,
                "report": "",
                "metadata": "",
            },
            {
                "profiler": "nsys",
                "attempted": True,
                "exit_code": 0,
                "report": "maxsim_profile.nsys-rep",
                "metadata": "profile_probe_nsys.json",
            },
        ),
        winner_record=winner,
    )

    assert result["succeeded"] is True
    assert result["profiler"] == "nsys"
    assert result["report"] == "maxsim_profile.nsys-rep"
    assert result["attempts"][0]["succeeded"] is False
    assert result["attempts"][1]["succeeded"] is True


def test_zero_exit_without_nonempty_profile_is_not_success(
    tmp_path: Path,
) -> None:
    winner = _write_profile_winner(tmp_path, profiler="ncu")
    (tmp_path / "maxsim_profile.ncu-rep").touch()
    result = profile_status.evaluate_profile_attempts(
        tmp_path,
        (
            {
                "profiler": "ncu",
                "attempted": True,
                "exit_code": 0,
                "report": "maxsim_profile.ncu-rep",
                "metadata": "profile_probe_ncu.json",
            },
            {
                "profiler": "nsys",
                "attempted": False,
                "exit_code": -1,
                "report": "",
                "metadata": "",
            },
        ),
        winner_record=winner,
    )

    assert result["succeeded"] is False
    assert result["attempts"][0]["report_valid"] is False


def test_profile_metadata_must_match_tuning_winner(
    tmp_path: Path,
) -> None:
    winner = _write_profile_winner(tmp_path, profiler="ncu")
    metadata_path = tmp_path / "profile_probe_ncu.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["launch_configuration"]["num_warps"] = 8
    metadata_path.write_text(json.dumps(metadata) + "\n")
    (tmp_path / "maxsim_profile.ncu-rep").write_bytes(b"ncu-report")

    result = profile_status.evaluate_profile_attempts(
        tmp_path,
        (
            {
                "profiler": "ncu",
                "attempted": True,
                "exit_code": 0,
                "report": "maxsim_profile.ncu-rep",
                "metadata": "profile_probe_ncu.json",
            },
            {
                "profiler": "nsys",
                "attempted": False,
                "exit_code": -1,
                "report": "",
                "metadata": "",
            },
        ),
        winner_record=winner,
    )

    assert result["succeeded"] is False
    assert result["attempts"][0]["metadata_matches_winner"] is False


def _synthetic_tuning_winner(
    *,
    block_document_tokens: int = 32,
) -> dict:
    launch = {
        "candidate_id": "tq16_td32_w4_s1",
        "block_query_tokens": 16,
        "block_document_tokens": block_document_tokens,
        "block_embedding_dimension": 128,
        "num_warps": 4,
        "num_stages": 1,
    }
    return {
        "status": "completed",
        "selection": {"canonical_full_matrix": True},
        "provenance": {
            "source_commit": {
                "effective_commit": SOURCE_COMMIT,
                "consistent": True,
                "sources": {
                    "git": None,
                    "environment": SOURCE_COMMIT,
                    "marker_file": SOURCE_COMMIT,
                },
            }
        },
        "winner": {
            "candidate_id": launch["candidate_id"],
            "configuration": launch,
            "overall": {
                "status": "selected",
                "winner_candidate_id": launch["candidate_id"],
            },
        },
    }


def test_winner_derivation_replaces_only_canonical_launch() -> None:
    base = json.loads(
        (PROJECT_ROOT / "configs" / "maxsim_benchmark.json").read_text()
    )
    derived, winner = derive_winner.derive_winner_contract(
        _synthetic_tuning_winner(),
        base,
        source_commit=SOURCE_COMMIT,
    )

    assert len(derived["cases"]) == 6
    assert len(derived["dtypes"]) == 2
    assert derived["providers"] == ["torch", "triton"]
    assert derived["triton_launch"] == {
        "block_query_tokens": 16,
        "block_document_tokens": 32,
        "block_embedding_dimension": 128,
        "num_warps": 4,
        "num_stages": 1,
    }
    unchanged = set(base) - {"benchmark_id", "triton_launch"}
    assert all(derived[key] == base[key] for key in unchanged)
    assert winner["candidate_id"] == "tq16_td32_w4_s1"
    assert winner["canonical_case_count"] == 6


def test_winner_derivation_uses_production_launch_validation() -> None:
    base = json.loads(
        (PROJECT_ROOT / "configs" / "maxsim_benchmark.json").read_text()
    )

    with pytest.raises(ValueError, match="must be one of"):
        derive_winner.derive_winner_contract(
            _synthetic_tuning_winner(block_document_tokens=48),
            base,
            source_commit=SOURCE_COMMIT,
        )


def test_l4_hardware_gate_records_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        total_memory=23 * 1024**3,
        major=8,
        minor=9,
    )
    monkeypatch.setattr(l4_assertion.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(l4_assertion.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        l4_assertion.torch.cuda,
        "get_device_properties",
        lambda index: properties,
    )
    monkeypatch.setattr(
        l4_assertion.torch.cuda,
        "get_device_name",
        lambda index: "NVIDIA L4",
    )

    result = l4_assertion.inspect_cuda_l4()

    assert result["status"] == "passed"
    assert result["device"]["total_memory_bytes"] == 23 * 1024**3
    assert result["device"]["compute_capability"] == [8, 9]


def test_l4_hardware_gate_rejects_wrong_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        total_memory=23 * 1024**3,
        major=8,
        minor=9,
    )
    monkeypatch.setattr(l4_assertion.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(l4_assertion.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        l4_assertion.torch.cuda,
        "get_device_properties",
        lambda index: properties,
    )
    monkeypatch.setattr(
        l4_assertion.torch.cuda,
        "get_device_name",
        lambda index: "NVIDIA A10G",
    )

    result = l4_assertion.inspect_cuda_l4()

    assert result["status"] == "failed"
    assert any("expected 'NVIDIA L4'" in item for item in result["blockers"])


def test_incomplete_contract_replays_identically_under_new_root(
    tmp_path: Path,
) -> None:
    remote_root = tmp_path / "remote-host" / "phase11-incomplete"
    remote_root.mkdir(parents=True)
    stored = workflow.validate_result_contract(
        remote_root,
        source_commit=SOURCE_COMMIT,
    )
    assert stored["status"] == "failed"
    assert stored["errors"] == [
        "tuning: FileNotFoundError errno=2",
        "winner: FileNotFoundError errno=2",
        "derived_config: FileNotFoundError errno=2",
        "benchmark: FileNotFoundError errno=2",
    ]
    (remote_root / "result_contract_validation.json").write_text(
        json.dumps(stored, sort_keys=True) + "\n"
    )
    payload = remote_root / "diagnostic.log"
    payload.write_text("tuning failed before producing an output\n")
    profiling = {
        "succeeded": False,
        "profiler": None,
        "report": None,
        "metadata": None,
        "winner_record": None,
        "winner_record_sha256": None,
        "attempts": [],
    }
    steps = {
        name: {"exit_code": 0, "succeeded": True}
        for name in workflow.MANDATORY_PHASE11_STEPS
    }
    steps["tuning"] = {"exit_code": 1, "succeeded": False}
    (remote_root / "step_status.json").write_text(
        json.dumps(
            {
                "format": "colpali-triton-phase11-step-status-v1",
                "steps": steps,
            },
            sort_keys=True,
        )
        + "\n"
    )
    files = []
    for path in sorted(remote_root.iterdir()):
        files.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "format": "colpali-triton-phase11-artifact-manifest-v2",
        "sealed_at": "2026-07-26T00:00:00Z",
        "status": "incomplete",
        "failure_reasons": ["mandatory step failed: tuning"],
        "profiling": profiling,
        "steps": steps,
        "source_commit": SOURCE_COMMIT,
        "run_id": "phase11-incomplete",
        "expected_host": _expected_host(),
        "files": files,
    }
    manifest_path = remote_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (remote_root / "SEALED.sha256").write_text(
        f"{manifest_sha}  artifact_manifest.json\n"
    )

    local_root = tmp_path / "retrieved-elsewhere" / "phase11-incomplete"
    shutil.copytree(remote_root, local_root)
    recomputed = workflow.validate_result_contract(
        local_root,
        source_commit=SOURCE_COMMIT,
    )

    assert recomputed == stored
    _, _, status, reasons = workflow._verify_artifact_manifest(
        local_root,
        source_commit=SOURCE_COMMIT,
        run_id="phase11-incomplete",
        expected_host=_expected_host(),
    )
    assert status == "incomplete"
    assert reasons == ["mandatory step failed: tuning"]
