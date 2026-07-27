from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSERTION_PATH = (
    PROJECT_ROOT / "infra" / "aws" / "assert_phase11_tuning_preflight.py"
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "triton_tuning.json"


def _load_assertion_module():
    spec = importlib.util.spec_from_file_location(
        "phase11_tuning_preflight_assertion",
        ASSERTION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assertion = _load_assertion_module()


def _ready_record() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text())
    return {
        "tuning_id": config["tuning_id"],
        "config_file_sha256": hashlib.sha256(
            CONFIG_PATH.read_bytes()
        ).hexdigest(),
        "semantic_config_sha256": "a" * 64,
        "selection": {
            "candidates": [
                item["candidate_id"] for item in config["candidates"]
            ],
            "dtypes": [item["name"] for item in config["dtypes"]],
            "cases": [item["name"] for item in config["cases"]],
        },
        "preflight": {
            "status": "ready",
            "blockers": [],
            "candidate_count": 12,
            "candidate_validation": {
                "valid": True,
                "validated_candidate_count": 12,
                "class": (
                    "colpali_triton.triton_maxsim.TritonMaxSimConfig"
                ),
                "error": None,
            },
            "cuda": {
                "pytorch_build_includes_cuda": True,
                "available": True,
                "device_count": 1,
                "configured_device_index": 0,
                "configured_device_in_range": True,
            },
            "triton": {
                "available": True,
                "entrypoint": "triton",
                "error": None,
            },
        },
    }


def _inspect(tmp_path: Path, record: dict[str, object]) -> dict[str, object]:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(record))
    return assertion.inspect_tuning_preflight(
        preflight,
        config_path=CONFIG_PATH,
    )


def test_ready_full_matrix_preflight_passes(tmp_path: Path) -> None:
    result = _inspect(tmp_path, _ready_record())

    assert result["status"] == "passed"
    assert result["errors"] == []
    assert result["required_counts"] == {
        "candidates": 12,
        "dtypes": 2,
        "cases": 4,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("blocked", "preflight status is not ready"),
        ("candidate_selection", "exact 12 configured candidates"),
        ("dtype_selection", "exact 2 configured dtypes"),
        ("case_selection", "exact 4 configured cases"),
        ("candidate_invalid", "candidate validation did not pass"),
        (
            "candidate_count",
            "candidate validation must cover exactly 12 candidates",
        ),
    ),
)
def test_incomplete_or_blocked_preflight_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    record = deepcopy(_ready_record())
    selection = record["selection"]
    preflight = record["preflight"]
    assert isinstance(selection, dict)
    assert isinstance(preflight, dict)
    validation = preflight["candidate_validation"]
    assert isinstance(validation, dict)

    if mutation == "blocked":
        preflight["status"] = "blocked"
        preflight["blockers"] = ["CUDA unavailable"]
    elif mutation == "candidate_selection":
        selection["candidates"] = selection["candidates"][:-1]
    elif mutation == "dtype_selection":
        selection["dtypes"] = selection["dtypes"][:-1]
    elif mutation == "case_selection":
        selection["cases"] = selection["cases"][:-1]
    elif mutation == "candidate_invalid":
        validation["valid"] = False
        validation["error"] = "invalid launch"
    elif mutation == "candidate_count":
        validation["validated_candidate_count"] = 11
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    result = _inspect(tmp_path, record)

    assert result["status"] == "failed"
    assert any(expected_error in error for error in result["errors"])


def test_malformed_preflight_is_a_failed_assertion(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{")

    result = assertion.inspect_tuning_preflight(
        preflight,
        config_path=CONFIG_PATH,
    )

    assert result["status"] == "failed"
    assert any(
        error.startswith("cannot parse tuning preflight:")
        for error in result["errors"]
    )


def test_cli_returns_nonzero_for_blocked_preflight(tmp_path: Path) -> None:
    record = _ready_record()
    preflight_record = record["preflight"]
    assert isinstance(preflight_record, dict)
    preflight_record["status"] = "blocked"
    preflight_record["blockers"] = ["CUDA unavailable"]
    preflight = tmp_path / "preflight.json"
    output = tmp_path / "assertion.json"
    preflight.write_text(json.dumps(record))

    result = subprocess.run(
        [
            sys.executable,
            str(ASSERTION_PATH),
            "--preflight",
            str(preflight),
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"
