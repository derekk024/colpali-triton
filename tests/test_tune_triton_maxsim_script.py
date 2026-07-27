from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

from benchmarks import tune_triton_maxsim as runner
from colpali_triton.triton_maxsim import TritonMaxSimConfig


CONFIG = runner.load_tuning_config(runner.DEFAULT_CONFIG)


def test_default_config_is_explicit_and_runtime_valid() -> None:
    assert CONFIG.tuning_id == "phase11-l4-triton-maxsim-v1"
    assert len(CONFIG.candidates) == 12
    assert len(CONFIG.dtypes) == 2
    assert len(CONFIG.cases) == 4
    assert CONFIG.warmup_iterations == 10
    assert CONFIG.measured_iterations == 25
    assert CONFIG.trials == 5
    assert len(
        {candidate.candidate_id for candidate in CONFIG.candidates}
    ) == len(CONFIG.candidates)
    assert len(
        {
            tuple(candidate.launch.to_dict().values())
            for candidate in CONFIG.candidates
        }
    ) == len(CONFIG.candidates)

    for candidate in CONFIG.candidates:
        actual = TritonMaxSimConfig(**candidate.launch.to_dict())
        assert actual.block_embedding_dimension == 128


def test_argument_parser_defaults_and_repeatable_filters() -> None:
    parsed = runner._parse_args([])

    assert parsed.config == runner.DEFAULT_CONFIG
    assert parsed.output == runner.DEFAULT_OUTPUT
    assert parsed.candidate is None
    assert parsed.dtype is None
    assert parsed.case is None
    assert parsed.preflight is False
    assert parsed.overwrite is False

    selected = runner._parse_args(
        [
            "--candidate",
            "tq16_td64_w4_s2",
            "--dtype",
            "float16",
            "--case",
            "retrieval_q15_d32",
            "--preflight",
        ]
    )
    assert selected.candidate == ["tq16_td64_w4_s2"]
    assert selected.dtype == ["float16"]
    assert selected.case == ["retrieval_q15_d32"]
    assert selected.preflight is True


def test_name_selection_rejects_duplicates_and_missing_values() -> None:
    configured = ("first", "second")
    assert runner._select_names(
        None, configured, kind="candidate"
    ) == configured
    assert runner._select_names(
        ["second"], configured, kind="candidate"
    ) == ("second",)

    with pytest.raises(ValueError, match="duplicates"):
        runner._select_names(
            ["first", "first"], configured, kind="candidate"
        )
    with pytest.raises(ValueError, match="not present"):
        runner._select_names(
            ["missing"], configured, kind="candidate"
        )


def test_config_rejects_unknown_fields_and_duplicate_candidates() -> None:
    raw = CONFIG.to_dict()
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        runner.TuningConfig.from_mapping(raw)

    duplicate = CONFIG.to_dict()
    duplicate["candidates"][1] = deepcopy(
        duplicate["candidates"][0]
    )
    duplicate["candidates"][1]["candidate_id"] = "other-id"
    with pytest.raises(ValueError, match="duplicate launch"):
        runner.TuningConfig.from_mapping(duplicate)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.load_tuning_config(path)


def test_semantic_fingerprint_ignores_json_formatting(
    tmp_path: Path,
) -> None:
    formatted = tmp_path / "formatted.json"
    formatted.write_text(
        json.dumps(CONFIG.to_dict(), indent=7), encoding="utf-8"
    )
    reloaded = runner.load_tuning_config(formatted)

    assert runner.semantic_config_fingerprint(reloaded) == (
        runner.semantic_config_fingerprint(CONFIG)
    )


def test_runtime_candidate_validation_uses_production_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: List[Dict[str, int]] = []

    class FakeConfig:
        def __init__(self, **kwargs: int) -> None:
            received.append(kwargs)

    module = SimpleNamespace(TritonMaxSimConfig=FakeConfig)
    monkeypatch.setattr(runner, "import_module", lambda name: module)

    record = runner._validate_runtime_candidate_configs(
        CONFIG.candidates[:2]
    )

    assert record["valid"] is True
    assert record["validated_candidate_count"] == 2
    assert received == [
        item.launch.to_dict() for item in CONFIG.candidates[:2]
    ]


def test_non_cuda_preflight_reports_blockers_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda, "is_built", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        runner,
        "_probe_triton",
        lambda: {
            "available": False,
            "entrypoint": "triton",
            "error": "missing",
        },
    )

    record = runner._preflight(
        CONFIG, candidates=CONFIG.candidates
    )

    assert record["status"] == "blocked"
    assert len(record["blockers"]) == 2
    assert record["candidate_validation"]["valid"] is True
    assert record["candidate_count"] == 12


def test_preflight_cli_is_safe_and_does_not_create_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "must-not-exist" / "result.json"

    return_code = runner.main(
        ["--preflight", "--output", str(output)]
    )
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert return_code == 0
    assert record["tuning_id"] == CONFIG.tuning_id
    assert record["preflight"]["status"] in ("ready", "blocked")
    assert not output.exists()
    assert not output.parent.exists()


def test_numerical_agreement_record_exposes_rejection() -> None:
    dtype_spec = CONFIG.dtypes[0]
    expected = torch.tensor([[1.0, 2.0]])

    passed = runner._agreement_record(
        expected.clone(), expected, dtype_spec=dtype_spec
    )
    rejected = runner._agreement_record(
        torch.tensor([[10.0, 20.0]]),
        expected,
        dtype_spec=dtype_spec,
    )
    wrong_shape = runner._agreement_record(
        torch.ones(1), expected, dtype_spec=dtype_spec
    )

    assert passed["agreement"] is True
    assert passed["maximum_absolute_error"] == 0.0
    assert rejected["agreement"] is False
    assert rejected["maximum_absolute_error"] == 18.0
    assert wrong_shape["agreement"] is False
    assert wrong_shape["shape_matches"] is False


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_numerical_agreement_uses_null_for_nonfinite_error_metrics(
    nonfinite: float,
) -> None:
    record = runner._agreement_record(
        torch.tensor([[nonfinite]]),
        torch.tensor([[1.0]]),
        dtype_spec=CONFIG.dtypes[0],
    )

    assert record["finite"] is False
    assert record["agreement"] is False
    assert record["maximum_absolute_error"] is None
    assert record["maximum_relative_error"] is None
    assert "NaN" not in json.dumps(record, allow_nan=False)
    assert "Infinity" not in json.dumps(record, allow_nan=False)


def test_rejected_candidate_never_reaches_warmup_or_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CONFIG.candidates[0]
    case = CONFIG.cases[0]
    dtype = CONFIG.dtypes[0]
    minimal = runner.TuningConfig(
        tuning_id="test",
        seed=7,
        device_index=0,
        warmup_iterations=1,
        measured_iterations=2,
        trials=2,
        dtypes=(dtype,),
        cases=(case,),
        candidates=(candidate,),
        normalize_embeddings=False,
    )
    queries = torch.ones((1, 2, 2), dtype=torch.float16)
    documents = torch.ones((1, 3, 2), dtype=torch.float16)
    query_mask = torch.ones((1, 2), dtype=torch.bool)
    document_mask = torch.ones((1, 3), dtype=torch.bool)
    environment_snapshots: List[Dict[str, int]] = []

    def incorrect_provider(*args: object, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[99.0]])

    def make_inputs(*args: object, **kwargs: object) -> object:
        assert len(environment_snapshots) == 1
        return queries, documents, query_mask, document_mask

    def capture_environment(*args: object, **kwargs: object) -> object:
        snapshot = {"capture_index": len(environment_snapshots)}
        environment_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda *args, **kwargs: {
            "status": "ready",
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        runner, "_load_provider", lambda item: incorrect_provider
    )
    monkeypatch.setattr(
        runner,
        "_make_inputs",
        make_inputs,
    )
    monkeypatch.setattr(
        runner.torch.cuda, "set_device", lambda device: None
    )
    monkeypatch.setattr(
        runner.torch.cuda, "manual_seed_all", lambda seed: None
    )
    monkeypatch.setattr(
        runner.torch.cuda, "synchronize", lambda device=None: None
    )
    monkeypatch.setattr(
        runner.torch.cuda, "empty_cache", lambda: None
    )
    monkeypatch.setattr(
        runner,
        "_warm_candidate",
        lambda *args, **kwargs: pytest.fail(
            "numerically rejected candidate was warmed"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_time_trial",
        lambda *args, **kwargs: pytest.fail(
            "numerically rejected candidate was timed"
        ),
    )
    monkeypatch.setattr(
        runner, "_environment", capture_environment
    )
    monkeypatch.setattr(
        runner,
        "_source_commit_provenance",
        lambda git_state: {
            "effective_commit": git_state["commit"],
            "consistent": True,
        },
    )

    result = runner.run_tuning(
        minimal,
        candidate_ids=(candidate.candidate_id,),
        dtype_names=(dtype.name,),
        case_names=(case.name,),
        config_path=runner.DEFAULT_CONFIG,
        argv=("test",),
    )

    candidate_result = result["results"][0]["candidates"][0]
    assert result["status"] == "completed_without_eligible_candidate"
    assert candidate_result["status"] == "numerical_rejected"
    assert candidate_result["timing"] is None
    assert environment_snapshots == [
        {"capture_index": 0},
        {"capture_index": 1},
    ]
    assert result["environment"] == {
        "start": {"capture_index": 0},
        "end": {"capture_index": 1},
    }


def test_trial_order_is_deterministic_and_trial_specific() -> None:
    candidate_ids = tuple(
        candidate.candidate_id for candidate in CONFIG.candidates
    )
    first = runner._trial_order(
        candidate_ids,
        seed=17,
        workload_key="float16-case",
        trial_index=0,
    )
    second = runner._trial_order(
        candidate_ids,
        seed=17,
        workload_key="float16-case",
        trial_index=1,
    )

    assert first == runner._trial_order(
        candidate_ids,
        seed=17,
        workload_key="float16-case",
        trial_index=0,
    )
    assert set(first) == set(candidate_ids)
    assert set(second) == set(candidate_ids)
    assert first != second


def test_cuda_event_trial_retains_every_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronizations = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.recorded = False

        def record(self) -> None:
            self.recorded = True

        def elapsed_time(self, other: "FakeEvent") -> float:
            assert self.recorded and other.recorded
            return 1.25

    monkeypatch.setattr(runner.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        runner.torch.cuda,
        "synchronize",
        lambda device=None: synchronizations.append(device),
    )
    monkeypatch.setattr(
        runner.torch.cuda, "memory_allocated", lambda device: 100
    )
    monkeypatch.setattr(
        runner.torch.cuda, "memory_reserved", lambda device: 200
    )
    monkeypatch.setattr(
        runner.torch.cuda, "reset_peak_memory_stats", lambda device: None
    )
    monkeypatch.setattr(
        runner.torch.cuda, "max_memory_allocated", lambda device: 150
    )
    monkeypatch.setattr(
        runner.torch.cuda, "max_memory_reserved", lambda device: 275
    )

    trial = runner._time_trial(
        lambda *args, **kwargs: torch.tensor([[1.0]]),
        queries=torch.ones((1, 2, 2)),
        documents=torch.ones((1, 3, 2)),
        query_mask=torch.ones((1, 2), dtype=torch.bool),
        document_mask=torch.ones((1, 3), dtype=torch.bool),
        measured_iterations=3,
        device=torch.device("cpu"),
        trial_index=4,
    )

    assert trial["status"] == "completed"
    assert trial["cuda_event_latency_ms"] == [1.25, 1.25, 1.25]
    assert len(synchronizations) == 1
    assert trial["memory"]["incremental_peak_allocated_bytes"] == 50
    assert trial["memory"]["incremental_peak_reserved_bytes"] == 75
    assert trial["cleanup_synchronization_error"] is None


def test_failure_cleanup_does_not_replace_primary_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cpu")
    queries = torch.ones((1, 2, 2))
    documents = torch.ones((1, 3, 2))
    query_mask = torch.ones((1, 2), dtype=torch.bool)
    document_mask = torch.ones((1, 3), dtype=torch.bool)

    def primary_failure(*args: object, **kwargs: object) -> torch.Tensor:
        raise ValueError("primary workload failure")

    synchronization_count = 0

    def numerical_synchronize(device: object = None) -> None:
        nonlocal synchronization_count
        synchronization_count += 1
        if synchronization_count == 2:
            raise RuntimeError("cleanup synchronization failure")

    monkeypatch.setattr(
        runner.torch.cuda, "synchronize", numerical_synchronize
    )
    numerical = runner._numerical_gate(
        primary_failure,
        queries=queries,
        documents=documents,
        query_mask=query_mask,
        document_mask=document_mask,
        reference=torch.ones((1, 1)),
        dtype_spec=CONFIG.dtypes[0],
        device=device,
    )
    assert numerical["error"] == "ValueError: primary workload failure"
    assert numerical["cleanup_synchronization_error"] == (
        "RuntimeError: cleanup synchronization failure"
    )

    def cleanup_failure(device: object = None) -> None:
        raise RuntimeError("cleanup synchronization failure")

    monkeypatch.setattr(
        runner.torch.cuda, "synchronize", cleanup_failure
    )
    warmup = runner._warm_candidate(
        primary_failure,
        queries=queries,
        documents=documents,
        query_mask=query_mask,
        document_mask=document_mask,
        iterations=1,
        device=device,
    )
    assert warmup["error"] == "ValueError: primary workload failure"
    assert warmup["cleanup_synchronization_error"] == (
        "RuntimeError: cleanup synchronization failure"
    )

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

    monkeypatch.setattr(runner.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        runner.torch.cuda, "memory_allocated", lambda device: 0
    )
    monkeypatch.setattr(
        runner.torch.cuda, "memory_reserved", lambda device: 0
    )
    monkeypatch.setattr(
        runner.torch.cuda, "reset_peak_memory_stats", lambda device: None
    )
    trial = runner._time_trial(
        primary_failure,
        queries=queries,
        documents=documents,
        query_mask=query_mask,
        document_mask=document_mask,
        measured_iterations=1,
        device=device,
        trial_index=0,
    )
    assert trial["error"] == "ValueError: primary workload failure"
    assert trial["cleanup_synchronization_error"] == (
        "RuntimeError: cleanup synchronization failure"
    )


def test_environment_snapshot_uses_capture_neutral_memory_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="NVIDIA L4",
        total_memory=24,
        major=8,
        minor=9,
    )
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda device: properties,
    )
    monkeypatch.setattr(
        runner.torch.cuda, "mem_get_info", lambda device: (12, 24)
    )
    monkeypatch.setattr(
        runner.torch.cuda, "get_arch_list", lambda: ["sm_89"]
    )
    monkeypatch.setattr(
        runner, "_nvidia_smi_snapshot", lambda: {"available": True}
    )

    snapshot = runner._environment(torch.device("cuda", 0), CONFIG)
    device = snapshot["cuda_device"]

    assert snapshot["captured_at_utc"].endswith("Z")
    assert device["free_memory_bytes"] == 12
    assert device["memory_total_bytes"] == 24
    assert "free_memory_bytes_at_start" not in device
    assert "memory_total_bytes_at_start" not in device


def _completed_candidate(
    candidate_id: str, p50: float, total: float
) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "status": "completed",
        "timing": {
            "total_cuda_event_latency_ms": total,
            "summary": {"latency_ms": {"p50": p50}},
        },
    }


def test_ranking_requires_all_workloads_and_uses_median_relative_p50() -> None:
    results = [
        {
            "case": {"name": "small"},
            "dtype": "float16",
            "candidates": [
                _completed_candidate("balanced", 2.0, 20.0),
                _completed_candidate("spiky", 1.0, 10.0),
                _completed_candidate("incomplete", 0.5, 5.0),
            ],
        },
        {
            "case": {"name": "large"},
            "dtype": "float16",
            "candidates": [
                _completed_candidate("balanced", 2.0, 20.0),
                _completed_candidate("spiky", 10.0, 100.0),
                {
                    "candidate_id": "incomplete",
                    "status": "numerical_rejected",
                    "timing": None,
                },
            ],
        },
    ]

    selection = runner._rank_candidates(
        results,
        candidate_ids=("balanced", "spiky", "incomplete"),
    )

    assert selection["status"] == "selected"
    assert selection["winner_candidate_id"] == "balanced"
    assert selection["workload_winners"][0][
        "winner_candidate_id"
    ] == "incomplete"
    incomplete = next(
        item
        for item in selection["ranking"]
        if item["candidate_id"] == "incomplete"
    )
    assert incomplete["eligible"] is False
    assert incomplete["completed_workloads"] == 1


def test_atomic_output_refuses_implicit_replacement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    runner._write_json_atomic(
        output, {"result": 1}, overwrite=False
    )

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_json_atomic(
            output, {"result": 2}, overwrite=False
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "result": 1
    }


def test_source_commit_provenance_requires_agreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    (tmp_path / ".source_commit").write_text(
        commit + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", commit)

    record = runner._source_commit_provenance(
        {"commit": commit, "available": True}
    )
    assert record["effective_commit"] == commit
    assert set(record["sources"].values()) == {commit}

    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", "b" * 40)
    with pytest.raises(RuntimeError, match="provenance disagrees"):
        runner._source_commit_provenance(
            {"commit": commit, "available": True}
        )


def test_source_commit_provenance_requires_one_exact_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("COLPALI_SOURCE_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="exact source commit"):
        runner._source_commit_provenance(
            {"commit": None, "available": False}
        )

    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", "A" * 40)
    with pytest.raises(RuntimeError, match="full lowercase Git SHA"):
        runner._source_commit_provenance(
            {"commit": None, "available": False}
        )
