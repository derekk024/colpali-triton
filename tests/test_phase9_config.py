from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from colpali_triton.phase9_config import (
    PHASE9_CONFIG,
    Phase9EvaluationSpec,
    Phase9SelectionSpec,
    load_phase9_config,
    phase9_config_from_dict,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "phase9_colpali_subset.json"
)


def test_committed_config_matches_the_builtin_contract() -> None:
    loaded = load_phase9_config(CONFIG_PATH)

    assert loaded == PHASE9_CONFIG
    assert loaded.to_dict() == PHASE9_CONFIG.to_dict()
    assert len(loaded.fingerprint) == 64


def test_selection_is_fixed_before_evaluation() -> None:
    selection = PHASE9_CONFIG.selection

    assert selection.queries_per_task == 50
    assert selection.documents_per_task == 100
    assert selection.method == "sha256_identifier_order"
    assert selection.positive_policy == (
        "include_all_selected_query_positives"
    )


def test_evaluation_contains_exact_required_scorers() -> None:
    evaluation = PHASE9_CONFIG.evaluation

    assert evaluation.k == 5
    assert evaluation.device == "mps"
    assert evaluation.dtype == "bfloat16"
    assert evaluation.scorers == (
        "late_interaction_maxsim",
        "l2_normalized_masked_mean",
    )


@pytest.mark.parametrize(
    "field",
    (
        "source_spec_fingerprint",
        "selection_fingerprint",
        "image_content_fingerprint",
    ),
)
def test_task_fingerprints_are_full_digests(field: str) -> None:
    for task in PHASE9_CONFIG.tasks.values():
        assert len(getattr(task, field)) == 64


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("schema_version",), 2, "schema_version"),
        (("format",), "other", "format"),
        (("seed",), -1, "seed"),
        (("selection", "queries_per_task"), 0, "positive integer"),
        (
            ("selection", "method"),
            "first_rows",
            "selection method",
        ),
        (
            ("evaluation", "document_batch_size"),
            2,
            "batch size must be 1",
        ),
        (("evaluation", "device"), "cpu", "device must be mps"),
        (
            ("tasks", "docvqa", "selection_fingerprint"),
            "x" * 64,
            "SHA-256",
        ),
    ),
)
def test_invalid_contract_values_are_rejected(
    path: tuple, value: object, message: str
) -> None:
    raw = deepcopy(PHASE9_CONFIG.to_dict())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        phase9_config_from_dict(raw)


def test_unknown_and_missing_keys_are_rejected() -> None:
    extra = deepcopy(PHASE9_CONFIG.to_dict())
    extra["unexpected"] = True
    missing = deepcopy(PHASE9_CONFIG.to_dict())
    del missing["evaluation"]["dtype"]

    with pytest.raises(ValueError, match="extra"):
        phase9_config_from_dict(extra)
    with pytest.raises(ValueError, match="missing"):
        phase9_config_from_dict(missing)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version": 1, "schema_version": 1}')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_phase9_config(path)


def test_round_trip_is_canonical() -> None:
    raw = json.loads(PHASE9_CONFIG.canonical_json)
    parsed = phase9_config_from_dict(raw)

    assert parsed == PHASE9_CONFIG
    assert parsed.canonical_json == PHASE9_CONFIG.canonical_json
    assert parsed.fingerprint == PHASE9_CONFIG.fingerprint


def test_selection_rejects_more_queries_than_documents() -> None:
    with pytest.raises(ValueError, match="may not exceed"):
        Phase9SelectionSpec(
            method="sha256_identifier_order",
            salt="salt",
            queries_per_task=3,
            documents_per_task=2,
            positive_policy="include_all_selected_query_positives",
            distractor_policy="sha256_identifier_order",
        )


def test_evaluation_rejects_scorer_reordering() -> None:
    with pytest.raises(ValueError, match="exact ordered pair"):
        Phase9EvaluationSpec(
            k=5,
            warmup_queries=1,
            document_batch_size=1,
            score_document_batch_size=16,
            dtype="bfloat16",
            device="mps",
            attention_implementation="eager",
            scorers=(
                "l2_normalized_masked_mean",
                "late_interaction_maxsim",
            ),
        )
