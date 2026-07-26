from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from colpali_triton.benchmarking import (
    BenchmarkConfig,
    canonical_config_fingerprint,
    load_benchmark_config,
    percentile,
    source_manifest,
    summarize_latencies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "maxsim_benchmark.json"


def _raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_config_pins_real_colpali_matrix() -> None:
    config = load_benchmark_config(CONFIG_PATH)

    assert config.benchmark_id == "phase10-l4-maxsim-v1"
    assert config.providers == ("torch", "triton")
    assert tuple(item.name for item in config.dtypes) == (
        "float16",
        "bfloat16",
    )
    assert config.warmup_iterations == 10
    assert config.measured_iterations == 50
    assert config.trials == 5
    assert len(config.cases) == 6
    assert {item.embedding_dimension for item in config.cases} == {128}
    assert {item.document_tokens for item in config.cases} == {1030}
    assert max(item.document_batch_size for item in config.cases) == 128
    assert config.triton_launch.to_dict() == {
        "block_query_tokens": 16,
        "block_document_tokens": 64,
        "block_embedding_dimension": 128,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_config_semantic_serialization_round_trips() -> None:
    config = load_benchmark_config(CONFIG_PATH)

    round_tripped = BenchmarkConfig.from_mapping(config.to_dict())

    assert round_tripped == config
    for case in config.to_dict()["cases"]:
        assert "query_document_pairs" not in case


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"unexpected": True}),
            "unexpected",
        ),
        (
            lambda value: value.update({"providers": ["triton", "torch"]}),
            "torch.*first",
        ),
        (
            lambda value: value.update({"providers": ["torch", "torch"]}),
            "duplicates",
        ),
        (
            lambda value: value["cases"][0].update(
                {"active_query_tokens": 16}
            ),
            "cannot exceed",
        ),
        (
            lambda value: value["triton_launch"].update(
                {"block_embedding_dimension": 64}
            ),
            "at least the largest",
        ),
        (
            lambda value: value["triton_launch"].update({"num_warps": 3}),
            "num_warps",
        ),
    ],
)
def test_config_rejects_invalid_contract(mutation, message: str) -> None:
    value = deepcopy(_raw_config())
    mutation(value)

    with pytest.raises((TypeError, ValueError), match=message):
        BenchmarkConfig.from_mapping(value)


def test_config_rejects_duplicate_case_and_dtype_names() -> None:
    duplicate_case = deepcopy(_raw_config())
    duplicate_case["cases"][1]["name"] = duplicate_case["cases"][0]["name"]
    with pytest.raises(ValueError, match="duplicate names"):
        BenchmarkConfig.from_mapping(duplicate_case)

    duplicate_dtype = deepcopy(_raw_config())
    duplicate_dtype["dtypes"][1]["name"] = duplicate_dtype["dtypes"][0][
        "name"
    ]
    with pytest.raises(ValueError, match="duplicate names"):
        BenchmarkConfig.from_mapping(duplicate_dtype)


def test_loader_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_benchmark_config(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_benchmark_config(nonfinite)


def test_semantic_fingerprint_ignores_json_formatting(tmp_path: Path) -> None:
    config = load_benchmark_config(CONFIG_PATH)
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(
        json.dumps(_raw_config(), separators=(",", ":")),
        encoding="utf-8",
    )

    assert canonical_config_fingerprint(
        load_benchmark_config(reformatted)
    ) == canonical_config_fingerprint(config)


def test_source_manifest_tracks_content_and_missing_files(
    tmp_path: Path,
) -> None:
    present = tmp_path / "present.py"
    missing = tmp_path / "future.py"
    present.write_text("first", encoding="utf-8")

    first_records, first_fingerprint = source_manifest(
        (present, missing), root=tmp_path
    )
    present.write_text("second", encoding="utf-8")
    second_records, second_fingerprint = source_manifest(
        (present, missing), root=tmp_path
    )

    assert first_records["future.py"] is None
    assert first_records["present.py"] != second_records["present.py"]
    assert first_fingerprint != second_fingerprint


def test_percentile_uses_linear_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile(values, 1.0) == 4.0


@pytest.mark.parametrize(
    ("values", "quantile", "message"),
    [
        ([], 0.5, "must not be empty"),
        ([1.0], -0.1, "between zero and one"),
        ([1.0], 1.1, "between zero and one"),
        ([float("inf")], 0.5, "finite"),
    ],
)
def test_percentile_rejects_invalid_input(
    values, quantile: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        percentile(values, quantile)


def test_latency_summary_uses_aggregate_work_over_total_time() -> None:
    summary = summarize_latencies(
        [1.0, 9.0],
        documents_per_call=10,
        query_document_pairs_per_call=20,
    )

    assert summary["sample_count"] == 2
    assert summary["latency_ms"]["p50"] == 5.0
    documents = summary["documents_scored_per_query_per_second"]
    pairs = summary["query_document_pairs_per_second"]
    assert documents["aggregate"] == pytest.approx(2_000.0)
    assert pairs["aggregate"] == pytest.approx(4_000.0)
    assert documents["mean_instantaneous"] > documents["aggregate"]


@pytest.mark.parametrize(
    "latencies",
    [[], [0.0], [-1.0], [float("nan")]],
)
def test_latency_summary_rejects_invalid_samples(latencies) -> None:
    with pytest.raises(ValueError):
        summarize_latencies(
            latencies,
            documents_per_call=1,
            query_document_pairs_per_call=1,
        )
