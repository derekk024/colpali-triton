import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_text_baseline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_text_baseline_script", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_load_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        RUNNER._load_json(path)


def test_output_preflight_happens_before_configuration_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output already exists"):
        RUNNER.main(
            (
                "--task",
                "docvqa",
                "--baseline",
                "bm25",
                "--config",
                str(tmp_path / "missing.json"),
                "--output",
                str(output),
            )
        )


def test_atomic_writer_never_replaces_without_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    RUNNER._write_json_atomic(output, {"run": 1}, overwrite=False)

    with pytest.raises(FileExistsError, match="output already exists"):
        RUNNER._write_json_atomic(output, {"run": 2}, overwrite=False)

    assert json.loads(output.read_text(encoding="utf-8")) == {"run": 1}
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_writer_replaces_when_explicitly_requested(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    RUNNER._write_json_atomic(output, {"run": 1}, overwrite=False)
    RUNNER._write_json_atomic(output, {"run": 2}, overwrite=True)

    assert json.loads(output.read_text(encoding="utf-8")) == {"run": 2}


def test_token_length_summary_reports_strict_limit_exceedance() -> None:
    summary = RUNNER._token_length_summary(
        np.asarray([2, 4, 5, 8], dtype=np.int64),
        truncation_limit=4,
    )

    assert summary["count"] == 4
    assert summary["minimum"] == 2
    assert summary["maximum"] == 8
    assert summary["count_exceeding_limit"] == 2
    assert summary["fraction_exceeding_limit"] == 0.5


@pytest.mark.parametrize(
    "lengths",
    (
        np.asarray([], dtype=np.int64),
        np.asarray([1.5], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
    ),
)
def test_token_length_summary_rejects_invalid_vectors(
    lengths: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="token lengths"):
        RUNNER._token_length_summary(lengths, truncation_limit=4)
