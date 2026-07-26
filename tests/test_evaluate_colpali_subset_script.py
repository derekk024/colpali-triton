from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import evaluate_colpali_subset as runner


def test_argument_parser_requires_a_known_task() -> None:
    parsed = runner._parse_args(["--task", "docvqa"])

    assert parsed.task == "docvqa"
    assert parsed.overwrite is False
    assert parsed.local_files_only is False

    with pytest.raises(SystemExit):
        runner._parse_args(["--task", "unknown"])


def test_default_outputs_are_task_specific() -> None:
    assert runner._default_output("docvqa").name == (
        "docvqa_colpali_subset.json"
    )
    assert runner._default_output("infovqa").name == (
        "infovqa_colpali_subset.json"
    )


def test_atomic_output_refuses_replacement(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    runner._write_json_atomic(output, {"result": 1}, overwrite=False)

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_json_atomic(output, {"result": 2}, overwrite=False)

    assert json.loads(output.read_text()) == {"result": 1}
    runner._write_json_atomic(output, {"result": 3}, overwrite=True)
    assert json.loads(output.read_text()) == {"result": 3}


def test_preflight_rejects_existing_file_and_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    output.write_text("{}")
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner._preflight_output(output, overwrite=False)
    runner._preflight_output(output, overwrite=True)
    with pytest.raises(IsADirectoryError, match="directory"):
        runner._preflight_output(directory, overwrite=True)


def test_source_fingerprint_changes_with_content(tmp_path: Path) -> None:
    first = runner.PROJECT_ROOT / "pyproject.toml"
    local = tmp_path / "local.txt"
    local.write_text("first")
    original_root = runner.PROJECT_ROOT
    try:
        runner.PROJECT_ROOT = tmp_path
        first_value = runner._source_fingerprint((local,))
        local.write_text("second")
        second_value = runner._source_fingerprint((local,))
    finally:
        runner.PROJECT_ROOT = original_root

    assert first.exists()
    assert first_value != second_value
