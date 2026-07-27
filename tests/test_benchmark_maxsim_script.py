from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from colpali_triton.benchmarking import load_benchmark_config
from benchmarks import benchmark_maxsim as runner


CONFIG = load_benchmark_config(runner.DEFAULT_CONFIG)


def test_argument_parser_defaults_to_canonical_matrix() -> None:
    parsed = runner._parse_args([])

    assert parsed.config == runner.DEFAULT_CONFIG
    assert parsed.output == runner.DEFAULT_OUTPUT
    assert parsed.provider is None
    assert parsed.dtype is None
    assert parsed.case is None
    assert parsed.preflight is False
    assert parsed.overwrite is False


def test_argument_parser_accepts_repeatable_selection() -> None:
    parsed = runner._parse_args(
        [
            "--provider",
            "torch",
            "--dtype",
            "float16",
            "--case",
            "colpali_q15_d1",
            "--preflight",
        ]
    )

    assert parsed.provider == ["torch"]
    assert parsed.dtype == ["float16"]
    assert parsed.case == ["colpali_q15_d1"]
    assert parsed.preflight is True


def test_selection_must_be_configured_and_unique() -> None:
    assert runner._select_names(
        None, ("torch", "triton"), kind="provider"
    ) == ("torch", "triton")
    assert runner._select_names(
        ["triton"], ("torch", "triton"), kind="provider"
    ) == ("triton",)

    with pytest.raises(ValueError, match="duplicates"):
        runner._select_names(
            ["torch", "torch"], ("torch", "triton"), kind="provider"
        )
    with pytest.raises(ValueError, match="not present"):
        runner._select_names(
            ["other"], ("torch", "triton"), kind="provider"
        )


def test_triton_probe_checks_runtime_not_only_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(
        TritonMaxSimConfig=lambda **kwargs: kwargs,
        maxsim_triton=lambda *args, **kwargs: None,
        triton_is_available=lambda: False,
    )
    monkeypatch.setattr(runner, "import_module", lambda name: module)

    record = runner._probe_provider("triton")

    assert record["available"] is False
    assert "could not be imported" in record["error"]


def test_triton_provider_receives_exact_launch_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = {}

    class FakeTritonConfig:
        def __init__(self, **kwargs: int) -> None:
            self.values = kwargs

    def fake_maxsim(
        queries: torch.Tensor,
        documents: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        document_mask: torch.Tensor,
        config: FakeTritonConfig,
    ) -> torch.Tensor:
        del queries, documents, query_mask, document_mask
        received.update(config.values)
        return torch.tensor([[1.0]])

    module = SimpleNamespace(
        TritonMaxSimConfig=FakeTritonConfig,
        maxsim_triton=fake_maxsim,
    )
    monkeypatch.setattr(runner, "import_module", lambda name: module)
    provider = runner._load_provider("triton", CONFIG)

    output = provider(
        torch.zeros((1, 1, 2)),
        torch.zeros((1, 1, 2)),
        query_mask=torch.ones((1, 1), dtype=torch.bool),
        document_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    assert output.item() == 1.0
    assert received == CONFIG.triton_launch.to_dict()


def test_preflight_records_launch_config_and_all_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_probe_provider",
        lambda name: {
            "available": name == "torch",
            "entrypoint": name,
            "error": None if name == "torch" else "missing",
        },
    )
    monkeypatch.setattr(
        torch.backends.cuda, "is_built", lambda: False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    record = runner._preflight(
        CONFIG, providers=("torch", "triton")
    )

    assert record["status"] == "blocked"
    assert len(record["blockers"]) == 2
    assert record["providers"]["triton"]["launch_configuration"] == (
        CONFIG.triton_launch.to_dict()
    )


def test_deterministic_input_seed_is_case_and_dtype_specific() -> None:
    first = runner._input_seed(17, "case-a", "float16")

    assert first == runner._input_seed(17, "case-a", "float16")
    assert first != runner._input_seed(17, "case-b", "float16")
    assert first != runner._input_seed(17, "case-a", "bfloat16")


def test_source_commit_provenance_agrees_without_git_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40
    (tmp_path / ".source_commit").write_text(
        commit + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", commit)

    record = runner._source_commit_provenance(
        {"commit": None, "available": False}
    )

    assert record["effective_commit"] == commit
    assert record["consistent"] is True
    assert record["sources"] == {
        "git": None,
        "environment": commit,
        "marker_file": commit,
    }


def test_source_commit_provenance_rejects_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".source_commit").write_text(
        "a" * 40, encoding="utf-8"
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", "b" * 40)

    with pytest.raises(RuntimeError, match="provenance disagrees"):
        runner._source_commit_provenance(
            {"commit": "a" * 40, "available": True}
        )


def test_source_commit_provenance_rejects_missing_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("COLPALI_SOURCE_COMMIT", raising=False)

    with pytest.raises(RuntimeError, match="exact source commit"):
        runner._source_commit_provenance(
            {"commit": None, "available": False}
        )


def test_source_commit_provenance_requires_lowercase_full_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("COLPALI_SOURCE_COMMIT", "A" * 40)

    with pytest.raises(RuntimeError, match="full lowercase Git SHA"):
        runner._source_commit_provenance(
            {"commit": None, "available": False}
        )


def test_git_state_survives_missing_git_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", missing_git)

    record = runner._git_state()

    assert record["available"] is False
    assert record["commit"] is None
    assert "FileNotFoundError" in record["error"]


def test_nvidia_smi_query_parses_rows_and_survives_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="0, NVIDIA L4, GPU-123, 550.54.15, 23034\n",
        stderr="",
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: completed
    )
    fields = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory.total",
    )

    record = runner._nvidia_smi_query(fields)

    assert record["available"] is True
    assert record["rows"][0]["name"] == "NVIDIA L4"
    assert record["rows"][0]["driver_version"] == "550.54.15"

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(runner.subprocess, "run", missing)
    unavailable = runner._nvidia_smi_query(fields)
    assert unavailable["available"] is False
    assert "FileNotFoundError" in unavailable["error"]


def test_make_inputs_builds_normalized_masked_shapes_on_cpu() -> None:
    case = CONFIG.cases[-2]

    queries, documents, query_mask, document_mask = runner._make_inputs(
        case,
        dtype=torch.float16,
        device=torch.device("cpu"),
        seed=23,
        normalize_embeddings=True,
    )

    assert queries.shape == (
        case.query_batch_size,
        case.query_tokens,
        case.embedding_dimension,
    )
    assert documents.shape == (
        case.document_batch_size,
        case.document_tokens,
        case.embedding_dimension,
    )
    assert query_mask.sum().item() == (
        case.query_batch_size * case.active_query_tokens
    )
    assert document_mask.sum().item() == (
        case.document_batch_size * case.active_document_tokens
    )
    torch.testing.assert_close(
        queries.float().norm(dim=-1),
        torch.ones(queries.shape[:-1]),
        atol=1e-3,
        rtol=1e-3,
    )


def test_correctness_record_rejects_shape_and_numerical_mismatch() -> None:
    dtype_spec = CONFIG.dtypes[0]
    expected = torch.tensor([[1.0, 2.0]])

    record = runner._correctness(
        expected.clone(), expected, dtype_spec=dtype_spec
    )
    assert record["agreement"] is True
    assert record["maximum_absolute_error"] == 0.0

    with pytest.raises(RuntimeError, match="returned shape"):
        runner._correctness(
            torch.ones(1), expected, dtype_spec=dtype_spec
        )
    with pytest.raises(RuntimeError, match="failed numerical agreement"):
        runner._correctness(
            torch.tensor([[10.0, 20.0]]),
            expected,
            dtype_spec=dtype_spec,
        )


def test_atomic_output_refuses_replacement(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    runner._write_json_atomic(
        output, {"result": 1}, overwrite=False
    )

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_json_atomic(
            output, {"result": 2}, overwrite=False
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"result": 1}
    runner._write_json_atomic(output, {"result": 3}, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"result": 3}


def test_preflight_cli_is_successful_without_cuda(
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = runner.main(["--preflight", "--provider", "torch"])
    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert return_code == 0
    assert output["benchmark_id"] == CONFIG.benchmark_id
    assert output["selection"]["providers"] == ["torch"]
    assert output["preflight"]["status"] in ("ready", "blocked")
