from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from colpali_triton.modeling import (
    ColPaliBatch,
    EncodingKind,
    MultiVectorEncoding,
)
from colpali_triton.phase7_config import PHASE7_CONFIG
from scripts import smoke_colpali_mps as runner


class FakeProcessor:
    def process_documents(self, images: object) -> ColPaliBatch:
        del images
        tokens = PHASE7_CONFIG.architecture.image_token_count + 6
        return ColPaliBatch(
            input_ids=torch.ones((1, tokens), dtype=torch.int64),
            attention_mask=torch.ones((1, tokens), dtype=torch.bool),
            kind=EncodingKind.DOCUMENT,
            pixel_values=torch.zeros((1, 3, 448, 448)),
        )

    def process_queries(self, queries: object) -> ColPaliBatch:
        del queries
        return ColPaliBatch(
            input_ids=torch.ones((1, 12), dtype=torch.int64),
            attention_mask=torch.ones((1, 12), dtype=torch.bool),
            kind=EncodingKind.QUERY,
        )


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> MultiVectorEncoding:
        dimension = PHASE7_CONFIG.architecture.projection_dimension
        values = torch.ones(
            (*input_ids.shape, dimension),
            dtype=torch.bfloat16,
            device=input_ids.device,
        )
        values = torch.nn.functional.normalize(values.float(), dim=-1).to(
            torch.bfloat16
        )
        return MultiVectorEncoding(
            embeddings=values,
            mask=attention_mask,
            kind=(
                EncodingKind.DOCUMENT
                if pixel_values is not None
                else EncodingKind.QUERY
            ),
        )


def fake_loader(*args: object, **kwargs: object) -> object:
    del args, kwargs
    return SimpleNamespace(
        model=FakeModel(),
        processor=FakeProcessor(),
        artifacts=(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        attention_implementation="eager",
        lora_target_count=127,
        lora_parameter_count=39_292_928,
    )


def test_smoke_page_is_deterministic() -> None:
    first = runner._image_record(runner._make_smoke_page())
    second = runner._image_record(runner._make_smoke_page())

    assert first == second
    assert first["size_pixels"] == [448, 448]
    assert first["visible_total"] == "$42.00"


def test_atomic_output_refuses_replacement(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    runner._write_json_atomic(output, {"value": 1}, overwrite=False)

    with pytest.raises(FileExistsError, match="already exists"):
        runner._write_json_atomic(output, {"value": 2}, overwrite=False)

    assert json.loads(output.read_text()) == {"value": 1}
    runner._write_json_atomic(output, {"value": 3}, overwrite=True)
    assert json.loads(output.read_text()) == {"value": 3}


def test_preflight_rejects_existing_file_and_directory(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("{}")
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner._preflight_output(existing, overwrite=False)
    runner._preflight_output(existing, overwrite=True)
    with pytest.raises(IsADirectoryError, match="directory"):
        runner._preflight_output(directory, overwrite=True)


def test_encoding_record_checks_norms_and_padding() -> None:
    mask = torch.tensor([[True, False]])
    values = torch.zeros((1, 2, 128))
    values[0, 0, 0] = 1.0
    encoding = MultiVectorEncoding(
        embeddings=values,
        mask=mask,
        kind=EncodingKind.QUERY,
    )

    record = runner._encoding_record(
        encoding, expected_kind=EncodingKind.QUERY
    )

    assert record["shape"] == [1, 2, 128]
    assert record["active_vector_count"] == 1
    assert record["masked_nonzero_values"] == 0


def test_encoding_record_rejects_wrong_kind_and_norm() -> None:
    values = torch.zeros((1, 1, 128))
    values[0, 0, 0] = 0.5
    encoding = MultiVectorEncoding(
        embeddings=values,
        mask=torch.ones((1, 1), dtype=torch.bool),
        kind=EncodingKind.QUERY,
    )

    with pytest.raises(RuntimeError, match="expected document"):
        runner._encoding_record(
            encoding, expected_kind=EncodingKind.DOCUMENT
        )
    with pytest.raises(RuntimeError, match="not unit normalized"):
        runner._encoding_record(
            encoding, expected_kind=EncodingKind.QUERY
        )


def test_run_smoke_exercises_full_contract_with_fake_loader(
    tmp_path: Path,
) -> None:
    result = runner.run_smoke(
        PHASE7_CONFIG,
        model_cache=tmp_path,
        device="cpu",
        attention_implementation="eager",
        query="What is the invoice total?",
        local_files_only=True,
        loader=fake_loader,
    )

    assert result["input"]["document_input_shape"] == [1, 1030]
    assert result["encoding"]["document"]["shape"] == [1, 1030, 128]
    assert result["encoding"]["query"]["shape"] == [1, 12, 128]
    assert result["score"]["shape"] == [1, 1]
    assert result["score"]["finite"] is True
    assert result["checkpoint"]["lora_target_count"] == 127
    assert result["model"]["trainable_parameters"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"device": "cuda"}, "device must be"),
        ({"attention_implementation": "flash"}, "must be eager"),
        ({"query": " "}, "nonblank"),
        ({"local_files_only": 1}, "must be a bool"),
    ],
)
def test_run_smoke_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    arguments = {
        "model_cache": tmp_path,
        "device": "cpu",
        "attention_implementation": "eager",
        "query": "valid",
        "local_files_only": True,
        "loader": fake_loader,
        **kwargs,
    }

    with pytest.raises((TypeError, ValueError), match=message):
        runner.run_smoke(PHASE7_CONFIG, **arguments)
