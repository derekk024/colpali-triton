from hashlib import sha256
from pathlib import Path

import pytest
import torch

from colpali_triton.checkpoints import (
    _validate_materialized_model,
    verify_checkpoint_artifacts,
)
from colpali_triton.phase7_config import (
    ModelArtifactSpec,
    ModelCheckpointSpec,
)


def _checkpoint(content: bytes) -> ModelCheckpointSpec:
    return ModelCheckpointSpec(
        repository="owner/model",
        revision="a" * 40,
        dtype="bfloat16",
        parameter_count=1,
        expected_artifact_count=1,
        artifacts=(
            ModelArtifactSpec(
                path="model.safetensors",
                size_bytes=len(content),
                sha256=sha256(content).hexdigest(),
            ),
        ),
    )


def test_checkpoint_artifact_verification_records_exact_observation(
    tmp_path: Path,
) -> None:
    content = b"deterministic-safe-tensor-fixture"
    artifact_path = tmp_path / "downloaded.safetensors"
    artifact_path.write_bytes(content)
    calls = []

    def download(
        repository: str,
        revision: str,
        filename: str,
        cache_dir: Path,
        local_files_only: bool,
    ) -> Path:
        calls.append(
            (
                repository,
                revision,
                filename,
                cache_dir,
                local_files_only,
            )
        )
        return artifact_path

    verified = verify_checkpoint_artifacts(
        _checkpoint(content),
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        download_file=download,
    )

    assert len(verified) == 1
    assert verified[0].repository == "owner/model"
    assert verified[0].revision == "a" * 40
    assert verified[0].path == "model.safetensors"
    assert verified[0].local_path == artifact_path.resolve()
    assert verified[0].size_bytes == len(content)
    assert verified[0].sha256 == sha256(content).hexdigest()
    assert verified[0].to_dict()["verified"] is True
    assert calls == [
        (
            "owner/model",
            "a" * 40,
            "model.safetensors",
            (tmp_path / "cache").resolve(),
            True,
        )
    ]


def test_checkpoint_verification_rejects_size_before_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"expected"
    actual_path = tmp_path / "wrong.safetensors"
    actual_path.write_bytes(b"wrong-size")
    hash_called = False

    def fail_if_hashed(_: Path) -> str:
        nonlocal hash_called
        hash_called = True
        raise AssertionError("size mismatch must fail before hashing")

    monkeypatch.setattr(
        "colpali_triton.checkpoints._sha256_file",
        fail_if_hashed,
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_checkpoint_artifacts(
            _checkpoint(expected),
            cache_dir=tmp_path / "cache",
            download_file=lambda *_: actual_path,
        )

    assert not hash_called


def test_checkpoint_verification_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    expected = b"same-size-a"
    actual_path = tmp_path / "wrong.safetensors"
    actual_path.write_bytes(b"same-size-b")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_checkpoint_artifacts(
            _checkpoint(expected),
            cache_dir=tmp_path / "cache",
            download_file=lambda *_: actual_path,
        )


def test_checkpoint_verification_rejects_missing_download(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.safetensors"

    with pytest.raises(FileNotFoundError, match="not a file"):
        verify_checkpoint_artifacts(
            _checkpoint(b"x"),
            cache_dir=tmp_path / "cache",
            download_file=lambda *_: missing,
        )


def test_materialization_audit_rejects_meta_parameters() -> None:
    model = torch.nn.Linear(2, 2, device="meta")

    with pytest.raises(RuntimeError, match="unmaterialized meta tensors"):
        _validate_materialized_model(model, "test model")


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    (
        ({"checkpoint": object()}, TypeError),
        ({"local_files_only": 1}, TypeError),
        ({"download_file": "not callable"}, TypeError),
    ),
)
def test_checkpoint_verification_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict,
    error_type: type,
) -> None:
    arguments = {
        "checkpoint": _checkpoint(b"x"),
        "cache_dir": tmp_path / "cache",
        "download_file": lambda *_: tmp_path / "missing",
    }
    arguments.update(kwargs)

    with pytest.raises(error_type):
        verify_checkpoint_artifacts(**arguments)
