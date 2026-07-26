"""Verified loading of the pinned Phase 7 ColPali base and adapter."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from colpali_triton.adaptation import (
    PAPER_LORA_TARGET_COUNT,
    PAPER_LORA_TRAINABLE_PARAMETERS,
    freeze_for_inference,
    resolve_lora_targets,
)
from colpali_triton.paligemma import ColPali
from colpali_triton.phase7_config import (
    ModelArtifactSpec,
    ModelCheckpointSpec,
    Phase7Config,
)
from colpali_triton.processing import ColPaliProcessor


DownloadFile = Callable[
    [str, str, str, Path, bool],
    Union[str, Path],
]


@dataclass(frozen=True)
class VerifiedModelArtifact:
    """Observed local path and digest for one pinned model artifact."""

    repository: str
    revision: str
    path: str
    local_path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "local_path": str(self.local_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "verified": True,
        }


@dataclass(frozen=True)
class LoadedColPali:
    """Released inference components plus their verified provenance."""

    model: nn.Module
    processor: ColPaliProcessor
    raw_processor: object
    config: Phase7Config
    artifacts: Tuple[VerifiedModelArtifact, ...]
    device: torch.device
    dtype: torch.dtype
    attention_implementation: Optional[str]
    lora_target_count: int
    lora_parameter_count: int


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _default_download_file(
    repository: str,
    revision: str,
    filename: str,
    cache_dir: Path,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "huggingface-hub is required; install the 'multimodal' extra"
        ) from error
    return Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
    )


def _verify_one_artifact(
    checkpoint: ModelCheckpointSpec,
    artifact: ModelArtifactSpec,
    *,
    cache_dir: Path,
    local_files_only: bool,
    download_file: DownloadFile,
) -> VerifiedModelArtifact:
    resolved = Path(
        download_file(
            checkpoint.repository,
            checkpoint.revision,
            artifact.path,
            cache_dir,
            local_files_only,
        )
    ).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"downloaded model artifact is not a file: {resolved}"
        )
    observed_size = resolved.stat().st_size
    if observed_size != artifact.size_bytes:
        raise RuntimeError(
            f"model artifact size mismatch for {artifact.path!r}: "
            f"got {observed_size}, expected {artifact.size_bytes}"
        )
    observed_hash = _sha256_file(resolved)
    if observed_hash != artifact.sha256:
        raise RuntimeError(
            f"model artifact SHA-256 mismatch for {artifact.path!r}: "
            f"got {observed_hash}, expected {artifact.sha256}"
        )
    return VerifiedModelArtifact(
        repository=checkpoint.repository,
        revision=checkpoint.revision,
        path=artifact.path,
        local_path=resolved,
        size_bytes=observed_size,
        sha256=observed_hash,
    )


def verify_checkpoint_artifacts(
    checkpoint: ModelCheckpointSpec,
    *,
    cache_dir: Union[str, Path],
    local_files_only: bool = False,
    download_file: Optional[DownloadFile] = None,
) -> Tuple[VerifiedModelArtifact, ...]:
    """Download or resolve every listed weight and verify bytes exactly."""

    if not isinstance(checkpoint, ModelCheckpointSpec):
        raise TypeError("checkpoint must be a ModelCheckpointSpec")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")
    prepared_cache = Path(cache_dir).resolve()
    prepared_cache.mkdir(parents=True, exist_ok=True)
    downloader = download_file or _default_download_file
    if not callable(downloader):
        raise TypeError("download_file must be callable or None")

    verified = tuple(
        _verify_one_artifact(
            checkpoint,
            artifact,
            cache_dir=prepared_cache,
            local_files_only=local_files_only,
            download_file=downloader,
        )
        for artifact in checkpoint.artifacts
    )
    if len(verified) != checkpoint.expected_artifact_count:
        raise RuntimeError("verified model artifact count drifted")
    if sum(item.size_bytes for item in verified) != (
        checkpoint.total_artifact_bytes
    ):
        raise RuntimeError("verified model artifact byte total drifted")
    return verified


def _select_device(requested: str) -> torch.device:
    if requested not in {"cpu", "mps", "cuda", "auto"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    selected = requested
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if selected == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(selected)


def _paper_mock_image() -> object:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "Pillow is required; install the 'multimodal' extra"
        ) from error
    return Image.new("RGB", (448, 448), color="white")


def _validate_loaded_base(model: ColPali, config: Phase7Config) -> None:
    if not isinstance(model, ColPali):
        raise TypeError("base loader did not return ColPali")
    architecture = config.architecture
    if model.hidden_size != architecture.hidden_size:
        raise RuntimeError(
            f"loaded hidden size is {model.hidden_size}; "
            f"expected {architecture.hidden_size}"
        )
    if (
        model.retrieval_embedding_dim
        != architecture.projection_dimension
    ):
        raise RuntimeError(
            "loaded retrieval projection dimension does not match manifest"
        )
    model_config = model.config
    vision_config = model_config.vision_config
    observed_geometry = (
        int(vision_config.image_size),
        int(vision_config.patch_size),
        int(
            getattr(
                model_config.text_config,
                "num_image_tokens",
                architecture.image_token_count,
            )
        ),
    )
    expected_geometry = (
        architecture.image_size_pixels,
        architecture.patch_size_pixels,
        architecture.image_token_count,
    )
    if observed_geometry != expected_geometry:
        raise RuntimeError(
            f"loaded image geometry is {observed_geometry}; "
            f"expected {expected_geometry}"
        )


def _validate_processor(processor: object, config: Phase7Config) -> None:
    architecture = config.architecture
    if getattr(processor, "image_seq_length", None) != (
        architecture.image_token_count
    ):
        raise RuntimeError("loaded processor image sequence length drifted")
    if getattr(processor, "image_token_id", None) != 257_152:
        raise RuntimeError("loaded processor image token ID drifted")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("loaded processor has no tokenizer")
    unused_id = tokenizer.convert_tokens_to_ids(
        config.processing.query_augmentation_token
    )
    if unused_id != 7:
        raise RuntimeError("loaded <unused0> token ID drifted")


def _adapter_topology(model: nn.Module) -> Tuple[int, int]:
    lora_a_names = tuple(
        name
        for name, _ in model.named_parameters()
        if ".lora_A." in name
    )
    lora_b_names = tuple(
        name
        for name, _ in model.named_parameters()
        if ".lora_B." in name
    )
    if len(lora_a_names) != len(lora_b_names):
        raise RuntimeError("adapter has unequal LoRA A/B tensor counts")
    parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )
    return len(lora_a_names), parameter_count


def _validate_materialized_model(model: nn.Module, label: str) -> None:
    meta_parameters = tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.device.type == "meta"
    )
    meta_buffers = tuple(
        name
        for name, buffer in model.named_buffers()
        if buffer.device.type == "meta"
    )
    if meta_parameters or meta_buffers:
        raise RuntimeError(
            f"{label} contains unmaterialized meta tensors: "
            f"parameters={meta_parameters!r}, buffers={meta_buffers!r}"
        )


def load_released_colpali(
    config: Phase7Config,
    *,
    cache_dir: Union[str, Path],
    device: str = "auto",
    local_files_only: bool = False,
    verify_artifacts: bool = True,
    attention_implementation: Optional[str] = None,
) -> LoadedColPali:
    """Load and audit the immutable released ColPali checkpoint."""

    if not isinstance(config, Phase7Config):
        raise TypeError("config must be a Phase7Config")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")
    if not isinstance(verify_artifacts, bool):
        raise TypeError("verify_artifacts must be a bool")
    if attention_implementation not in {None, "eager", "sdpa"}:
        raise ValueError(
            "attention_implementation must be None, eager, or sdpa"
        )
    prepared_cache = Path(cache_dir).resolve()
    prepared_cache.mkdir(parents=True, exist_ok=True)
    selected_device = _select_device(device)

    artifacts: Tuple[VerifiedModelArtifact, ...] = ()
    if verify_artifacts:
        artifacts = (
            *verify_checkpoint_artifacts(
                config.runtime_base,
                cache_dir=prepared_cache,
                local_files_only=local_files_only,
            ),
            *verify_checkpoint_artifacts(
                config.paper_adapter,
                cache_dir=prepared_cache,
                local_files_only=local_files_only,
            ),
        )

    model_kwargs = {
        "revision": config.runtime_base.revision,
        "cache_dir": str(prepared_cache),
        "local_files_only": local_files_only,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if attention_implementation is not None:
        model_kwargs["attn_implementation"] = attention_implementation
    model = ColPali.from_pretrained(
        config.runtime_base.repository,
        **model_kwargs,
    )
    _validate_materialized_model(model, "loaded base")
    _validate_loaded_base(model, config)
    targets = resolve_lora_targets(model)
    if len(targets) != PAPER_LORA_TARGET_COUNT:
        raise RuntimeError(
            f"loaded base resolves {len(targets)} LoRA targets; "
            f"expected {PAPER_LORA_TARGET_COUNT}"
        )

    try:
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "PEFT and Transformers are required; install the "
            "'multimodal' extra"
        ) from error

    adapted = PeftModel.from_pretrained(
        model,
        config.paper_adapter.repository,
        revision=config.paper_adapter.revision,
        cache_dir=str(prepared_cache),
        local_files_only=local_files_only,
        is_trainable=False,
    )
    _validate_materialized_model(adapted, "loaded adapter model")
    target_count, adapter_parameter_count = _adapter_topology(adapted)
    if target_count != PAPER_LORA_TARGET_COUNT:
        raise RuntimeError(
            f"loaded adapter has {target_count} targets; "
            f"expected {PAPER_LORA_TARGET_COUNT}"
        )
    if adapter_parameter_count != PAPER_LORA_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"loaded adapter has {adapter_parameter_count} parameters; "
            f"expected {PAPER_LORA_TRAINABLE_PARAMETERS}"
        )
    freeze_for_inference(adapted)
    adapted.eval()
    adapted.to(selected_device)

    raw_processor = AutoProcessor.from_pretrained(
        config.runtime_base.repository,
        revision=config.runtime_base.revision,
        cache_dir=str(prepared_cache),
        local_files_only=local_files_only,
    )
    _validate_processor(raw_processor, config)
    processor = ColPaliProcessor(
        raw_processor,
        mock_image_factory=_paper_mock_image,
    )
    return LoadedColPali(
        model=adapted,
        processor=processor,
        raw_processor=raw_processor,
        config=config,
        artifacts=tuple(artifacts),
        device=selected_device,
        dtype=torch.bfloat16,
        attention_implementation=attention_implementation,
        lora_target_count=target_count,
        lora_parameter_count=adapter_parameter_count,
    )


__all__ = [
    "LoadedColPali",
    "VerifiedModelArtifact",
    "load_released_colpali",
    "verify_checkpoint_artifacts",
]
