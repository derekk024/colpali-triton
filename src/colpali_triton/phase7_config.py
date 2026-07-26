"""Strict, immutable configuration for the Phase 7 ColPali checkpoints.

The manifest deliberately separates the gated Google source checkpoint from
the public, BF16 ViDoRe base used for deterministic local loading.  Only
weight artifacts with authoritative Hugging Face LFS SHA-256 metadata are
listed.  Small tokenizer and processor files remain revision-pinned and are
loaded through the same checkpoint revision, but are not represented as
content-verified artifacts here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path, PurePosixPath
import re
from typing import Dict, FrozenSet, Mapping, Sequence, Tuple, Union
from urllib.parse import quote


_FULL_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
_FORMAT = "colpali-triton-phase7-checkpoints-v1"
_SOURCE_REPOSITORY = "google/paligemma-3b-mix-448"
_SOURCE_REVISION = "ead2d9a35598cb89119af004f5d023b311d1c4a1"
_BASE_REPOSITORY = "vidore/colpaligemma-3b-mix-448-base"
_BASE_REVISION = "6ff0d944ea09c3ead97d2bc57427e3d4f01d192f"
_ADAPTER_REPOSITORY = "vidore/colpali"
_ADAPTER_REVISION = "234ecbefa176542348dc5fae4f95c9736858edc2"

_SOURCE_ARTIFACTS = (
    (
        "model-00001-of-00003.safetensors",
        4_956_951_424,
        "570dab6f84d3b784a06707cdc4742f97545dfd57d73742bb2fcb3190a09696a4",
    ),
    (
        "model-00002-of-00003.safetensors",
        4_999_820_608,
        "334b225c0ec1db8f3952121f5f67a78b37167623e2178c0babe3086fcc8ea4ad",
    ),
    (
        "model-00003-of-00003.safetensors",
        1_740_714_288,
        "8c75421941def510a8c2364726d8ab36cf1a0653b355368d2e2a80766b5a4f5f",
    ),
)
_BASE_ARTIFACTS = (
    (
        "model-00001-of-00002.safetensors",
        4_986_817_288,
        "8b439a322fa98964517fca8fbf0a1d17e6e60e7e8812bc1c373ad87d654108d5",
    ),
    (
        "model-00002-of-00002.safetensors",
        862_495_528,
        "ad075b39b4a24b309028f3d003a0e7cad9e45fa2dbbd267867474e77e10df814",
    ),
)
_ADAPTER_ARTIFACTS = (
    (
        "adapter_model.safetensors",
        78_625_112,
        "961b72c2b2a1bebc3e11e7d98cd173d263437aad5ef36e760e9265c184c88d64",
    ),
)

_LORA_TARGET_MODULES_PATTERN = (
    r"(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|"
    r"v_proj|o_proj).*$|.*(custom_text_proj).*$)"
)


@dataclass(frozen=True)
class ModelArtifactSpec:
    """Expected metadata for one revision-pinned model weight artifact."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("artifact path must be a nonempty string")
        parsed = PurePosixPath(self.path)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or self.path.endswith("/")
            or parsed.suffix != ".safetensors"
        ):
            raise ValueError(
                "artifact path must be a relative .safetensors path "
                "without '..'"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("artifact size_bytes must be a positive integer")
        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ValueError(
                "artifact sha256 must be 64 lowercase hex characters"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ModelCheckpointSpec:
    """Pinned Hugging Face model repository and verifiable weight files."""

    repository: str
    revision: str
    dtype: str
    parameter_count: int
    expected_artifact_count: int
    artifacts: Tuple[ModelArtifactSpec, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository, str)
            or _REPOSITORY_PATTERN.fullmatch(self.repository) is None
        ):
            raise ValueError(
                "checkpoint repository must have the form 'owner/model'"
            )
        if (
            not isinstance(self.revision, str)
            or _FULL_REVISION_PATTERN.fullmatch(self.revision) is None
        ):
            raise ValueError(
                "checkpoint revision must be a full 40-character lowercase "
                "commit SHA"
            )
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                "checkpoint dtype must be 'float32' or 'bfloat16'"
            )
        if (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, int)
            or self.parameter_count <= 0
        ):
            raise ValueError(
                "checkpoint parameter_count must be a positive integer"
            )
        if (
            isinstance(self.expected_artifact_count, bool)
            or not isinstance(self.expected_artifact_count, int)
            or self.expected_artifact_count <= 0
        ):
            raise ValueError(
                "checkpoint expected_artifact_count must be a positive integer"
            )

        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ModelArtifactSpec) for item in artifacts):
            raise TypeError(
                "checkpoint artifacts must contain ModelArtifactSpec values"
            )
        if len(artifacts) != self.expected_artifact_count:
            raise ValueError(
                "checkpoint artifact count does not match "
                "expected_artifact_count"
            )
        paths = tuple(item.path for item in artifacts)
        if len(set(paths)) != len(paths):
            raise ValueError("checkpoint artifact paths must be unique")
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def total_artifact_bytes(self) -> int:
        """Return the exact combined size of listed weight artifacts."""

        return sum(item.size_bytes for item in self.artifacts)

    def artifact(self, path: str) -> ModelArtifactSpec:
        """Return one listed artifact by path."""

        for item in self.artifacts:
            if item.path == path:
                return item
        raise KeyError(
            f"checkpoint {self.repository!r} has no artifact {path!r}"
        )

    def resolve_url(self, path: str) -> str:
        """Return an immutable Hugging Face URL for a listed artifact."""

        artifact = self.artifact(path)
        return huggingface_model_resolve_url(
            self.repository,
            self.revision,
            artifact.path,
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "repository": self.repository,
            "revision": self.revision,
            "dtype": self.dtype,
            "parameter_count": self.parameter_count,
            "expected_artifact_count": self.expected_artifact_count,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


@dataclass(frozen=True)
class SourceTermsSpec:
    """Access and license metadata for the upstream Google backbone."""

    access: str
    license_id: str

    def __post_init__(self) -> None:
        if self.access != "manual_gated":
            raise ValueError("source access must be exactly 'manual_gated'")
        if self.license_id != "gemma":
            raise ValueError("source license_id must be exactly 'gemma'")

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-ready representation."""

        return {"access": self.access, "license_id": self.license_id}


@dataclass(frozen=True)
class ColPaliArchitectureSpec:
    """Retrieval-specific dimensions and representation behavior."""

    hidden_size: int
    projection_dimension: int
    image_size_pixels: int
    patch_size_pixels: int
    image_token_count: int
    embedding_normalization: str
    masked_embedding_value: float

    def __post_init__(self) -> None:
        for label, value in (
            ("hidden_size", self.hidden_size),
            ("projection_dimension", self.projection_dimension),
            ("image_size_pixels", self.image_size_pixels),
            ("patch_size_pixels", self.patch_size_pixels),
            ("image_token_count", self.image_token_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        if self.image_size_pixels % self.patch_size_pixels != 0:
            raise ValueError(
                "image_size_pixels must be divisible by patch_size_pixels"
            )
        derived_count = (
            self.image_size_pixels // self.patch_size_pixels
        ) ** 2
        if self.image_token_count != derived_count:
            raise ValueError(
                "image_token_count must equal the square patch-grid size"
            )
        if self.embedding_normalization != "l2_last_dimension":
            raise ValueError(
                "embedding_normalization must be 'l2_last_dimension'"
            )
        if (
            isinstance(self.masked_embedding_value, bool)
            or not isinstance(self.masked_embedding_value, Real)
            or not math.isfinite(float(self.masked_embedding_value))
            or float(self.masked_embedding_value) != 0.0
        ):
            raise ValueError("masked_embedding_value must be exactly 0.0")
        expected = (2048, 128, 448, 14, 1024)
        observed = (
            self.hidden_size,
            self.projection_dimension,
            self.image_size_pixels,
            self.patch_size_pixels,
            self.image_token_count,
        )
        if observed != expected:
            raise ValueError(
                "Phase 7 architecture must use H=2048, D=128, a 448-pixel "
                "image, 14-pixel patches, and 1024 image tokens"
            )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "hidden_size": self.hidden_size,
            "projection_dimension": self.projection_dimension,
            "image_size_pixels": self.image_size_pixels,
            "patch_size_pixels": self.patch_size_pixels,
            "image_token_count": self.image_token_count,
            "embedding_normalization": self.embedding_normalization,
            "masked_embedding_value": float(self.masked_embedding_value),
        }


@dataclass(frozen=True)
class ColPaliProcessingSpec:
    """Exact document prompt and paper query augmentation contract."""

    document_prompt: str
    query_prefix: str
    query_augmentation_token: str
    query_augmentation_count: int
    text_max_length: int
    query_augmentation_separator: str

    def __post_init__(self) -> None:
        if self.document_prompt != "Describe the image.":
            raise ValueError(
                "document_prompt must be exactly 'Describe the image.'"
            )
        if self.query_prefix != "Question: ":
            raise ValueError("query_prefix must be exactly 'Question: '")
        if self.query_augmentation_token != "<unused0>":
            raise ValueError(
                "query_augmentation_token must be exactly '<unused0>'"
            )
        if (
            isinstance(self.query_augmentation_count, bool)
            or self.query_augmentation_count != 5
        ):
            raise ValueError("query_augmentation_count must be exactly 5")
        if (
            isinstance(self.text_max_length, bool)
            or self.text_max_length != 50
        ):
            raise ValueError("text_max_length must be exactly 50")
        if self.query_augmentation_separator != "":
            raise ValueError(
                "query_augmentation_separator must be the empty string"
            )

    @property
    def query_augmentation_suffix(self) -> str:
        """Return the exact five-token query augmentation suffix."""

        separator = self.query_augmentation_separator
        return separator.join(
            [self.query_augmentation_token] * self.query_augmentation_count
        )

    def format_query(self, query: str) -> str:
        """Apply the exact Phase 7 prefix and augmentation to one query."""

        if not isinstance(query, str) or not query:
            raise ValueError("query must be a nonempty string")
        return f"{self.query_prefix}{query}{self.query_augmentation_suffix}"

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "document_prompt": self.document_prompt,
            "query_prefix": self.query_prefix,
            "query_augmentation_token": self.query_augmentation_token,
            "query_augmentation_count": self.query_augmentation_count,
            "query_augmentation_separator": (
                self.query_augmentation_separator
            ),
            "text_max_length": self.text_max_length,
        }


@dataclass(frozen=True)
class LoraContractSpec:
    """Exact LoRA configuration recorded with the paper checkpoint."""

    rank: int
    alpha: int
    dropout: float
    bias: str
    initialization: str
    task_type: str
    target_modules_pattern: str
    peft_type: str
    inference_mode: bool
    use_dora: bool
    use_rslora: bool
    modules_to_save: None

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or self.rank != 32:
            raise ValueError("LoRA rank must be exactly 32")
        if isinstance(self.alpha, bool) or self.alpha != 32:
            raise ValueError("LoRA alpha must be exactly 32")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, Real)
            or not math.isfinite(float(self.dropout))
            or float(self.dropout) != 0.1
        ):
            raise ValueError("LoRA dropout must be exactly 0.1")
        if self.bias != "none":
            raise ValueError("LoRA bias must be exactly 'none'")
        if self.initialization != "gaussian":
            raise ValueError(
                "LoRA initialization must be exactly 'gaussian'"
            )
        if self.task_type != "FEATURE_EXTRACTION":
            raise ValueError(
                "LoRA task_type must be exactly 'FEATURE_EXTRACTION'"
            )
        if self.target_modules_pattern != _LORA_TARGET_MODULES_PATTERN:
            raise ValueError("LoRA target_modules_pattern does not match")
        if self.peft_type != "LORA":
            raise ValueError("LoRA peft_type must be exactly 'LORA'")
        for label, value in (
            ("inference_mode", self.inference_mode),
            ("use_dora", self.use_dora),
            ("use_rslora", self.use_rslora),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"LoRA {label} must be a boolean")
        if self.inference_mode:
            raise ValueError("LoRA inference_mode must be false for training")
        if self.use_dora:
            raise ValueError("LoRA use_dora must be false")
        if self.use_rslora:
            raise ValueError("LoRA use_rslora must be false")
        if self.modules_to_save is not None:
            raise ValueError("LoRA modules_to_save must be null")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": float(self.dropout),
            "bias": self.bias,
            "initialization": self.initialization,
            "task_type": self.task_type,
            "target_modules_pattern": self.target_modules_pattern,
            "peft_type": self.peft_type,
            "inference_mode": self.inference_mode,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "modules_to_save": self.modules_to_save,
        }


@dataclass(frozen=True)
class Phase7Config:
    """Complete, immutable Phase 7 checkpoint and architecture contract."""

    schema_version: int
    format: str
    source_backbone: ModelCheckpointSpec
    source_terms: SourceTermsSpec
    runtime_base: ModelCheckpointSpec
    paper_adapter: ModelCheckpointSpec
    architecture: ColPaliArchitectureSpec
    processing: ColPaliProcessingSpec
    lora: LoraContractSpec

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("Phase 7 schema_version must be exactly 1")
        if self.format != _FORMAT:
            raise ValueError(f"Phase 7 format must be exactly {_FORMAT!r}")
        for label, value, expected_type in (
            ("source_backbone", self.source_backbone, ModelCheckpointSpec),
            ("source_terms", self.source_terms, SourceTermsSpec),
            ("runtime_base", self.runtime_base, ModelCheckpointSpec),
            ("paper_adapter", self.paper_adapter, ModelCheckpointSpec),
            ("architecture", self.architecture, ColPaliArchitectureSpec),
            ("processing", self.processing, ColPaliProcessingSpec),
            ("lora", self.lora, LoraContractSpec),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{label} has the wrong configuration type")

        self._require_checkpoint(
            self.source_backbone,
            label="source_backbone",
            repository=_SOURCE_REPOSITORY,
            revision=_SOURCE_REVISION,
            dtype="float32",
            parameter_count=2_924_351_216,
            expected_artifacts=_SOURCE_ARTIFACTS,
        )
        self._require_checkpoint(
            self.runtime_base,
            label="runtime_base",
            repository=_BASE_REPOSITORY,
            revision=_BASE_REVISION,
            dtype="bfloat16",
            parameter_count=2_924_613_488,
            expected_artifacts=_BASE_ARTIFACTS,
        )
        self._require_checkpoint(
            self.paper_adapter,
            label="paper_adapter",
            repository=_ADAPTER_REPOSITORY,
            revision=_ADAPTER_REVISION,
            dtype="bfloat16",
            parameter_count=39_292_928,
            expected_artifacts=_ADAPTER_ARTIFACTS,
        )

    @staticmethod
    def _require_checkpoint(
        checkpoint: ModelCheckpointSpec,
        *,
        label: str,
        repository: str,
        revision: str,
        dtype: str,
        parameter_count: int,
        expected_artifacts: Sequence[Tuple[str, int, str]],
    ) -> None:
        expected_metadata = tuple(expected_artifacts)
        observed_metadata = tuple(
            (item.path, item.size_bytes, item.sha256)
            for item in checkpoint.artifacts
        )
        if (
            checkpoint.repository != repository
            or checkpoint.revision != revision
            or checkpoint.dtype != dtype
            or checkpoint.parameter_count != parameter_count
            or checkpoint.expected_artifact_count != len(expected_metadata)
            or observed_metadata != expected_metadata
        ):
            raise ValueError(
                f"{label} does not match the immutable Phase 7 checkpoint pin"
            )

    @property
    def fingerprint(self) -> str:
        """Return SHA-256 of the canonical semantic JSON representation."""

        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def canonical_json(self) -> str:
        """Return deterministic, whitespace-free, key-sorted JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "source_backbone": self.source_backbone.to_dict(),
            "source_terms": self.source_terms.to_dict(),
            "runtime_base": self.runtime_base.to_dict(),
            "paper_adapter": self.paper_adapter.to_dict(),
            "architecture": self.architecture.to_dict(),
            "processing": self.processing.to_dict(),
            "lora": self.lora.to_dict(),
        }


def huggingface_model_resolve_url(
    repository: str,
    revision: str,
    path: str,
) -> str:
    """Build an immutable Hugging Face model-file URL."""

    if (
        not isinstance(repository, str)
        or _REPOSITORY_PATTERN.fullmatch(repository) is None
    ):
        raise ValueError("repository must have the form 'owner/model'")
    if (
        not isinstance(revision, str)
        or _FULL_REVISION_PATTERN.fullmatch(revision) is None
    ):
        raise ValueError("revision must be a full lowercase commit SHA")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a nonempty string")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or path.endswith("/"):
        raise ValueError("path must be relative and may not contain '..'")
    return (
        "https://huggingface.co/"
        f"{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(path, safe='/')}"
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _keys(
    value: Mapping[str, object],
    expected: FrozenSet[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing keys {missing!r}")
        if unknown:
            details.append(f"unknown keys {unknown!r}")
        raise ValueError(f"{label} has " + " and ".join(details))


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _null(value: object, label: str) -> None:
    if value is not None:
        raise ValueError(f"{label} must be null")
    return None


def _parse_artifact(value: object, label: str) -> ModelArtifactSpec:
    data = _mapping(value, label)
    _keys(data, frozenset({"path", "size_bytes", "sha256"}), label)
    return ModelArtifactSpec(
        path=_string(data["path"], f"{label}.path"),
        size_bytes=_integer(data["size_bytes"], f"{label}.size_bytes"),
        sha256=_string(data["sha256"], f"{label}.sha256"),
    )


def _parse_checkpoint(value: object, label: str) -> ModelCheckpointSpec:
    data = _mapping(value, label)
    _keys(
        data,
        frozenset(
            {
                "repository",
                "revision",
                "dtype",
                "parameter_count",
                "expected_artifact_count",
                "artifacts",
            }
        ),
        label,
    )
    raw_artifacts = data["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ValueError(f"{label}.artifacts must be a JSON array")
    artifacts = tuple(
        _parse_artifact(item, f"{label}.artifacts[{index}]")
        for index, item in enumerate(raw_artifacts)
    )
    return ModelCheckpointSpec(
        repository=_string(data["repository"], f"{label}.repository"),
        revision=_string(data["revision"], f"{label}.revision"),
        dtype=_string(data["dtype"], f"{label}.dtype"),
        parameter_count=_integer(
            data["parameter_count"], f"{label}.parameter_count"
        ),
        expected_artifact_count=_integer(
            data["expected_artifact_count"],
            f"{label}.expected_artifact_count",
        ),
        artifacts=artifacts,
    )


def phase7_config_from_dict(value: object) -> Phase7Config:
    """Parse and fully validate an in-memory Phase 7 manifest object."""

    root = _mapping(value, "phase7_config")
    _keys(
        root,
        frozenset(
            {
                "schema_version",
                "format",
                "source_backbone",
                "source_terms",
                "runtime_base",
                "paper_adapter",
                "architecture",
                "processing",
                "lora",
            }
        ),
        "phase7_config",
    )

    terms = _mapping(root["source_terms"], "source_terms")
    _keys(terms, frozenset({"access", "license_id"}), "source_terms")

    architecture = _mapping(root["architecture"], "architecture")
    _keys(
        architecture,
        frozenset(
            {
                "hidden_size",
                "projection_dimension",
                "image_size_pixels",
                "patch_size_pixels",
                "image_token_count",
                "embedding_normalization",
                "masked_embedding_value",
            }
        ),
        "architecture",
    )

    processing = _mapping(root["processing"], "processing")
    _keys(
        processing,
        frozenset(
            {
                "document_prompt",
                "query_prefix",
                "query_augmentation_token",
                "query_augmentation_count",
                "query_augmentation_separator",
                "text_max_length",
            }
        ),
        "processing",
    )

    lora = _mapping(root["lora"], "lora")
    _keys(
        lora,
        frozenset(
            {
                "rank",
                "alpha",
                "dropout",
                "bias",
                "initialization",
                "task_type",
                "target_modules_pattern",
                "peft_type",
                "inference_mode",
                "use_dora",
                "use_rslora",
                "modules_to_save",
            }
        ),
        "lora",
    )

    return Phase7Config(
        schema_version=_integer(root["schema_version"], "schema_version"),
        format=_string(root["format"], "format"),
        source_backbone=_parse_checkpoint(
            root["source_backbone"], "source_backbone"
        ),
        source_terms=SourceTermsSpec(
            access=_string(terms["access"], "source_terms.access"),
            license_id=_string(
                terms["license_id"], "source_terms.license_id"
            ),
        ),
        runtime_base=_parse_checkpoint(root["runtime_base"], "runtime_base"),
        paper_adapter=_parse_checkpoint(
            root["paper_adapter"], "paper_adapter"
        ),
        architecture=ColPaliArchitectureSpec(
            hidden_size=_integer(
                architecture["hidden_size"], "architecture.hidden_size"
            ),
            projection_dimension=_integer(
                architecture["projection_dimension"],
                "architecture.projection_dimension",
            ),
            image_size_pixels=_integer(
                architecture["image_size_pixels"],
                "architecture.image_size_pixels",
            ),
            patch_size_pixels=_integer(
                architecture["patch_size_pixels"],
                "architecture.patch_size_pixels",
            ),
            image_token_count=_integer(
                architecture["image_token_count"],
                "architecture.image_token_count",
            ),
            embedding_normalization=_string(
                architecture["embedding_normalization"],
                "architecture.embedding_normalization",
            ),
            masked_embedding_value=_number(
                architecture["masked_embedding_value"],
                "architecture.masked_embedding_value",
            ),
        ),
        processing=ColPaliProcessingSpec(
            document_prompt=_string(
                processing["document_prompt"],
                "processing.document_prompt",
            ),
            query_prefix=_string(
                processing["query_prefix"], "processing.query_prefix"
            ),
            query_augmentation_token=_string(
                processing["query_augmentation_token"],
                "processing.query_augmentation_token",
            ),
            query_augmentation_count=_integer(
                processing["query_augmentation_count"],
                "processing.query_augmentation_count",
            ),
            query_augmentation_separator=_string(
                processing["query_augmentation_separator"],
                "processing.query_augmentation_separator",
            ),
            text_max_length=_integer(
                processing["text_max_length"],
                "processing.text_max_length",
            ),
        ),
        lora=LoraContractSpec(
            rank=_integer(lora["rank"], "lora.rank"),
            alpha=_integer(lora["alpha"], "lora.alpha"),
            dropout=_number(lora["dropout"], "lora.dropout"),
            bias=_string(lora["bias"], "lora.bias"),
            initialization=_string(
                lora["initialization"], "lora.initialization"
            ),
            task_type=_string(lora["task_type"], "lora.task_type"),
            target_modules_pattern=_string(
                lora["target_modules_pattern"],
                "lora.target_modules_pattern",
            ),
            peft_type=_string(lora["peft_type"], "lora.peft_type"),
            inference_mode=_boolean(
                lora["inference_mode"], "lora.inference_mode"
            ),
            use_dora=_boolean(lora["use_dora"], "lora.use_dora"),
            use_rslora=_boolean(lora["use_rslora"], "lora.use_rslora"),
            modules_to_save=_null(
                lora["modules_to_save"], "lora.modules_to_save"
            ),
        ),
    )


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def load_phase7_config(path: Union[str, Path]) -> Phase7Config:
    """Read and strictly validate a Phase 7 JSON manifest."""

    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError) as error:
        raise ValueError(
            f"could not read Phase 7 config {path!r}: {error}"
        ) from error
    try:
        parsed = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Phase 7 config {str(path)!r} is not valid JSON: {error}"
        ) from error
    return phase7_config_from_dict(parsed)


def _artifact_specs(
    metadata: Sequence[Tuple[str, int, str]],
) -> Tuple[ModelArtifactSpec, ...]:
    return tuple(
        ModelArtifactSpec(path=path, size_bytes=size, sha256=digest)
        for path, size, digest in metadata
    )


PHASE7_CONFIG = Phase7Config(
    schema_version=1,
    format=_FORMAT,
    source_backbone=ModelCheckpointSpec(
        repository=_SOURCE_REPOSITORY,
        revision=_SOURCE_REVISION,
        dtype="float32",
        parameter_count=2_924_351_216,
        expected_artifact_count=3,
        artifacts=_artifact_specs(_SOURCE_ARTIFACTS),
    ),
    source_terms=SourceTermsSpec(
        access="manual_gated",
        license_id="gemma",
    ),
    runtime_base=ModelCheckpointSpec(
        repository=_BASE_REPOSITORY,
        revision=_BASE_REVISION,
        dtype="bfloat16",
        parameter_count=2_924_613_488,
        expected_artifact_count=2,
        artifacts=_artifact_specs(_BASE_ARTIFACTS),
    ),
    paper_adapter=ModelCheckpointSpec(
        repository=_ADAPTER_REPOSITORY,
        revision=_ADAPTER_REVISION,
        dtype="bfloat16",
        parameter_count=39_292_928,
        expected_artifact_count=1,
        artifacts=_artifact_specs(_ADAPTER_ARTIFACTS),
    ),
    architecture=ColPaliArchitectureSpec(
        hidden_size=2048,
        projection_dimension=128,
        image_size_pixels=448,
        patch_size_pixels=14,
        image_token_count=1024,
        embedding_normalization="l2_last_dimension",
        masked_embedding_value=0.0,
    ),
    processing=ColPaliProcessingSpec(
        document_prompt="Describe the image.",
        query_prefix="Question: ",
        query_augmentation_token="<unused0>",
        query_augmentation_count=5,
        query_augmentation_separator="",
        text_max_length=50,
    ),
    lora=LoraContractSpec(
        rank=32,
        alpha=32,
        dropout=0.1,
        bias="none",
        initialization="gaussian",
        task_type="FEATURE_EXTRACTION",
        target_modules_pattern=_LORA_TARGET_MODULES_PATTERN,
        peft_type="LORA",
        inference_mode=False,
        use_dora=False,
        use_rslora=False,
        modules_to_save=None,
    ),
)
