"""Validated configuration and statistics for reproducible MaxSim benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


CONFIG_SCHEMA_VERSION = 1
SUPPORTED_PROVIDERS = ("torch", "triton")
SUPPORTED_DTYPES = ("float16", "bfloat16")
SUPPORTED_TRITON_BLOCK_SIZES = (16, 32, 64, 128)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    *,
    name: str,
    required: Iterable[str],
) -> None:
    required_keys = set(required)
    actual_keys = set(value)
    missing = sorted(required_keys - actual_keys)
    unexpected = sorted(actual_keys - required_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{name} has invalid keys: {', '.join(details)}")


def _require_int(
    value: object, name: str, *, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _require_float(
    value: object, name: str, *, minimum: float = 0.0
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(
            f"{name} must be finite and at least {minimum}; got {value}"
        )
    return result


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be blank")
    return result


@dataclass(frozen=True)
class BenchmarkCase:
    """One exact all-pairs MaxSim input shape."""

    name: str
    query_batch_size: int
    document_batch_size: int
    query_tokens: int
    document_tokens: int
    embedding_dimension: int
    active_query_tokens: int
    active_document_tokens: int

    @property
    def query_document_pairs(self) -> int:
        return self.query_batch_size * self.document_batch_size

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, index: int
    ) -> "BenchmarkCase":
        name = f"cases[{index}]"
        _require_exact_keys(
            value,
            name=name,
            required=(
                "name",
                "query_batch_size",
                "document_batch_size",
                "query_tokens",
                "document_tokens",
                "embedding_dimension",
                "active_query_tokens",
                "active_document_tokens",
            ),
        )
        result = cls(
            name=_require_nonempty_string(value["name"], f"{name}.name"),
            query_batch_size=_require_int(
                value["query_batch_size"],
                f"{name}.query_batch_size",
                minimum=1,
            ),
            document_batch_size=_require_int(
                value["document_batch_size"],
                f"{name}.document_batch_size",
                minimum=1,
            ),
            query_tokens=_require_int(
                value["query_tokens"], f"{name}.query_tokens", minimum=1
            ),
            document_tokens=_require_int(
                value["document_tokens"],
                f"{name}.document_tokens",
                minimum=1,
            ),
            embedding_dimension=_require_int(
                value["embedding_dimension"],
                f"{name}.embedding_dimension",
                minimum=1,
            ),
            active_query_tokens=_require_int(
                value["active_query_tokens"],
                f"{name}.active_query_tokens",
                minimum=1,
            ),
            active_document_tokens=_require_int(
                value["active_document_tokens"],
                f"{name}.active_document_tokens",
                minimum=1,
            ),
        )
        if result.active_query_tokens > result.query_tokens:
            raise ValueError(
                f"{name}.active_query_tokens cannot exceed query_tokens"
            )
        if result.active_document_tokens > result.document_tokens:
            raise ValueError(
                f"{name}.active_document_tokens cannot exceed document_tokens"
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "query_batch_size": self.query_batch_size,
            "document_batch_size": self.document_batch_size,
            "query_tokens": self.query_tokens,
            "document_tokens": self.document_tokens,
            "embedding_dimension": self.embedding_dimension,
            "active_query_tokens": self.active_query_tokens,
            "active_document_tokens": self.active_document_tokens,
        }


@dataclass(frozen=True)
class DTypeSpec:
    """A benchmark dtype and its numerical-agreement thresholds."""

    name: str
    absolute_tolerance: float
    relative_tolerance: float

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, index: int
    ) -> "DTypeSpec":
        name = f"dtypes[{index}]"
        _require_exact_keys(
            value,
            name=name,
            required=("name", "absolute_tolerance", "relative_tolerance"),
        )
        dtype_name = _require_nonempty_string(value["name"], f"{name}.name")
        if dtype_name not in SUPPORTED_DTYPES:
            raise ValueError(
                f"{name}.name must be one of {SUPPORTED_DTYPES}; "
                f"got {dtype_name!r}"
            )
        return cls(
            name=dtype_name,
            absolute_tolerance=_require_float(
                value["absolute_tolerance"],
                f"{name}.absolute_tolerance",
            ),
            relative_tolerance=_require_float(
                value["relative_tolerance"],
                f"{name}.relative_tolerance",
            ),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True)
class TritonLaunchSpec:
    """Exact Triton launch parameters used by every configured case."""

    block_query_tokens: int
    block_document_tokens: int
    block_embedding_dimension: int
    num_warps: int
    num_stages: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "TritonLaunchSpec":
        _require_exact_keys(
            value,
            name="triton_launch",
            required=(
                "block_query_tokens",
                "block_document_tokens",
                "block_embedding_dimension",
                "num_warps",
                "num_stages",
            ),
        )
        result = cls(
            block_query_tokens=_require_int(
                value["block_query_tokens"],
                "triton_launch.block_query_tokens",
                minimum=1,
            ),
            block_document_tokens=_require_int(
                value["block_document_tokens"],
                "triton_launch.block_document_tokens",
                minimum=1,
            ),
            block_embedding_dimension=_require_int(
                value["block_embedding_dimension"],
                "triton_launch.block_embedding_dimension",
                minimum=1,
            ),
            num_warps=_require_int(
                value["num_warps"],
                "triton_launch.num_warps",
                minimum=1,
            ),
            num_stages=_require_int(
                value["num_stages"],
                "triton_launch.num_stages",
                minimum=1,
            ),
        )
        for field_name in (
            "block_query_tokens",
            "block_document_tokens",
            "block_embedding_dimension",
        ):
            field_value = getattr(result, field_name)
            if field_value not in SUPPORTED_TRITON_BLOCK_SIZES:
                raise ValueError(
                    f"triton_launch.{field_name} must be one of "
                    f"{SUPPORTED_TRITON_BLOCK_SIZES}; got {field_value}"
                )
        if result.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "triton_launch.num_warps must be one of (1, 2, 4, 8)"
            )
        if result.num_stages > 5:
            raise ValueError(
                "triton_launch.num_stages must be between 1 and 5"
            )
        return result

    def to_dict(self) -> Dict[str, int]:
        return {
            "block_query_tokens": self.block_query_tokens,
            "block_document_tokens": self.block_document_tokens,
            "block_embedding_dimension": self.block_embedding_dimension,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Strict schema for one canonical CUDA benchmark matrix."""

    benchmark_id: str
    seed: int
    device_index: int
    warmup_iterations: int
    measured_iterations: int
    trials: int
    providers: Tuple[str, ...]
    dtypes: Tuple[DTypeSpec, ...]
    cases: Tuple[BenchmarkCase, ...]
    normalize_embeddings: bool
    triton_launch: TritonLaunchSpec

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "BenchmarkConfig":
        _require_exact_keys(
            value,
            name="config",
            required=(
                "schema_version",
                "benchmark_id",
                "seed",
                "device",
                "timing",
                "providers",
                "dtypes",
                "cases",
                "normalize_embeddings",
                "triton_launch",
            ),
        )
        schema_version = _require_int(
            value["schema_version"], "schema_version", minimum=1
        )
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported schema_version "
                f"{schema_version}; expected {CONFIG_SCHEMA_VERSION}"
            )

        device = _require_mapping(value["device"], "device")
        _require_exact_keys(
            device, name="device", required=("type", "index")
        )
        if device["type"] != "cuda":
            raise ValueError("device.type must be 'cuda'")

        timing = _require_mapping(value["timing"], "timing")
        _require_exact_keys(
            timing,
            name="timing",
            required=(
                "warmup_iterations",
                "measured_iterations",
                "trials",
            ),
        )

        raw_providers = value["providers"]
        if not isinstance(raw_providers, list) or not raw_providers:
            raise TypeError("providers must be a nonempty JSON array")
        providers = tuple(
            _require_nonempty_string(item, f"providers[{index}]")
            for index, item in enumerate(raw_providers)
        )
        invalid_providers = sorted(
            set(providers) - set(SUPPORTED_PROVIDERS)
        )
        if invalid_providers:
            raise ValueError(
                f"providers contains unsupported values {invalid_providers}"
            )
        if len(set(providers)) != len(providers):
            raise ValueError("providers must not contain duplicates")
        if providers[0] != "torch":
            raise ValueError(
                "providers must list 'torch' first as the correctness oracle"
            )

        raw_dtypes = value["dtypes"]
        if not isinstance(raw_dtypes, list) or not raw_dtypes:
            raise TypeError("dtypes must be a nonempty JSON array")
        dtypes = tuple(
            DTypeSpec.from_mapping(
                _require_mapping(item, f"dtypes[{index}]"), index=index
            )
            for index, item in enumerate(raw_dtypes)
        )
        if len({item.name for item in dtypes}) != len(dtypes):
            raise ValueError("dtypes must not contain duplicate names")

        raw_cases = value["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise TypeError("cases must be a nonempty JSON array")
        cases = tuple(
            BenchmarkCase.from_mapping(
                _require_mapping(item, f"cases[{index}]"), index=index
            )
            for index, item in enumerate(raw_cases)
        )
        if len({item.name for item in cases}) != len(cases):
            raise ValueError("cases must not contain duplicate names")

        normalize_embeddings = value["normalize_embeddings"]
        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a bool")
        triton_launch = TritonLaunchSpec.from_mapping(
            _require_mapping(value["triton_launch"], "triton_launch")
        )
        maximum_embedding_dimension = max(
            item.embedding_dimension for item in cases
        )
        if (
            triton_launch.block_embedding_dimension
            < maximum_embedding_dimension
        ):
            raise ValueError(
                "triton_launch.block_embedding_dimension must be at least "
                "the largest case embedding_dimension "
                f"({maximum_embedding_dimension})"
            )

        return cls(
            benchmark_id=_require_nonempty_string(
                value["benchmark_id"], "benchmark_id"
            ),
            seed=_require_int(value["seed"], "seed"),
            device_index=_require_int(device["index"], "device.index"),
            warmup_iterations=_require_int(
                timing["warmup_iterations"],
                "timing.warmup_iterations",
                minimum=1,
            ),
            measured_iterations=_require_int(
                timing["measured_iterations"],
                "timing.measured_iterations",
                minimum=2,
            ),
            trials=_require_int(
                timing["trials"], "timing.trials", minimum=2
            ),
            providers=providers,
            dtypes=dtypes,
            cases=cases,
            normalize_embeddings=normalize_embeddings,
            triton_launch=triton_launch,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "device": {"type": "cuda", "index": self.device_index},
            "timing": {
                "warmup_iterations": self.warmup_iterations,
                "measured_iterations": self.measured_iterations,
                "trials": self.trials,
            },
            "providers": list(self.providers),
            "dtypes": [item.to_dict() for item in self.dtypes],
            "cases": [item.to_dict() for item in self.cases],
            "normalize_embeddings": self.normalize_embeddings,
            "triton_launch": self.triton_launch.to_dict(),
        }


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


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load and strictly validate a benchmark JSON file."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    return BenchmarkConfig.from_mapping(_require_mapping(raw, "config"))


def canonical_config_fingerprint(config: BenchmarkConfig) -> str:
    """Hash the validated semantic config independent of JSON formatting."""

    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_manifest(
    paths: Sequence[Path], *, root: Path
) -> Tuple[Dict[str, Optional[str]], str]:
    """Record per-file digests and an aggregate that also tracks absence."""

    records: Dict[str, Optional[str]] = {}
    digest = sha256()
    relative_paths = sorted(path.relative_to(root) for path in paths)
    for relative_path in relative_paths:
        path = root / relative_path
        key = relative_path.as_posix()
        file_digest = sha256_file(path) if path.is_file() else None
        records[key] = file_digest
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            (file_digest if file_digest is not None else "<missing>").encode(
                "ascii"
            )
        )
        digest.update(b"\0")
    return records, digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("values must all be finite")
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def summarize_latencies(
    latency_ms: Sequence[float],
    *,
    documents_per_call: int,
    query_document_pairs_per_call: int,
) -> Dict[str, object]:
    """Summarize CUDA-event samples and derive two explicit throughputs."""

    if not latency_ms:
        raise ValueError("latency_ms must not be empty")
    documents = _require_int(
        documents_per_call, "documents_per_call", minimum=1
    )
    pairs = _require_int(
        query_document_pairs_per_call,
        "query_document_pairs_per_call",
        minimum=1,
    )
    samples = tuple(float(value) for value in latency_ms)
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("latency_ms values must be finite and positive")

    per_sample_seconds = tuple(value / 1000.0 for value in samples)
    total_seconds = sum(per_sample_seconds)
    document_rates = tuple(
        documents / seconds for seconds in per_sample_seconds
    )
    pair_rates = tuple(pairs / seconds for seconds in per_sample_seconds)
    return {
        "sample_count": len(samples),
        "latency_ms": {
            "minimum": min(samples),
            "mean": fmean(samples),
            "standard_deviation": (
                pstdev(samples) if len(samples) > 1 else 0.0
            ),
            "p50": percentile(samples, 0.50),
            "p95": percentile(samples, 0.95),
            "maximum": max(samples),
        },
        "documents_scored_per_query_per_second": {
            "aggregate": len(samples) * documents / total_seconds,
            "mean_instantaneous": fmean(document_rates),
            "p50_instantaneous": percentile(document_rates, 0.50),
        },
        "query_document_pairs_per_second": {
            "aggregate": len(samples) * pairs / total_seconds,
            "mean_instantaneous": fmean(pair_rates),
            "p50_instantaneous": percentile(pair_rates, 0.50),
        },
    }


__all__ = [
    "BenchmarkCase",
    "BenchmarkConfig",
    "CONFIG_SCHEMA_VERSION",
    "DTypeSpec",
    "SUPPORTED_DTYPES",
    "SUPPORTED_PROVIDERS",
    "SUPPORTED_TRITON_BLOCK_SIZES",
    "TritonLaunchSpec",
    "canonical_config_fingerprint",
    "load_benchmark_config",
    "percentile",
    "sha256_file",
    "source_manifest",
    "summarize_latencies",
]
