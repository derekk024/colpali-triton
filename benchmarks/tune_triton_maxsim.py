#!/usr/bin/env python3
"""Tune explicit Triton MaxSim launch candidates on a CUDA GPU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean, median
import subprocess
import sys
from time import perf_counter
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import Tensor

from colpali_triton import __version__
from colpali_triton.benchmarking import (
    BenchmarkCase,
    DTypeSpec,
    TritonLaunchSpec,
    sha256_file,
    source_manifest,
    summarize_latencies,
)
from colpali_triton.maxsim import maxsim_vectorized
try:
    from benchmarks.benchmark_maxsim import (
        _input_seed,
        _make_inputs,
        _nvidia_smi_snapshot,
        _preflight_output,
        _property,
        _torch_dtype,
        _write_json_atomic,
    )
except ModuleNotFoundError as error:
    if error.name not in ("benchmarks", "benchmarks.benchmark_maxsim"):
        raise
    from benchmark_maxsim import (  # type: ignore[no-redef]
        _input_seed,
        _make_inputs,
        _nvidia_smi_snapshot,
        _preflight_output,
        _property,
        _torch_dtype,
        _write_json_atomic,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "triton_tuning.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "phase11" / "triton_tuning.json"
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / ".source_commit",
    PROJECT_ROOT / "benchmarks" / "benchmark_maxsim.py",
    PROJECT_ROOT / "configs" / "triton_tuning.json",
    PROJECT_ROOT / "docker" / "Dockerfile.cuda",
    PROJECT_ROOT / "docker" / "requirements-cuda.txt",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "src" / "colpali_triton" / "__init__.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "benchmarking.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "maxsim.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "triton_maxsim.py",
)
PACKAGE_NAMES = (
    "colpali-triton",
    "numpy",
    "nvidia-ml-py",
    "torch",
    "triton",
)
SCHEMA_VERSION = 1
WORKLOAD_LATENCY_STATISTIC = "pooled_p50_cuda_event_ms"
AGGREGATE_METRIC = "median_workload_relative_p50"
TIE_BREAKERS = ("aggregate_cuda_event_latency_ms", "candidate_id")
Provider = Callable[..., Tensor]


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
    expected = set(required)
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{name} has invalid keys: {', '.join(details)}")


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be blank")
    return result


@dataclass(frozen=True)
class TuningCandidate:
    """One named, explicit ``TritonMaxSimConfig`` candidate."""

    candidate_id: str
    launch: TritonLaunchSpec

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, index: int
    ) -> "TuningCandidate":
        name = f"candidates[{index}]"
        _require_exact_keys(
            value,
            name=name,
            required=(
                "candidate_id",
                "block_query_tokens",
                "block_document_tokens",
                "block_embedding_dimension",
                "num_warps",
                "num_stages",
            ),
        )
        launch_mapping = {
            key: value[key]
            for key in (
                "block_query_tokens",
                "block_document_tokens",
                "block_embedding_dimension",
                "num_warps",
                "num_stages",
            )
        }
        return cls(
            candidate_id=_require_string(
                value["candidate_id"], f"{name}.candidate_id"
            ),
            launch=TritonLaunchSpec.from_mapping(launch_mapping),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            **self.launch.to_dict(),
        }


@dataclass(frozen=True)
class TuningConfig:
    """Strict schema for the canonical Phase 11 tuning experiment."""

    tuning_id: str
    seed: int
    device_index: int
    warmup_iterations: int
    measured_iterations: int
    trials: int
    dtypes: Tuple[DTypeSpec, ...]
    cases: Tuple[BenchmarkCase, ...]
    candidates: Tuple[TuningCandidate, ...]
    normalize_embeddings: bool

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "TuningConfig":
        _require_exact_keys(
            value,
            name="config",
            required=(
                "schema_version",
                "tuning_id",
                "seed",
                "device",
                "timing",
                "selection",
                "dtypes",
                "cases",
                "candidates",
                "normalize_embeddings",
            ),
        )
        version = _require_int(
            value["schema_version"], "schema_version", minimum=1
        )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version}; "
                f"expected {SCHEMA_VERSION}"
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

        selection = _require_mapping(value["selection"], "selection")
        _require_exact_keys(
            selection,
            name="selection",
            required=(
                "workload_latency_statistic",
                "aggregate_metric",
                "tie_breakers",
                "require_candidate_pass_all_workloads",
            ),
        )
        if (
            selection["workload_latency_statistic"]
            != WORKLOAD_LATENCY_STATISTIC
        ):
            raise ValueError(
                "selection.workload_latency_statistic must be "
                f"{WORKLOAD_LATENCY_STATISTIC!r}"
            )
        if selection["aggregate_metric"] != AGGREGATE_METRIC:
            raise ValueError(
                "selection.aggregate_metric must be "
                f"{AGGREGATE_METRIC!r}"
            )
        raw_tie_breakers = selection["tie_breakers"]
        if not isinstance(raw_tie_breakers, list):
            raise TypeError("selection.tie_breakers must be a JSON array")
        if tuple(raw_tie_breakers) != TIE_BREAKERS:
            raise ValueError(
                f"selection.tie_breakers must be {list(TIE_BREAKERS)!r}"
            )
        if selection["require_candidate_pass_all_workloads"] is not True:
            raise ValueError(
                "selection.require_candidate_pass_all_workloads must be true"
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

        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise TypeError("candidates must be a nonempty JSON array")
        candidates = tuple(
            TuningCandidate.from_mapping(
                _require_mapping(item, f"candidates[{index}]"), index=index
            )
            for index, item in enumerate(raw_candidates)
        )
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidates must not contain duplicate IDs")
        launch_tuples = tuple(
            tuple(item.launch.to_dict().values()) for item in candidates
        )
        if len(set(launch_tuples)) != len(launch_tuples):
            raise ValueError(
                "candidates must not contain duplicate launch configurations"
            )
        maximum_dimension = max(case.embedding_dimension for case in cases)
        for candidate in candidates:
            if (
                candidate.launch.block_embedding_dimension
                < maximum_dimension
            ):
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} cannot cover "
                    f"embedding dimension {maximum_dimension}"
                )

        normalize = value["normalize_embeddings"]
        if not isinstance(normalize, bool):
            raise TypeError("normalize_embeddings must be a bool")
        return cls(
            tuning_id=_require_string(value["tuning_id"], "tuning_id"),
            seed=_require_int(value["seed"], "seed"),
            device_index=_require_int(
                device["index"], "device.index", minimum=0
            ),
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
            dtypes=dtypes,
            cases=cases,
            candidates=candidates,
            normalize_embeddings=normalize,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tuning_id": self.tuning_id,
            "seed": self.seed,
            "device": {"type": "cuda", "index": self.device_index},
            "timing": {
                "warmup_iterations": self.warmup_iterations,
                "measured_iterations": self.measured_iterations,
                "trials": self.trials,
            },
            "selection": {
                "workload_latency_statistic": (
                    WORKLOAD_LATENCY_STATISTIC
                ),
                "aggregate_metric": AGGREGATE_METRIC,
                "tie_breakers": list(TIE_BREAKERS),
                "require_candidate_pass_all_workloads": True,
            },
            "dtypes": [item.to_dict() for item in self.dtypes],
            "cases": [item.to_dict() for item in self.cases],
            "candidates": [item.to_dict() for item in self.candidates],
            "normalize_embeddings": self.normalize_embeddings,
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


def load_tuning_config(path: Path) -> TuningConfig:
    """Load and strictly validate a tuning JSON file."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    return TuningConfig.from_mapping(_require_mapping(raw, "config"))


def semantic_config_fingerprint(config: TuningConfig) -> str:
    """Hash the validated config independent of JSON whitespace."""

    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate",
        action="append",
        help="Run only this configured candidate ID (repeatable).",
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("float16", "bfloat16"),
        help="Run only this configured dtype (repeatable).",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this configured case name (repeatable).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Print readiness without compiling a kernel or writing an "
            "artifact."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output artifact.",
    )
    return parser.parse_args(argv)


def _select_names(
    requested: Optional[Sequence[str]],
    configured: Sequence[str],
    *,
    kind: str,
) -> Tuple[str, ...]:
    if requested is None:
        return tuple(configured)
    selected = tuple(requested)
    if len(set(selected)) != len(selected):
        raise ValueError(f"{kind} selection contains duplicates")
    missing = sorted(set(selected) - set(configured))
    if missing:
        raise ValueError(
            f"{kind} selection is not present in config: {missing}"
        )
    return selected


def _probe_triton() -> Dict[str, object]:
    entrypoint = "colpali_triton.triton_maxsim.maxsim_triton"
    try:
        module = import_module("colpali_triton.triton_maxsim")
        provider = getattr(module, "maxsim_triton")
        config_type = getattr(module, "TritonMaxSimConfig")
        availability = getattr(module, "triton_is_available")
        if not callable(provider) or not callable(config_type):
            raise TypeError("Triton MaxSim entry points are not callable")
        if not callable(availability) or not availability():
            raise RuntimeError("the Triton Python package is unavailable")
    except Exception as error:
        return {
            "available": False,
            "entrypoint": entrypoint,
            "error": f"{type(error).__name__}: {error}",
        }
    return {"available": True, "entrypoint": entrypoint, "error": None}


def _validate_runtime_candidate_configs(
    candidates: Sequence[TuningCandidate],
) -> Dict[str, object]:
    """Instantiate every candidate with the production config class."""

    try:
        module = import_module("colpali_triton.triton_maxsim")
        config_type = getattr(module, "TritonMaxSimConfig")
        for candidate in candidates:
            config_type(**candidate.launch.to_dict())
    except Exception as error:
        return {
            "valid": False,
            "validated_candidate_count": 0,
            "class": (
                "colpali_triton.triton_maxsim.TritonMaxSimConfig"
            ),
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "valid": True,
        "validated_candidate_count": len(candidates),
        "class": "colpali_triton.triton_maxsim.TritonMaxSimConfig",
        "error": None,
    }


def _preflight(
    config: TuningConfig, *, candidates: Sequence[TuningCandidate]
) -> Dict[str, object]:
    cuda_built = torch.backends.cuda.is_built()
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_in_range = cuda_available and config.device_index < device_count
    triton_probe = _probe_triton()
    candidate_validation = _validate_runtime_candidate_configs(candidates)
    blockers = []
    if not cuda_built:
        blockers.append("the installed PyTorch build does not include CUDA")
    elif not cuda_available:
        blockers.append("PyTorch cannot access a CUDA device")
    elif not device_in_range:
        blockers.append(
            f"CUDA device index {config.device_index} is unavailable "
            f"(device_count={device_count})"
        )
    if not triton_probe["available"]:
        blockers.append(
            f"Triton provider unavailable: {triton_probe['error']}"
        )
    if not candidate_validation["valid"]:
        blockers.append(
            "candidate configuration validation failed: "
            f"{candidate_validation['error']}"
        )
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "cuda": {
            "pytorch_build_includes_cuda": cuda_built,
            "available": cuda_available,
            "device_count": device_count,
            "configured_device_index": config.device_index,
            "configured_device_in_range": device_in_range,
        },
        "triton": triton_probe,
        "candidate_count": len(candidates),
        "candidate_validation": candidate_validation,
    }


def _require_ready(preflight: Mapping[str, object]) -> None:
    if preflight["status"] == "ready":
        return
    blockers = preflight["blockers"]
    if not isinstance(blockers, list):
        raise RuntimeError("invalid preflight result")
    raise RuntimeError(
        "tuning preflight failed: "
        + "; ".join(str(item) for item in blockers)
    )


def _load_provider(candidate: TuningCandidate) -> Provider:
    module = import_module("colpali_triton.triton_maxsim")
    provider = getattr(module, "maxsim_triton")
    config_type = getattr(module, "TritonMaxSimConfig")
    launch_config = config_type(**candidate.launch.to_dict())

    def configured_provider(
        queries: Tensor,
        documents: Tensor,
        *,
        query_mask: Tensor,
        document_mask: Tensor,
    ) -> Tensor:
        return provider(
            queries,
            documents,
            query_mask=query_mask,
            document_mask=document_mask,
            config=launch_config,
        )

    return configured_provider


def _call_provider(
    provider: Provider,
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
) -> Tensor:
    return provider(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )


def _agreement_record(
    actual: Tensor,
    expected: Tensor,
    *,
    dtype_spec: DTypeSpec,
) -> Dict[str, object]:
    shape_matches = tuple(actual.shape) == tuple(expected.shape)
    if not shape_matches:
        return {
            "reference_provider": "torch",
            "shape_matches": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "finite": False,
            "agreement": False,
            "absolute_tolerance": dtype_spec.absolute_tolerance,
            "relative_tolerance": dtype_spec.relative_tolerance,
            "maximum_absolute_error": None,
            "maximum_relative_error": None,
        }
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    finite = bool(torch.isfinite(actual_float).all().item())
    difference = (actual_float - expected_float).abs()
    relative = difference / expected_float.abs().clamp_min(1e-12)
    raw_maximum_absolute_error = float(difference.max().item())
    raw_maximum_relative_error = float(relative.max().item())
    maximum_absolute_error = (
        raw_maximum_absolute_error
        if math.isfinite(raw_maximum_absolute_error)
        else None
    )
    maximum_relative_error = (
        raw_maximum_relative_error
        if math.isfinite(raw_maximum_relative_error)
        else None
    )
    agreement = finite and bool(
        torch.allclose(
            actual_float,
            expected_float,
            atol=dtype_spec.absolute_tolerance,
            rtol=dtype_spec.relative_tolerance,
        )
    )
    return {
        "reference_provider": "torch",
        "shape_matches": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "finite": finite,
        "agreement": agreement,
        "absolute_tolerance": dtype_spec.absolute_tolerance,
        "relative_tolerance": dtype_spec.relative_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def _failure_cleanup_synchronization(
    device: torch.device,
) -> Optional[str]:
    """Best-effort synchronization without replacing a workload failure."""

    try:
        torch.cuda.synchronize(device)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _numerical_gate(
    provider: Provider,
    *,
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    reference: Tensor,
    dtype_spec: DTypeSpec,
    device: torch.device,
) -> Dict[str, object]:
    torch.cuda.synchronize(device)
    started = perf_counter()
    try:
        actual = _call_provider(
            provider,
            queries,
            documents,
            query_mask,
            document_mask,
        )
        torch.cuda.synchronize(device)
        wall_ms = (perf_counter() - started) * 1000.0
        agreement = _agreement_record(
            actual, reference, dtype_spec=dtype_spec
        )
        del actual
    except Exception as error:
        failed_wall_ms = (perf_counter() - started) * 1000.0
        cleanup_error = _failure_cleanup_synchronization(device)
        return {
            "status": "execution_failed",
            "cold_compile_and_call_wall_ms": failed_wall_ms,
            "agreement": None,
            "error": f"{type(error).__name__}: {error}",
            "cleanup_synchronization_error": cleanup_error,
        }
    return {
        "status": (
            "passed" if agreement["agreement"] else "numerical_rejected"
        ),
        "cold_compile_and_call_wall_ms": wall_ms,
        "agreement": agreement,
        "error": None,
        "cleanup_synchronization_error": None,
    }


def _warm_candidate(
    provider: Provider,
    *,
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    iterations: int,
    device: torch.device,
) -> Dict[str, object]:
    started = perf_counter()
    try:
        for _ in range(iterations):
            output = _call_provider(
                provider,
                queries,
                documents,
                query_mask,
                document_mask,
            )
        torch.cuda.synchronize(device)
        wall_ms = (perf_counter() - started) * 1000.0
        del output
        return {
            "status": "completed",
            "iterations": iterations,
            "total_wall_ms": wall_ms,
            "error": None,
            "cleanup_synchronization_error": None,
        }
    except Exception as error:
        failed_wall_ms = (perf_counter() - started) * 1000.0
        cleanup_error = _failure_cleanup_synchronization(device)
        return {
            "status": "failed",
            "iterations": iterations,
            "total_wall_ms": failed_wall_ms,
            "error": f"{type(error).__name__}: {error}",
            "cleanup_synchronization_error": cleanup_error,
        }


def _time_trial(
    provider: Provider,
    *,
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    measured_iterations: int,
    device: torch.device,
    trial_index: int,
) -> Dict[str, object]:
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(measured_iterations)
    ]
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    host_started = perf_counter()
    try:
        for start_event, end_event in events:
            start_event.record()
            output = _call_provider(
                provider,
                queries,
                documents,
                query_mask,
                document_mask,
            )
            end_event.record()
        torch.cuda.synchronize(device)
        host_wall_ms = (perf_counter() - host_started) * 1000.0
        samples = [
            float(start.elapsed_time(end)) for start, end in events
        ]
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        del output
        return {
            "status": "completed",
            "trial_index": trial_index,
            "cuda_event_latency_ms": samples,
            "host_wall_total_ms": host_wall_ms,
            "host_wall_per_call_ms": host_wall_ms / measured_iterations,
            "memory": {
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "incremental_peak_allocated_bytes": max(
                    0, peak_allocated - baseline_allocated
                ),
                "incremental_peak_reserved_bytes": max(
                    0, peak_reserved - baseline_reserved
                ),
            },
            "error": None,
            "cleanup_synchronization_error": None,
        }
    except Exception as error:
        failed_wall_ms = (perf_counter() - host_started) * 1000.0
        cleanup_error = _failure_cleanup_synchronization(device)
        return {
            "status": "failed",
            "trial_index": trial_index,
            "cuda_event_latency_ms": [],
            "host_wall_total_ms": failed_wall_ms,
            "host_wall_per_call_ms": None,
            "memory": None,
            "error": f"{type(error).__name__}: {error}",
            "cleanup_synchronization_error": cleanup_error,
        }


def _trial_order(
    candidate_ids: Sequence[str],
    *,
    seed: int,
    workload_key: str,
    trial_index: int,
) -> Tuple[str, ...]:
    def ordering_key(candidate_id: str) -> bytes:
        return sha256(
            (
                f"{seed}\0{workload_key}\0{trial_index}\0"
                f"{candidate_id}"
            ).encode("utf-8")
        ).digest()

    return tuple(sorted(candidate_ids, key=ordering_key))


def _finalize_candidate_timing(
    record: Dict[str, object],
    *,
    case: BenchmarkCase,
    measured_iterations: int,
    trials: int,
) -> None:
    raw_trials = record["trial_records"]
    if not isinstance(raw_trials, list):
        raise RuntimeError("invalid candidate trial records")
    completed_trials = [
        item
        for item in raw_trials
        if isinstance(item, dict) and item["status"] == "completed"
    ]
    if len(completed_trials) != trials:
        record["status"] = "timing_failed"
        record["timing"] = {
            "method": "torch.cuda.Event(enable_timing=True)",
            "measured_iterations_per_trial": measured_iterations,
            "trial_count_requested": trials,
            "trial_count_completed": len(completed_trials),
            "trials": raw_trials,
            "summary": None,
        }
        del record["trial_records"]
        return
    samples = [
        float(value)
        for item in completed_trials
        for value in item["cuda_event_latency_ms"]
    ]
    memory = [
        item["memory"]
        for item in completed_trials
        if isinstance(item["memory"], dict)
    ]
    record["status"] = "completed"
    record["timing"] = {
        "method": "torch.cuda.Event(enable_timing=True)",
        "synchronization": (
            "one torch.cuda.synchronize after each candidate trial"
        ),
        "trial_interleaving": (
            "candidate order is deterministic and reshuffled per trial"
        ),
        "measured_iterations_per_trial": measured_iterations,
        "trial_count_requested": trials,
        "trial_count_completed": len(completed_trials),
        "total_cuda_event_latency_ms": sum(samples),
        "summary": summarize_latencies(
            samples,
            documents_per_call=case.document_batch_size,
            query_document_pairs_per_call=case.query_document_pairs,
        ),
        "trials": raw_trials,
        "memory": {
            "maximum_peak_allocated_bytes": max(
                int(item["peak_allocated_bytes"]) for item in memory
            ),
            "maximum_peak_reserved_bytes": max(
                int(item["peak_reserved_bytes"]) for item in memory
            ),
            "maximum_incremental_peak_allocated_bytes": max(
                int(item["incremental_peak_allocated_bytes"])
                for item in memory
            ),
            "maximum_incremental_peak_reserved_bytes": max(
                int(item["incremental_peak_reserved_bytes"])
                for item in memory
            ),
        },
    }
    del record["trial_records"]


def _candidate_latency(record: Mapping[str, object]) -> Optional[float]:
    if record.get("status") != "completed":
        return None
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    summary = timing.get("summary")
    if not isinstance(summary, dict):
        return None
    latency = summary.get("latency_ms")
    if not isinstance(latency, dict):
        return None
    value = latency.get("p50")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _candidate_total_latency(
    record: Mapping[str, object],
) -> Optional[float]:
    if record.get("status") != "completed":
        return None
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    value = timing.get("total_cuda_event_latency_ms")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _rank_candidates(
    results: Sequence[Mapping[str, object]],
    *,
    candidate_ids: Sequence[str],
    dtype_name: Optional[str] = None,
) -> Dict[str, object]:
    selected_results = [
        result
        for result in results
        if dtype_name is None or result["dtype"] == dtype_name
    ]
    workload_records = []
    relative_by_candidate: Dict[str, List[float]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    total_by_candidate: Dict[str, float] = {
        candidate_id: 0.0 for candidate_id in candidate_ids
    }
    completed_by_candidate: Dict[str, int] = {
        candidate_id: 0 for candidate_id in candidate_ids
    }
    for result in selected_results:
        raw_candidates = result["candidates"]
        if not isinstance(raw_candidates, list):
            raise RuntimeError("invalid workload candidate records")
        records = {
            str(item["candidate_id"]): item
            for item in raw_candidates
            if isinstance(item, dict)
        }
        latencies = {
            candidate_id: latency
            for candidate_id in candidate_ids
            if (
                (latency := _candidate_latency(records[candidate_id]))
                is not None
            )
        }
        if not latencies:
            workload_records.append(
                {
                    "case": result["case"]["name"],
                    "dtype": result["dtype"],
                    "status": "no_eligible_candidate",
                    "winner_candidate_id": None,
                    "winner_p50_cuda_event_ms": None,
                }
            )
            continue
        winner_id, winner_latency = min(
            latencies.items(), key=lambda item: (item[1], item[0])
        )
        workload_records.append(
            {
                "case": result["case"]["name"],
                "dtype": result["dtype"],
                "status": "selected",
                "winner_candidate_id": winner_id,
                "winner_p50_cuda_event_ms": winner_latency,
            }
        )
        for candidate_id, latency in latencies.items():
            relative_by_candidate[candidate_id].append(
                latency / winner_latency
            )
            total = _candidate_total_latency(records[candidate_id])
            if total is None:
                raise RuntimeError("completed candidate lacks total latency")
            total_by_candidate[candidate_id] += total
            completed_by_candidate[candidate_id] += 1

    expected_workloads = len(selected_results)
    ranking = []
    for candidate_id in candidate_ids:
        relative = relative_by_candidate[candidate_id]
        eligible = (
            expected_workloads > 0
            and completed_by_candidate[candidate_id] == expected_workloads
        )
        ranking.append(
            {
                "candidate_id": candidate_id,
                "eligible": eligible,
                "completed_workloads": completed_by_candidate[candidate_id],
                "required_workloads": expected_workloads,
                "relative_p50_by_workload": relative,
                "median_workload_relative_p50": (
                    median(relative) if relative else None
                ),
                "mean_workload_relative_p50": (
                    fmean(relative) if relative else None
                ),
                "aggregate_cuda_event_latency_ms": (
                    total_by_candidate[candidate_id]
                    if completed_by_candidate[candidate_id]
                    else None
                ),
            }
        )
    eligible_ranking = [
        item for item in ranking if bool(item["eligible"])
    ]
    eligible_ranking.sort(
        key=lambda item: (
            float(item["median_workload_relative_p50"]),
            float(item["aggregate_cuda_event_latency_ms"]),
            str(item["candidate_id"]),
        )
    )
    ineligible_ranking = [
        item for item in ranking if not bool(item["eligible"])
    ]
    ineligible_ranking.sort(key=lambda item: str(item["candidate_id"]))
    ordered_ranking = eligible_ranking + ineligible_ranking
    winner = eligible_ranking[0] if eligible_ranking else None
    return {
        "status": (
            "selected" if winner is not None else "no_eligible_candidate"
        ),
        "scope_dtype": dtype_name,
        "workload_count": expected_workloads,
        "workload_winners": workload_records,
        "winner_candidate_id": (
            winner["candidate_id"] if winner is not None else None
        ),
        "winner_metrics": (
            {
                "median_workload_relative_p50": winner[
                    "median_workload_relative_p50"
                ],
                "aggregate_cuda_event_latency_ms": winner[
                    "aggregate_cuda_event_latency_ms"
                ],
            }
            if winner is not None
            else None
        ),
        "ranking": ordered_ranking,
    }


def _git_state() -> Dict[str, object]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "status_entry_count": None,
            "status_sha256": None,
            "error": f"{type(error).__name__}: {error}",
        }
    status_lines = [
        line for line in status.stdout.splitlines() if line
    ]
    return {
        "available": revision.returncode == 0 and status.returncode == 0,
        "commit": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "status_sha256": (
            sha256(status.stdout.encode("utf-8")).hexdigest()
            if status.returncode == 0
            else None
        ),
        "error": None,
    }


def _validated_source_commit(
    value: object, *, source: str
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"{source} source commit must be a string")
    commit = value.strip()
    if commit in ("", "unknown"):
        return None
    if (
        commit != commit.lower()
        or len(commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in commit
        )
    ):
        raise RuntimeError(
            f"{source} source commit is not a full lowercase Git SHA: "
            f"{value!r}"
        )
    return commit


def _source_commit_provenance(
    git_state: Mapping[str, object],
) -> Dict[str, object]:
    marker_path = PROJECT_ROOT / ".source_commit"
    marker_raw = (
        marker_path.read_text(encoding="utf-8")
        if marker_path.is_file()
        else None
    )
    values = {
        "git": _validated_source_commit(
            git_state.get("commit"), source="Git"
        ),
        "environment": _validated_source_commit(
            os.environ.get("COLPALI_SOURCE_COMMIT"),
            source="COLPALI_SOURCE_COMMIT",
        ),
        "marker_file": _validated_source_commit(
            marker_raw, source=".source_commit"
        ),
    }
    distinct = {value for value in values.values() if value is not None}
    if len(distinct) > 1:
        raise RuntimeError(
            "source commit provenance disagrees across Git, "
            "COLPALI_SOURCE_COMMIT, and .source_commit"
        )
    if not distinct:
        raise RuntimeError(
            "an exact source commit is required from Git, "
            "COLPALI_SOURCE_COMMIT, or .source_commit"
        )
    return {
        "effective_commit": next(iter(distinct), None),
        "consistent": True,
        "sources": values,
        "marker_path": str(marker_path),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _environment(
    device: torch.device, config: TuningConfig
) -> Dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "captured_at_utc": _utc_now(),
        "colpali_triton_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            name: _package_version(name) for name in PACKAGE_NAMES
        },
        "pytorch": {
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "cuda_build_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_arch_list": torch.cuda.get_arch_list(),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
            ),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "cuda_device": {
            "index": device.index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "free_memory_bytes": int(free_bytes),
            "memory_total_bytes": int(total_bytes),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "multi_processor_count": _property(
                properties, "multi_processor_count"
            ),
            "uuid": _property(properties, "uuid"),
            "pci_bus_id": _property(properties, "pci_bus_id"),
            "l2_cache_size_bytes": _property(
                properties, "L2_cache_size"
            ),
            "shared_memory_per_multiprocessor_bytes": _property(
                properties, "shared_memory_per_multiprocessor"
            ),
            "warp_size": _property(properties, "warp_size"),
        },
        "nvidia_smi": _nvidia_smi_snapshot(),
        "candidate_count": len(config.candidates),
        "relevant_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "COLPALI_SOURCE_COMMIT",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_CUDA_ARCH_LIST",
                "TRITON_CACHE_DIR",
            )
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _case_record(case: BenchmarkCase) -> Dict[str, object]:
    record = case.to_dict()
    record["query_document_pairs"] = case.query_document_pairs
    return record


def _storage_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def run_tuning(
    config: TuningConfig,
    *,
    candidate_ids: Sequence[str],
    dtype_names: Sequence[str],
    case_names: Sequence[str],
    config_path: Path,
    argv: Sequence[str],
) -> Dict[str, object]:
    """Execute correctness-gated, interleaved candidate measurements."""

    candidate_by_id = {
        item.candidate_id: item for item in config.candidates
    }
    selected_candidates = tuple(
        candidate_by_id[item] for item in candidate_ids
    )
    preflight = _preflight(config, candidates=selected_candidates)
    _require_ready(preflight)
    device = torch.device("cuda", config.device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    environment_at_start = _environment(device, config)

    dtype_by_name = {item.name: item for item in config.dtypes}
    case_by_name = {item.name: item for item in config.cases}
    providers = {
        item.candidate_id: _load_provider(item)
        for item in selected_candidates
    }
    source_files, source_fingerprint = source_manifest(
        SOURCE_FILES, root=PROJECT_ROOT
    )
    config_file_digest = sha256_file(config_path)
    combined_fingerprint = sha256(
        (source_fingerprint + "\0" + config_file_digest).encode("ascii")
    ).hexdigest()
    git_state = _git_state()
    source_commit = _source_commit_provenance(git_state)
    started_at = _utc_now()
    wall_started = perf_counter()
    results: List[Dict[str, object]] = []

    with torch.inference_mode():
        for dtype_name in dtype_names:
            dtype_spec = dtype_by_name[dtype_name]
            dtype = _torch_dtype(dtype_name)
            for case_name in case_names:
                case = case_by_name[case_name]
                seed = _input_seed(config.seed, case.name, dtype_name)
                (
                    queries,
                    documents,
                    query_mask,
                    document_mask,
                ) = _make_inputs(
                    case,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    normalize_embeddings=config.normalize_embeddings,
                )
                torch.cuda.synchronize(device)
                reference = maxsim_vectorized(
                    queries,
                    documents,
                    query_mask=query_mask,
                    document_mask=document_mask,
                )
                torch.cuda.synchronize(device)

                candidate_records: Dict[str, Dict[str, object]] = {}
                eligible_ids = []
                for candidate in selected_candidates:
                    candidate_id = candidate.candidate_id
                    numerical_gate = _numerical_gate(
                        providers[candidate_id],
                        queries=queries,
                        documents=documents,
                        query_mask=query_mask,
                        document_mask=document_mask,
                        reference=reference,
                        dtype_spec=dtype_spec,
                        device=device,
                    )
                    record: Dict[str, object] = {
                        "candidate_id": candidate_id,
                        "launch_configuration": (
                            candidate.launch.to_dict()
                        ),
                        "status": numerical_gate["status"],
                        "numerical_gate": numerical_gate,
                        "warmup": None,
                        "timing": None,
                        "trial_records": [],
                    }
                    candidate_records[candidate_id] = record
                    if numerical_gate["status"] == "passed":
                        warmup = _warm_candidate(
                            providers[candidate_id],
                            queries=queries,
                            documents=documents,
                            query_mask=query_mask,
                            document_mask=document_mask,
                            iterations=config.warmup_iterations,
                            device=device,
                        )
                        record["warmup"] = warmup
                        if warmup["status"] == "completed":
                            eligible_ids.append(candidate_id)
                            record["status"] = "timing_pending"
                        else:
                            record["status"] = "warmup_failed"

                workload_key = f"{dtype_name}\0{case.name}"
                trial_execution_orders = []
                for trial_index in range(config.trials):
                    active_ids = [
                        candidate_id
                        for candidate_id in eligible_ids
                        if candidate_records[candidate_id]["status"]
                        != "timing_failed"
                    ]
                    order = _trial_order(
                        active_ids,
                        seed=config.seed,
                        workload_key=workload_key,
                        trial_index=trial_index,
                    )
                    trial_execution_orders.append(
                        {
                            "trial_index": trial_index,
                            "candidate_ids": list(order),
                        }
                    )
                    for candidate_id in order:
                        trial = _time_trial(
                            providers[candidate_id],
                            queries=queries,
                            documents=documents,
                            query_mask=query_mask,
                            document_mask=document_mask,
                            measured_iterations=(
                                config.measured_iterations
                            ),
                            device=device,
                            trial_index=trial_index,
                        )
                        raw_trials = candidate_records[candidate_id][
                            "trial_records"
                        ]
                        if not isinstance(raw_trials, list):
                            raise RuntimeError(
                                "invalid internal trial records"
                            )
                        raw_trials.append(trial)
                        if trial["status"] != "completed":
                            candidate_records[candidate_id][
                                "status"
                            ] = "timing_failed"

                for candidate_id in eligible_ids:
                    _finalize_candidate_timing(
                        candidate_records[candidate_id],
                        case=case,
                        measured_iterations=config.measured_iterations,
                        trials=config.trials,
                    )
                for record in candidate_records.values():
                    if "trial_records" in record:
                        del record["trial_records"]

                results.append(
                    {
                        "case": _case_record(case),
                        "dtype": dtype_name,
                        "input_seed": seed,
                        "input_storage_bytes": {
                            "query_embeddings": _storage_bytes(queries),
                            "document_embeddings": _storage_bytes(documents),
                            "query_mask": _storage_bytes(query_mask),
                            "document_mask": _storage_bytes(document_mask),
                            "total": sum(
                                _storage_bytes(tensor)
                                for tensor in (
                                    queries,
                                    documents,
                                    query_mask,
                                    document_mask,
                                )
                            ),
                        },
                        "reference": {
                            "provider": "torch",
                            "entrypoint": (
                                "colpali_triton.maxsim.maxsim_vectorized"
                            ),
                            "shape": list(reference.shape),
                            "dtype": str(reference.dtype),
                        },
                        "trial_execution_orders": (
                            trial_execution_orders
                        ),
                        "candidates": [
                            candidate_records[candidate_id]
                            for candidate_id in candidate_ids
                        ],
                    }
                )
                del (
                    reference,
                    query_mask,
                    document_mask,
                    queries,
                    documents,
                )
                torch.cuda.empty_cache()

    overall_selection = _rank_candidates(
        results, candidate_ids=candidate_ids
    )
    dtype_selections = {
        dtype_name: _rank_candidates(
            results,
            candidate_ids=candidate_ids,
            dtype_name=dtype_name,
        )
        for dtype_name in dtype_names
    }
    winner_id = overall_selection["winner_candidate_id"]
    winner_config = (
        candidate_by_id[str(winner_id)].to_dict()
        if winner_id is not None
        else None
    )
    canonical = (
        tuple(candidate_ids)
        == tuple(item.candidate_id for item in config.candidates)
        and tuple(dtype_names)
        == tuple(item.name for item in config.dtypes)
        and tuple(case_names)
        == tuple(item.name for item in config.cases)
    )
    status = (
        "completed"
        if winner_id is not None
        else "completed_without_eligible_candidate"
    )
    environment_at_end = _environment(device, config)
    completed_at = _utc_now()
    wall_duration_seconds = perf_counter() - wall_started
    return {
        "schema_version": 1,
        "kind": "triton_maxsim_tuning",
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "wall_duration_seconds": wall_duration_seconds,
        "selection": {
            "canonical_full_matrix": canonical,
            "candidates": list(candidate_ids),
            "dtypes": list(dtype_names),
            "cases": list(case_names),
        },
        "invocation": {
            "argv": list(argv),
            "config_path": str(config_path.resolve()),
        },
        "provenance": {
            "config_file_path": str(config_path.resolve()),
            "config_file_sha256": config_file_digest,
            "semantic_config_sha256": (
                semantic_config_fingerprint(config)
            ),
            "source_sha256": source_files,
            "source_fingerprint_sha256": source_fingerprint,
            "source_and_config_fingerprint_sha256": combined_fingerprint,
            "git": git_state,
            "source_commit": source_commit,
        },
        "protocol": {
            "config": config.to_dict(),
            "inference_mode": True,
            "input_generation_dtype": "float32",
            "input_normalization_before_cast": (
                config.normalize_embeddings
            ),
            "float32_matmul_precision": "highest",
            "allow_tf32": False,
            "cudnn_benchmark": False,
            "correctness_checked_before_any_timing": True,
            "numerically_rejected_candidates_are_not_timed": True,
            "cold_compile_call_is_not_timed_by_cuda_events": True,
            "warmups_are_not_measured": True,
            "trial_candidate_order": (
                "deterministic SHA-256 ordering by seed, workload, "
                "trial, and candidate ID"
            ),
            "winner_selection": {
                "workload_latency_statistic": (
                    WORKLOAD_LATENCY_STATISTIC
                ),
                "aggregate_metric": AGGREGATE_METRIC,
                "tie_breakers": list(TIE_BREAKERS),
                "require_candidate_pass_all_workloads": True,
            },
        },
        "preflight": preflight,
        "environment": {
            "start": environment_at_start,
            "end": environment_at_end,
        },
        "winner": {
            "candidate_id": winner_id,
            "configuration": winner_config,
            "overall": overall_selection,
            "by_dtype": dtype_selections,
        },
        "results": results,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(argv)
    try:
        config = load_tuning_config(parsed.config)
        candidate_ids = _select_names(
            parsed.candidate,
            tuple(item.candidate_id for item in config.candidates),
            kind="candidate",
        )
        dtype_names = _select_names(
            parsed.dtype,
            tuple(item.name for item in config.dtypes),
            kind="dtype",
        )
        case_names = _select_names(
            parsed.case,
            tuple(item.name for item in config.cases),
            kind="case",
        )
        selected_candidates = tuple(
            item
            for item in config.candidates
            if item.candidate_id in candidate_ids
        )
        preflight = _preflight(
            config, candidates=selected_candidates
        )
        if parsed.preflight:
            print(
                json.dumps(
                    {
                        "tuning_id": config.tuning_id,
                        "config_file_sha256": sha256_file(parsed.config),
                        "semantic_config_sha256": (
                            semantic_config_fingerprint(config)
                        ),
                        "selection": {
                            "candidates": list(candidate_ids),
                            "dtypes": list(dtype_names),
                            "cases": list(case_names),
                        },
                        "preflight": preflight,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0

        _require_ready(preflight)
        _preflight_output(parsed.output, parsed.overwrite)
        command_argv = tuple(sys.argv if argv is None else argv)
        result = run_tuning(
            config,
            candidate_ids=candidate_ids,
            dtype_names=dtype_names,
            case_names=case_names,
            config_path=parsed.config,
            argv=command_argv,
        )
        _write_json_atomic(
            parsed.output, result, overwrite=parsed.overwrite
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(parsed.output.resolve()),
                    "output_sha256": sha256_file(parsed.output),
                    "winner_candidate_id": result["winner"][
                        "candidate_id"
                    ],
                    "workload_count": len(result["results"]),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0 if result["status"] == "completed" else 3
    except (
        FileExistsError,
        IsADirectoryError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
