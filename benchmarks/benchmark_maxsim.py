#!/usr/bin/env python3
"""Benchmark PyTorch and Triton MaxSim providers on a CUDA GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import subprocess
import sys
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch import Tensor
from torch.nn import functional as F

from colpali_triton import __version__
from colpali_triton.benchmarking import (
    BenchmarkCase,
    BenchmarkConfig,
    DTypeSpec,
    canonical_config_fingerprint,
    load_benchmark_config,
    sha256_file,
    source_manifest,
    summarize_latencies,
)
from colpali_triton.maxsim import maxsim_vectorized


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "maxsim_benchmark.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "phase10" / "maxsim_cuda_benchmark.json"
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "src" / "colpali_triton" / "__init__.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "benchmarking.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "maxsim.py",
    # Absence is fingerprinted before Phase 11; adding the kernel changes it.
    PROJECT_ROOT / "src" / "colpali_triton" / "triton_maxsim.py",
)
PACKAGE_NAMES = (
    "colpali-triton",
    "numpy",
    "nvidia-ml-py",
    "torch",
    "triton",
)
Provider = Callable[..., Tensor]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--provider",
        action="append",
        choices=("torch", "triton"),
        help=(
            "Run only this provider (repeatable). Omitting this flag uses "
            "every provider in the config."
        ),
    )
    parser.add_argument(
        "--dtype",
        action="append",
        choices=("float16", "bfloat16"),
        help=(
            "Run only this dtype (repeatable). Omitting this flag uses every "
            "dtype in the config."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "Run only this configured case name (repeatable). Omitting this "
            "flag uses every case."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Print readiness and provider availability without running CUDA.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output record.",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_state() -> Dict[str, object]:
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
    status_lines = tuple(
        line for line in status.stdout.splitlines() if line
    )
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
    }


def _probe_provider(name: str) -> Dict[str, object]:
    if name == "torch":
        return {
            "available": True,
            "entrypoint": "colpali_triton.maxsim.maxsim_vectorized",
            "error": None,
        }
    if name != "triton":
        return {
            "available": False,
            "entrypoint": None,
            "error": f"unsupported provider {name!r}",
        }

    entrypoint = "colpali_triton.triton_maxsim.maxsim_triton"
    try:
        module = import_module("colpali_triton.triton_maxsim")
        provider = getattr(module, "maxsim_triton")
        if not callable(provider):
            raise TypeError(f"{entrypoint} is not callable")
        config_type = getattr(module, "TritonMaxSimConfig")
        if not callable(config_type):
            raise TypeError(
                "colpali_triton.triton_maxsim.TritonMaxSimConfig "
                "is not callable"
            )
        availability = getattr(module, "triton_is_available", None)
        if callable(availability) and not availability():
            raise RuntimeError(
                "the Triton Python package could not be imported"
            )
    except Exception as error:
        return {
            "available": False,
            "entrypoint": entrypoint,
            "error": f"{type(error).__name__}: {error}",
        }
    return {"available": True, "entrypoint": entrypoint, "error": None}


def _load_provider(name: str, config: BenchmarkConfig) -> Provider:
    if name == "torch":
        return maxsim_vectorized
    if name == "triton":
        module = import_module("colpali_triton.triton_maxsim")
        provider = getattr(module, "maxsim_triton")
        if not callable(provider):
            raise TypeError(
                "colpali_triton.triton_maxsim.maxsim_triton is not callable"
            )
        config_type = getattr(module, "TritonMaxSimConfig")
        launch_config = config_type(**config.triton_launch.to_dict())

        def configured_provider(
            query_embeddings: Tensor,
            document_embeddings: Tensor,
            *,
            query_mask: Optional[Tensor] = None,
            document_mask: Optional[Tensor] = None,
        ) -> Tensor:
            return provider(
                query_embeddings,
                document_embeddings,
                query_mask=query_mask,
                document_mask=document_mask,
                config=launch_config,
            )

        return configured_provider
    raise ValueError(f"unsupported provider {name!r}")


def _select_names(
    requested: Optional[Sequence[str]],
    configured: Sequence[str],
    *,
    kind: str,
) -> Tuple[str, ...]:
    if requested is None:
        return tuple(configured)
    result = tuple(requested)
    if len(set(result)) != len(result):
        raise ValueError(f"{kind} overrides must not contain duplicates")
    unknown = sorted(set(result) - set(configured))
    if unknown:
        raise ValueError(
            f"{kind} overrides are not present in the config: {unknown}"
        )
    return result


def _preflight(
    config: BenchmarkConfig, *, providers: Sequence[str]
) -> Dict[str, object]:
    provider_records = {
        name: _probe_provider(name) for name in providers
    }
    if "triton" in provider_records:
        provider_records["triton"]["launch_configuration"] = (
            config.triton_launch.to_dict()
        )
    cuda_built = bool(torch.backends.cuda.is_built())
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_in_range = (
        cuda_available and config.device_index < device_count
    )
    blockers: List[str] = []
    if not cuda_built:
        blockers.append("this PyTorch build does not include CUDA")
    elif not cuda_available:
        blockers.append("PyTorch cannot access a CUDA device")
    elif not device_in_range:
        blockers.append(
            f"CUDA device index {config.device_index} is unavailable "
            f"(device_count={device_count})"
        )
    for name, record in provider_records.items():
        if not record["available"]:
            blockers.append(f"{name} provider unavailable: {record['error']}")
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
        "providers": provider_records,
    }


def _require_ready(preflight: Mapping[str, object]) -> None:
    if preflight["status"] == "ready":
        return
    blockers = preflight["blockers"]
    if not isinstance(blockers, list):
        raise RuntimeError("invalid preflight result")
    raise RuntimeError(
        "benchmark preflight failed: " + "; ".join(str(item) for item in blockers)
    )


def _nvidia_smi_query(
    fields: Sequence[str],
) -> Dict[str, object]:
    command = (
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "available": False,
            "command": list(command),
            "error": f"{type(error).__name__}: {error}",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "command": list(command),
            "error": completed.stderr.strip() or "nvidia-smi failed",
        }
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        rows.append(dict(zip(fields, values)))
    return {
        "available": True,
        "command": list(command),
        "units": {
            "memory.total": "MiB",
            "clocks.current.sm": "MHz",
            "clocks.current.memory": "MHz",
            "power.limit": "W",
        },
        "rows": rows,
    }


def _nvidia_smi_snapshot() -> Dict[str, object]:
    core_fields = (
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "driver_version",
        "memory.total",
    )
    telemetry_fields = (
        "index",
        "compute_cap",
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.limit",
    )
    return {
        "core": _nvidia_smi_query(core_fields),
        "telemetry": _nvidia_smi_query(telemetry_fields),
    }


def _property(
    properties: object, name: str
) -> Optional[object]:
    value = getattr(properties, name, None)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _environment(
    device: torch.device, config: BenchmarkConfig
) -> Dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
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
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
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
            "free_memory_bytes_at_start": int(free_bytes),
            "memory_total_bytes_at_start": int(total_bytes),
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
        "provider_runtime": {
            "triton": {
                "launch_configuration": config.triton_launch.to_dict()
            }
        },
        "relevant_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_CUDA_ARCH_LIST",
                "TRITON_CACHE_DIR",
            )
        },
    }


def _input_seed(base_seed: int, case_name: str, dtype_name: str) -> int:
    digest = sha256(
        f"{base_seed}\0{case_name}\0{dtype_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _make_inputs(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    normalize_embeddings: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    queries = torch.randn(
        (
            case.query_batch_size,
            case.query_tokens,
            case.embedding_dimension,
        ),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    documents = torch.randn(
        (
            case.document_batch_size,
            case.document_tokens,
            case.embedding_dimension,
        ),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    if normalize_embeddings:
        queries = F.normalize(queries, dim=-1)
        documents = F.normalize(documents, dim=-1)
    queries = queries.to(dtype=dtype)
    documents = documents.to(dtype=dtype)
    query_positions = torch.arange(
        case.query_tokens, device=device
    ).unsqueeze(0)
    document_positions = torch.arange(
        case.document_tokens, device=device
    ).unsqueeze(0)
    query_mask = (
        query_positions < case.active_query_tokens
    ).expand(case.query_batch_size, -1).clone()
    document_mask = (
        document_positions < case.active_document_tokens
    ).expand(case.document_batch_size, -1).clone()
    return queries, documents, query_mask, document_mask


def _storage_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _correctness(
    actual: Tensor,
    expected: Tensor,
    *,
    dtype_spec: DTypeSpec,
) -> Dict[str, object]:
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(
            f"provider returned shape {tuple(actual.shape)}; "
            f"expected {tuple(expected.shape)}"
        )
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    finite = bool(torch.isfinite(actual_float).all().item())
    absolute_difference = (actual_float - expected_float).abs()
    relative_difference = absolute_difference / expected_float.abs().clamp_min(
        1e-12
    )
    maximum_absolute_error = float(absolute_difference.max().item())
    maximum_relative_error = float(relative_difference.max().item())
    agreement = bool(
        torch.allclose(
            actual_float,
            expected_float,
            atol=dtype_spec.absolute_tolerance,
            rtol=dtype_spec.relative_tolerance,
        )
    )
    record = {
        "reference_provider": "torch",
        "finite": finite,
        "agreement": agreement,
        "absolute_tolerance": dtype_spec.absolute_tolerance,
        "relative_tolerance": dtype_spec.relative_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }
    if not finite or not agreement:
        raise RuntimeError(
            "provider failed numerical agreement: "
            f"finite={finite}, max_abs={maximum_absolute_error}, "
            f"max_rel={maximum_relative_error}, "
            f"atol={dtype_spec.absolute_tolerance}, "
            f"rtol={dtype_spec.relative_tolerance}"
        )
    return record


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


def _benchmark_provider(
    name: str,
    provider: Provider,
    *,
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    reference: Tensor,
    case: BenchmarkCase,
    dtype_spec: DTypeSpec,
    config: BenchmarkConfig,
    device: torch.device,
) -> Dict[str, object]:
    torch.cuda.synchronize(device)
    first_call_start = perf_counter()
    actual = _call_provider(
        provider, queries, documents, query_mask, document_mask
    )
    torch.cuda.synchronize(device)
    first_call_wall_ms = (perf_counter() - first_call_start) * 1000.0
    correctness = _correctness(actual, reference, dtype_spec=dtype_spec)
    del actual

    warmup_start = perf_counter()
    for _ in range(config.warmup_iterations):
        warmup_output = _call_provider(
            provider, queries, documents, query_mask, document_mask
        )
    torch.cuda.synchronize(device)
    warmup_wall_ms = (perf_counter() - warmup_start) * 1000.0
    del warmup_output

    trial_records: List[Dict[str, object]] = []
    all_latency_ms: List[float] = []
    for trial_index in range(config.trials):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(config.measured_iterations)
        ]
        host_start = perf_counter()
        for start_event, end_event in events:
            start_event.record()
            output = _call_provider(
                provider, queries, documents, query_mask, document_mask
            )
            end_event.record()
        torch.cuda.synchronize(device)
        host_wall_ms = (perf_counter() - host_start) * 1000.0
        samples = [
            float(start.elapsed_time(end)) for start, end in events
        ]
        all_latency_ms.extend(samples)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        del output
        trial_records.append(
            {
                "trial_index": trial_index,
                "cuda_event_latency_ms": samples,
                "host_wall_total_ms": host_wall_ms,
                "host_wall_per_call_ms": (
                    host_wall_ms / config.measured_iterations
                ),
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
            }
        )

    memory_records = [
        record["memory"] for record in trial_records
    ]
    if not all(isinstance(record, dict) for record in memory_records):
        raise RuntimeError("invalid internal memory record")
    return {
        "provider": name,
        "entrypoint": (
            "colpali_triton.maxsim.maxsim_vectorized"
            if name == "torch"
            else "colpali_triton.triton_maxsim.maxsim_triton"
        ),
        "launch_configuration": (
            config.triton_launch.to_dict() if name == "triton" else None
        ),
        "correctness": correctness,
        "cold_first_call_wall_ms": first_call_wall_ms,
        "warmup": {
            "iterations": config.warmup_iterations,
            "total_wall_ms": warmup_wall_ms,
        },
        "timing": {
            "method": "torch.cuda.Event(enable_timing=True)",
            "synchronization": (
                "one torch.cuda.synchronize after each enqueued trial"
            ),
            "measured_iterations_per_trial": config.measured_iterations,
            "trial_count": config.trials,
            "summary": summarize_latencies(
                all_latency_ms,
                documents_per_call=case.document_batch_size,
                query_document_pairs_per_call=(
                    case.query_document_pairs
                ),
            ),
            "trials": trial_records,
        },
        "memory": {
            "method": "torch.cuda peak allocator statistics per trial",
            "maximum_peak_allocated_bytes": max(
                int(record["peak_allocated_bytes"])
                for record in memory_records
            ),
            "maximum_peak_reserved_bytes": max(
                int(record["peak_reserved_bytes"])
                for record in memory_records
            ),
            "maximum_incremental_peak_allocated_bytes": max(
                int(record["incremental_peak_allocated_bytes"])
                for record in memory_records
            ),
            "maximum_incremental_peak_reserved_bytes": max(
                int(record["incremental_peak_reserved_bytes"])
                for record in memory_records
            ),
        },
    }


def _configured_case_record(case: BenchmarkCase) -> Dict[str, object]:
    record = case.to_dict()
    record["query_document_pairs"] = case.query_document_pairs
    return record


def run_benchmarks(
    config: BenchmarkConfig,
    *,
    provider_names: Sequence[str],
    dtype_names: Sequence[str],
    case_names: Sequence[str],
    config_path: Path,
    argv: Sequence[str],
) -> Dict[str, object]:
    """Run selected config entries after a strict CUDA/provider preflight."""

    preflight = _preflight(config, providers=provider_names)
    _require_ready(preflight)
    device = torch.device("cuda", config.device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    dtype_by_name = {item.name: item for item in config.dtypes}
    case_by_name = {item.name: item for item in config.cases}
    providers = {
        name: _load_provider(name, config) for name in provider_names
    }
    source_files, source_fingerprint = source_manifest(
        SOURCE_FILES, root=PROJECT_ROOT
    )
    environment = _environment(device, config)
    started_at = _utc_now()
    wall_start = perf_counter()
    results = []

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
                provider_results = []
                for name in provider_names:
                    provider_results.append(
                        _benchmark_provider(
                            name,
                            providers[name],
                            queries=queries,
                            documents=documents,
                            query_mask=query_mask,
                            document_mask=document_mask,
                            reference=reference,
                            case=case,
                            dtype_spec=dtype_spec,
                            config=config,
                            device=device,
                        )
                    )
                results.append(
                    {
                        "case": _configured_case_record(case),
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
                        "providers": provider_results,
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

    duration_seconds = perf_counter() - wall_start
    selection_is_canonical = (
        tuple(provider_names) == config.providers
        and tuple(dtype_names)
        == tuple(item.name for item in config.dtypes)
        and tuple(case_names)
        == tuple(item.name for item in config.cases)
    )
    config_file_digest = sha256_file(config_path)
    combined_fingerprint = sha256(
        (source_fingerprint + "\0" + config_file_digest).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "maxsim_cuda_benchmark",
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "wall_duration_seconds": duration_seconds,
        "selection": {
            "canonical_full_matrix": selection_is_canonical,
            "providers": list(provider_names),
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
            "semantic_config_sha256": canonical_config_fingerprint(config),
            "source_sha256": source_files,
            "source_fingerprint_sha256": source_fingerprint,
            "source_and_config_fingerprint_sha256": combined_fingerprint,
            "git": _git_state(),
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
            "provider_order": list(provider_names),
            "correctness_checked_before_timing": True,
            "warmups_exclude_cold_first_call": True,
        },
        "preflight": preflight,
        "environment": environment,
        "results": results,
    }


def _preflight_output(path: Path, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )


def _write_json_atomic(
    path: Path, value: Mapping[str, object], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if overwrite:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"output already exists: {path}; "
                    "pass --overwrite to replace it"
                ) from error
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed = _parse_args(argv)
    try:
        config = load_benchmark_config(parsed.config)
        provider_names = _select_names(
            parsed.provider,
            config.providers,
            kind="provider",
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
        preflight = _preflight(config, providers=provider_names)
        if parsed.preflight:
            print(
                json.dumps(
                    {
                        "benchmark_id": config.benchmark_id,
                        "config_file_sha256": sha256_file(parsed.config),
                        "semantic_config_sha256": (
                            canonical_config_fingerprint(config)
                        ),
                        "selection": {
                            "providers": list(provider_names),
                            "dtypes": list(dtype_names),
                            "cases": list(case_names),
                        },
                        "preflight": preflight,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        _require_ready(preflight)
        _preflight_output(parsed.output, parsed.overwrite)
        command_argv = tuple(sys.argv if argv is None else argv)
        result = run_benchmarks(
            config,
            provider_names=provider_names,
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
                    "status": "completed",
                    "output": str(parsed.output.resolve()),
                    "output_sha256": sha256_file(parsed.output),
                    "result_count": len(result["results"]),
                },
                sort_keys=True,
            )
        )
        return 0
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
