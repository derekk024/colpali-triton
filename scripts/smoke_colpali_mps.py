#!/usr/bin/env python3
"""Run one released-checkpoint ColPali document/query smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn

from colpali_triton import __version__, late_interaction_scores
from colpali_triton.checkpoints import (
    LoadedColPali,
    load_released_colpali,
)
from colpali_triton.modeling import EncodingKind, MultiVectorEncoding
from colpali_triton.phase7_config import (
    PHASE7_CONFIG,
    Phase7Config,
    load_phase7_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "colpali_phase7.json"
DEFAULT_CONSTRAINTS = (
    PROJECT_ROOT / "configs" / "phase7_macos_arm64_constraints.txt"
)
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "artifacts" / "huggingface_phase7"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "phase8" / "colpali_mps_smoke.json"
)
DEFAULT_QUERY = "What is the invoice total?"
SEED = 17
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "configs" / "colpali_phase7.json",
    PROJECT_ROOT / "src" / "colpali_triton" / "__init__.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "adaptation.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "checkpoints.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "maxsim.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "modeling.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "paligemma.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "phase7_config.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "processing.py",
)
PACKAGE_NAMES = (
    "accelerate",
    "huggingface-hub",
    "numpy",
    "peft",
    "Pillow",
    "safetensors",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="MPS is the Phase 8 acceptance path; CPU is for diagnostics.",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa"),
        default="eager",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require every pinned checkpoint file to exist in the cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output record.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(paths: Sequence[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        relative = path.relative_to(PROJECT_ROOT)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _preflight_output(path: Path, overwrite: bool) -> None:
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )


def _write_json_atomic(
    path: Path, value: Mapping[str, object], overwrite: bool
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
    }


def _sysctl(name: str) -> Optional[str]:
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        ("/usr/sbin/sysctl", "-n", name),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value or None


def _peak_process_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _mps_memory_snapshot(device: torch.device) -> Dict[str, Optional[int]]:
    if device.type != "mps":
        return {
            "current_allocated_bytes": None,
            "driver_allocated_bytes": None,
            "recommended_max_bytes": None,
        }
    return {
        "current_allocated_bytes": int(
            torch.mps.current_allocated_memory()
        ),
        "driver_allocated_bytes": int(
            torch.mps.driver_allocated_memory()
        ),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_smoke_page() -> Image.Image:
    image = Image.new("RGB", (448, 448), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((24, 24, 424, 424), outline=(20, 20, 20), width=3)
    draw.text((48, 54), "COLPALI TEST INVOICE", fill=(0, 0, 0), font=font)
    draw.line((48, 82, 400, 82), fill=(0, 0, 0), width=2)
    draw.text((48, 118), "Document retrieval service", fill=(0, 0, 0), font=font)
    draw.text((48, 150), "Quantity: 3", fill=(0, 0, 0), font=font)
    draw.text((48, 182), "Unit price: $14.00", fill=(0, 0, 0), font=font)
    draw.rectangle((44, 260, 404, 330), outline=(0, 0, 0), width=2)
    draw.text((66, 285), "INVOICE TOTAL: $42.00", fill=(0, 0, 0), font=font)
    return image


def _image_record(image: Image.Image) -> Dict[str, object]:
    return {
        "kind": "deterministic_generated_invoice",
        "mode": image.mode,
        "size_pixels": list(image.size),
        "pixel_bytes_sha256": sha256(image.tobytes()).hexdigest(),
        "visible_total": "$42.00",
    }


def _encoding_record(
    encoding: MultiVectorEncoding,
    *,
    expected_kind: EncodingKind,
    norm_tolerance: float = 0.02,
) -> Dict[str, object]:
    if not isinstance(encoding, MultiVectorEncoding):
        raise TypeError("model must return MultiVectorEncoding")
    if encoding.kind is not expected_kind:
        raise RuntimeError(
            f"encoding kind is {encoding.kind.value}; "
            f"expected {expected_kind.value}"
        )
    active = encoding.embeddings[encoding.mask]
    if active.numel() == 0:
        raise RuntimeError("encoding contains no active vectors")
    active_float = active.detach().to(device="cpu", dtype=torch.float32)
    norms = torch.linalg.vector_norm(active_float, ord=2, dim=-1)
    if not bool(torch.isfinite(norms).all().item()):
        raise RuntimeError("encoding norms are not finite")
    maximum_error = float(torch.max(torch.abs(norms - 1.0)).item())
    if maximum_error > norm_tolerance:
        raise RuntimeError(
            "active vectors are not unit normalized: "
            f"maximum error {maximum_error}"
        )
    masked = encoding.embeddings.masked_select(
        ~encoding.mask.unsqueeze(-1)
    )
    masked_nonzero = (
        int(torch.count_nonzero(masked).item()) if masked.numel() else 0
    )
    if masked_nonzero:
        raise RuntimeError("masked embedding positions are not exactly zero")
    return {
        "kind": encoding.kind.value,
        "shape": list(encoding.embeddings.shape),
        "dtype": str(encoding.embeddings.dtype).removeprefix("torch."),
        "device": str(encoding.embeddings.device),
        "active_vector_count": int(encoding.mask.sum().item()),
        "active_norm_minimum": float(norms.min().item()),
        "active_norm_maximum": float(norms.max().item()),
        "active_norm_maximum_absolute_error": maximum_error,
        "masked_nonzero_values": masked_nonzero,
        "payload_bytes": (
            encoding.embeddings.numel()
            * encoding.embeddings.element_size()
        ),
    }


def _parameter_inventory(model: nn.Module) -> Dict[str, object]:
    parameters = tuple(model.named_parameters())
    dtype_counts: Dict[str, int] = {}
    device_counts: Dict[str, int] = {}
    for _, parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        device = str(parameter.device)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + parameter.numel()
        device_counts[device] = device_counts.get(device, 0) + parameter.numel()
    return {
        "total_parameters": sum(
            parameter.numel() for _, parameter in parameters
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for _, parameter in parameters
            if parameter.requires_grad
        ),
        "parameter_counts_by_dtype": dtype_counts,
        "parameter_counts_by_device": device_counts,
        "training": bool(model.training),
    }


def run_smoke(
    config: Phase7Config,
    *,
    model_cache: Path,
    device: str,
    attention_implementation: str,
    query: str,
    local_files_only: bool,
    loader: Callable[..., LoadedColPali] = load_released_colpali,
) -> Dict[str, object]:
    """Run the measured model path and return its runtime record."""

    if not isinstance(config, Phase7Config):
        raise TypeError("config must be a Phase7Config")
    if device not in {"mps", "cpu"}:
        raise ValueError("device must be mps or cpu")
    if attention_implementation not in {"eager", "sdpa"}:
        raise ValueError("attention_implementation must be eager or sdpa")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonblank string")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")

    torch.manual_seed(SEED)
    page = _make_smoke_page()
    memory_before_load = _mps_memory_snapshot(torch.device(device))
    load_started = perf_counter()
    released = loader(
        config,
        cache_dir=model_cache,
        device=device,
        local_files_only=local_files_only,
        verify_artifacts=True,
        attention_implementation=attention_implementation,
    )
    _synchronize(released.device)
    load_seconds = perf_counter() - load_started
    memory_after_load = _mps_memory_snapshot(released.device)

    processing_started = perf_counter()
    document_batch = released.processor.process_documents([page])
    query_batch = released.processor.process_queries([query])
    document_batch = document_batch.to(
        released.device,
        pixel_dtype=released.dtype,
    )
    query_batch = query_batch.to(released.device)
    _synchronize(released.device)
    processing_seconds = perf_counter() - processing_started

    expected_document_tokens = (
        config.architecture.image_token_count + 6
    )
    if tuple(document_batch.input_ids.shape) != (
        1,
        expected_document_tokens,
    ):
        raise RuntimeError(
            "document token shape drifted: "
            f"{tuple(document_batch.input_ids.shape)}"
        )
    if int(document_batch.attention_mask.sum().item()) != (
        expected_document_tokens
    ):
        raise RuntimeError("document active-token count drifted")
    query_active_tokens = int(query_batch.attention_mask.sum().item())
    if not 0 < query_active_tokens <= config.processing.text_max_length:
        raise RuntimeError("query active-token count is outside the contract")

    with torch.inference_mode():
        _synchronize(released.device)
        document_started = perf_counter()
        documents = released.model(**document_batch.as_model_inputs())
        _synchronize(released.device)
        document_seconds = perf_counter() - document_started
        memory_after_document = _mps_memory_snapshot(released.device)

        query_started = perf_counter()
        queries = released.model(**query_batch.as_model_inputs())
        _synchronize(released.device)
        query_seconds = perf_counter() - query_started
        memory_after_query = _mps_memory_snapshot(released.device)

        scoring_started = perf_counter()
        scores = late_interaction_scores(queries, documents)
        _synchronize(released.device)
        scoring_seconds = perf_counter() - scoring_started
        memory_after_score = _mps_memory_snapshot(released.device)

    if not isinstance(scores, Tensor) or tuple(scores.shape) != (1, 1):
        raise RuntimeError("smoke score must have shape [1, 1]")
    score = float(scores.detach().to(device="cpu", dtype=torch.float32).item())
    if not torch.isfinite(torch.tensor(score)):
        raise RuntimeError("smoke score must be finite")

    document_record = _encoding_record(
        documents,
        expected_kind=EncodingKind.DOCUMENT,
    )
    query_record = _encoding_record(
        queries,
        expected_kind=EncodingKind.QUERY,
    )
    parameter_inventory = _parameter_inventory(released.model)
    if parameter_inventory["trainable_parameters"] != 0:
        raise RuntimeError("released inference model has trainable parameters")
    if parameter_inventory["training"]:
        raise RuntimeError("released inference model is in training mode")

    snapshots = (
        memory_before_load,
        memory_after_load,
        memory_after_document,
        memory_after_query,
        memory_after_score,
    )
    current_values = [
        item["current_allocated_bytes"]
        for item in snapshots
        if item["current_allocated_bytes"] is not None
    ]
    driver_values = [
        item["driver_allocated_bytes"]
        for item in snapshots
        if item["driver_allocated_bytes"] is not None
    ]
    return {
        "checkpoint": {
            "runtime_base": config.runtime_base.to_dict(),
            "paper_adapter": config.paper_adapter.to_dict(),
            "verified_artifacts": [
                artifact.to_dict() for artifact in released.artifacts
            ],
            "verified_artifact_count": len(released.artifacts),
            "lora_target_count": released.lora_target_count,
            "lora_parameter_count": released.lora_parameter_count,
        },
        "model": {
            "class": type(released.model).__name__,
            "device": str(released.device),
            "declared_dtype": str(released.dtype).removeprefix("torch."),
            "attention_implementation": released.attention_implementation,
            **parameter_inventory,
        },
        "input": {
            "page": _image_record(page),
            "query": query,
            "document_input_shape": list(document_batch.input_ids.shape),
            "document_active_tokens": int(
                document_batch.attention_mask.sum().item()
            ),
            "document_pixel_shape": list(
                document_batch.pixel_values.shape
            ),
            "document_pixel_dtype": str(
                document_batch.pixel_values.dtype
            ).removeprefix("torch."),
            "query_input_shape": list(query_batch.input_ids.shape),
            "query_active_tokens": query_active_tokens,
        },
        "encoding": {
            "document": document_record,
            "query": query_record,
        },
        "score": {
            "function": "pytorch_vectorized_maxsim",
            "shape": list(scores.shape),
            "dtype": str(scores.dtype).removeprefix("torch."),
            "value": score,
            "finite": True,
        },
        "timing_seconds": {
            "artifact_verification_and_model_load": load_seconds,
            "processing_and_device_transfer": processing_seconds,
            "document_encoding": document_seconds,
            "query_encoding": query_seconds,
            "maxsim_scoring": scoring_seconds,
        },
        "memory": {
            "mps_before_load": memory_before_load,
            "mps_after_load": memory_after_load,
            "mps_after_document": memory_after_document,
            "mps_after_query": memory_after_query,
            "mps_after_score": memory_after_score,
            "observed_mps_current_allocated_max_bytes": (
                max(current_values) if current_values else None
            ),
            "observed_mps_driver_allocated_max_bytes": (
                max(driver_values) if driver_values else None
            ),
            "peak_process_rss_bytes": _peak_process_rss_bytes(),
            "measurement_note": (
                "MPS values are synchronized allocator snapshots, not a "
                "continuous profiler trace."
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_path = args.output.resolve()
    _preflight_output(output_path, args.overwrite)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config_path = args.config.resolve()
    config = load_phase7_config(config_path)
    if config != PHASE7_CONFIG:
        raise ValueError(
            "the supplied Phase 7 config does not match the built-in contract"
        )
    runtime = run_smoke(
        config,
        model_cache=args.model_cache.resolve(),
        device=args.device,
        attention_implementation=args.attention_implementation,
        query=args.query,
        local_files_only=args.local_files_only,
    )
    constraints = (
        {
            "path": str(DEFAULT_CONSTRAINTS.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(DEFAULT_CONSTRAINTS),
        }
        if DEFAULT_CONSTRAINTS.exists()
        else None
    )
    record: Dict[str, object] = {
        "schema_version": 1,
        "result_kind": "measured_mps_smoke",
        "phase": 8,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "project_version": __version__,
        "configuration": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "canonical_fingerprint": config.fingerprint,
            "contract": config.to_dict(),
        },
        "runtime": runtime,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "apple_cpu_brand": _sysctl("machdep.cpu.brand_string"),
            "physical_memory_bytes": _sysctl("hw.memsize"),
            "torch_mps_built": torch.backends.mps.is_built(),
            "torch_mps_available": torch.backends.mps.is_available(),
            "torch_cuda_available": torch.cuda.is_available(),
            "package_versions": {
                name: _package_version(name) for name in PACKAGE_NAMES
            },
            "constraints": constraints,
        },
        "provenance": {
            "git": _git_state(),
            "source_hashes": {
                str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
                for path in SOURCE_FILES
            },
            "source_fingerprint": _source_fingerprint(SOURCE_FILES),
            "invocation_arguments": list(
                sys.argv[1:] if argv is None else argv
            ),
        },
        "scope": {
            "is_quality_evaluation": False,
            "is_benchmark": False,
            "claim": (
                "One released-checkpoint document/query forward pass and "
                "MaxSim score completed with validated shapes and values."
            ),
        },
    }
    _write_json_atomic(output_path, record, args.overwrite)
    summary = {
        "output": str(output_path),
        "device": runtime["model"]["device"],
        "document_shape": runtime["encoding"]["document"]["shape"],
        "query_shape": runtime["encoding"]["query"]["shape"],
        "score": runtime["score"]["value"],
        "peak_process_rss_bytes": runtime["memory"][
            "peak_process_rss_bytes"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
