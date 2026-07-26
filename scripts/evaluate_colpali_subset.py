#!/usr/bin/env python3
"""Evaluate released ColPali and mean pooling on one fixed ViDoRe subset."""

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
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from colpali_triton import __version__
from colpali_triton.checkpoints import load_released_colpali
from colpali_triton.multimodal_evaluation import (
    encode_document_index,
    evaluate_document_index,
)
from colpali_triton.phase7_config import (
    PHASE7_CONFIG,
    load_phase7_config,
)
from colpali_triton.phase9_config import (
    PHASE9_CONFIG,
    load_phase9_config,
)
from colpali_triton.vidore import load_manifest
from colpali_triton.vidore_images import load_vidore_image_subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE7_CONFIG = PROJECT_ROOT / "configs" / "colpali_phase7.json"
DEFAULT_PHASE9_CONFIG = (
    PROJECT_ROOT / "configs" / "phase9_colpali_subset.json"
)
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "configs" / "vidore_phase6.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "artifacts" / "huggingface_phase7"
DEFAULT_DATASET_CACHE = PROJECT_ROOT / "artifacts" / "huggingface_phase9"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase9"
DEFAULT_CONSTRAINTS = (
    PROJECT_ROOT / "configs" / "phase7_macos_arm64_constraints.txt"
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "configs" / "colpali_phase7.json",
    PROJECT_ROOT / "configs" / "phase9_colpali_subset.json",
    PROJECT_ROOT / "configs" / "vidore_phase6.json",
    PROJECT_ROOT / "src" / "colpali_triton" / "checkpoints.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "maxsim.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "metrics.py",
    PROJECT_ROOT
    / "src"
    / "colpali_triton"
    / "multimodal_evaluation.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "paligemma.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "phase9_config.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "processing.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "vidore.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "vidore_images.py",
)
PACKAGE_NAMES = (
    "accelerate",
    "huggingface-hub",
    "numpy",
    "peft",
    "Pillow",
    "pyarrow",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=("docvqa", "infovqa")
    )
    parser.add_argument(
        "--phase7-config", type=Path, default=DEFAULT_PHASE7_CONFIG
    )
    parser.add_argument(
        "--phase9-config", type=Path, default=DEFAULT_PHASE9_CONFIG
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    parser.add_argument(
        "--dataset-cache", type=Path, default=DEFAULT_DATASET_CACHE
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require all model and dataset artifacts in their caches.",
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
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(
            str(path.relative_to(PROJECT_ROOT)).encode("utf-8")
        )
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
    lines = tuple(line for line in status.stdout.splitlines() if line)
    return {
        "available": revision.returncode == 0 and status.returncode == 0,
        "commit": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "dirty": bool(lines),
        "status_entry_count": len(lines),
    }


def _pip_check() -> Dict[str, object]:
    completed = subprocess.run(
        (sys.executable, "-m", "pip", "check"),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    details = tuple(
        line
        for line in (completed.stdout + completed.stderr).splitlines()
        if line
    )
    return {
        "passed": completed.returncode == 0,
        "return_code": completed.returncode,
        "details": list(details),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


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


def _mps_memory() -> Dict[str, int]:
    return {
        "current_allocated_bytes": int(
            torch.mps.current_allocated_memory()
        ),
        "driver_allocated_bytes": int(
            torch.mps.driver_allocated_memory()
        ),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _model_inventory(model: torch.nn.Module) -> Dict[str, object]:
    parameters = tuple(model.named_parameters())
    dtype_counts: Dict[str, int] = {}
    device_counts: Dict[str, int] = {}
    for _, parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        device = str(parameter.device)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + parameter.numel()
        device_counts[device] = device_counts.get(device, 0) + parameter.numel()
    return {
        "class": type(model).__name__,
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


def _default_output(task: str) -> Path:
    return DEFAULT_OUTPUT_DIRECTORY / f"{task}_colpali_subset.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else _default_output(args.task).resolve()
    )
    _preflight_output(output_path, args.overwrite)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    phase7_path = args.phase7_config.resolve()
    phase9_path = args.phase9_config.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    phase7 = load_phase7_config(phase7_path)
    phase9 = load_phase9_config(phase9_path)
    dataset_manifest = load_manifest(dataset_manifest_path)
    if phase7 != PHASE7_CONFIG:
        raise ValueError("Phase 7 model configuration drifted")
    if phase9 != PHASE9_CONFIG:
        raise ValueError("Phase 9 evaluation configuration drifted")
    if phase7.fingerprint != phase9.model_manifest_fingerprint:
        raise ValueError("Phase 9 model manifest fingerprint drifted")
    if dataset_manifest.fingerprint != (
        phase9.dataset_manifest_fingerprint
    ):
        raise ValueError("Phase 9 dataset manifest fingerprint drifted")
    source_spec = dataset_manifest.datasets[args.task]
    task_contract = phase9.tasks[args.task]
    evaluation = phase9.evaluation
    torch.manual_seed(phase9.seed)
    np.random.seed(phase9.seed)

    dataset_started = perf_counter()
    dataset = load_vidore_image_subset(
        source_spec,
        phase9.selection,
        task_contract,
        cache_dir=args.dataset_cache.resolve(),
        local_files_only=args.local_files_only,
    )
    dataset_load_seconds = perf_counter() - dataset_started

    memory_before_model = _mps_memory()
    model_started = perf_counter()
    released = load_released_colpali(
        phase7,
        cache_dir=args.model_cache.resolve(),
        device=evaluation.device,
        local_files_only=args.local_files_only,
        verify_artifacts=True,
        attention_implementation=evaluation.attention_implementation,
    )
    torch.mps.synchronize()
    model_load_seconds = perf_counter() - model_started
    memory_after_model = _mps_memory()

    index = encode_document_index(
        released.model,
        released.processor,
        dataset,
        device=released.device,
        pixel_dtype=released.dtype,
        document_batch_size=evaluation.document_batch_size,
    )
    memory_after_index = _mps_memory()
    result = evaluate_document_index(
        released.model,
        released.processor,
        dataset,
        index,
        k=evaluation.k,
        warmup_count=evaluation.warmup_queries,
        score_document_batch_size=(
            evaluation.score_document_batch_size
        ),
    )
    torch.mps.synchronize()
    memory_after_evaluation = _mps_memory()

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
        "result_kind": "measured_colpali_subset_evaluation",
        "phase": 9,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": __version__,
        "seed": phase9.seed,
        "task": {
            "name": dataset.name,
            "split": source_spec.split,
            "full_source_counts": {
                "documents": source_spec.expected_document_count,
                "queries": source_spec.expected_query_count,
                "qrels": source_spec.expected_qrel_count,
            },
            "selected_counts": {
                "documents": len(dataset.documents),
                "queries": len(dataset.queries),
                "qrels": sum(
                    len(query.relevant_document_ids)
                    for query in dataset.queries
                ),
            },
            "selection": phase9.selection.to_dict(),
            "source_spec": source_spec.to_dict(),
            "source_spec_fingerprint": source_spec.fingerprint,
            "selection_fingerprint": dataset.selection_fingerprint,
            "image_content_fingerprint": (
                dataset.image_content_fingerprint
            ),
            "verified_artifacts": [
                artifact.to_dict() for artifact in dataset.artifacts
            ],
            "documents": [
                document.metadata() for document in dataset.documents
            ],
            "queries": [
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "relevant_document_ids": sorted(
                        query.relevant_document_ids
                    ),
                }
                for query in dataset.queries
            ],
        },
        "model": {
            "runtime_base": phase7.runtime_base.to_dict(),
            "paper_adapter": phase7.paper_adapter.to_dict(),
            "verified_artifacts": [
                artifact.to_dict() for artifact in released.artifacts
            ],
            "lora_target_count": released.lora_target_count,
            "lora_parameter_count": released.lora_parameter_count,
            "device": str(released.device),
            "dtype": str(released.dtype).removeprefix("torch."),
            "attention_implementation": (
                released.attention_implementation
            ),
            **_model_inventory(released.model),
        },
        "evaluation": {
            "configuration": evaluation.to_dict(),
            "dataset_load_seconds": dataset_load_seconds,
            "model_artifact_verification_and_load_seconds": (
                model_load_seconds
            ),
            **result.to_dict(),
        },
        "memory": {
            "mps_before_model": memory_before_model,
            "mps_after_model": memory_after_model,
            "mps_after_index": memory_after_index,
            "mps_after_evaluation": memory_after_evaluation,
            "peak_process_rss_bytes": _peak_process_rss_bytes(),
            "measurement_note": (
                "MPS values are synchronized allocator snapshots, not a "
                "continuous profiler trace."
            ),
        },
        "configuration": {
            "phase7": {
                "path": str(phase7_path),
                "sha256": _sha256_file(phase7_path),
                "fingerprint": phase7.fingerprint,
            },
            "phase9": {
                "path": str(phase9_path),
                "sha256": _sha256_file(phase9_path),
                "fingerprint": phase9.fingerprint,
            },
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": _sha256_file(dataset_manifest_path),
                "fingerprint": dataset_manifest.fingerprint,
            },
        },
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
            "pip_check": _pip_check(),
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
        "comparability": {
            "paper_comparable_metric": "nDCG@5",
            "project_extensions": ["Recall@5", "MRR@5"],
            "is_full_task": False,
            "is_paper_result_reproduction": False,
            "notes": [
                "The subset was fixed by identifier hashes before scoring.",
                "Every selected query is scored against all 100 selected pages.",
                "Every positive for each selected query is present.",
                "The same model encodings feed MaxSim and mean pooling.",
                "Local Apple MPS timings are not comparable to NVIDIA results.",
            ],
        },
    }
    _write_json_atomic(output_path, record, args.overwrite)
    summary = {
        "output": str(output_path),
        "task": args.task,
        "documents": len(dataset.documents),
        "queries": len(dataset.queries),
        "indexing_seconds": result.indexing["seconds"],
        "late_interaction": {
            "ndcg@5": result.late_interaction.ndcg_at_k,
            "recall@5": result.late_interaction.recall_at_k,
            "mrr@5": result.late_interaction.mrr_at_k,
        },
        "mean_pooling": {
            "ndcg@5": result.mean_pooling.ndcg_at_k,
            "recall@5": result.mean_pooling.recall_at_k,
            "mrr@5": result.mean_pooling.mrr_at_k,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
