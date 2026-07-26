#!/usr/bin/env python3
"""Run one measured OCR-text baseline on one pinned ViDoRe task."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import re
import subprocess
import sys
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from colpali_triton import __version__
from colpali_triton.baselines import (
    BM25ChunkMaxIndex,
    DenseChunkMaxIndex,
    SentenceTransformerTextEncoder,
)
from colpali_triton.evaluation import evaluate_retrieval
from colpali_triton.vidore import (
    VIDORE_DATASETS,
    load_manifest,
    load_vidore_text_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "text_baselines.json"
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "configs" / "vidore_phase6.json"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "artifacts" / "huggingface"
DEFAULT_CONSTRAINTS = (
    PROJECT_ROOT / "configs" / "phase6_macos_arm64_constraints.txt"
)
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "src" / "colpali_triton" / "__init__.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "baselines.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "evaluation.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "metrics.py",
    PROJECT_ROOT / "src" / "colpali_triton" / "vidore.py",
)


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for raw_key, value in pairs:
        if not isinstance(raw_key, str):
            raise ValueError("configuration keys must be strings")
        if raw_key in result:
            raise ValueError(f"duplicate JSON key: {raw_key!r}")
        result[raw_key] = value
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(VIDORE_DATASETS))
    parser.add_argument(
        "--baseline", required=True, choices=("bm25", "dense")
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Dense encoder device; ignored by BM25.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--model-cache", type=Path, default=DEFAULT_MODEL_CACHE
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"failed to read configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("baseline configuration must be a JSON object")
    if value.get("schema_version") != 1:
        raise ValueError("baseline configuration schema_version must be 1")
    return value


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
        "available": status.returncode == 0,
        "commit": (
            revision.stdout.strip() if revision.returncode == 0 else None
        ),
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
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
        "details": details,
    }


def _preflight_output(path: Optional[Path], overwrite: bool) -> None:
    if path is None:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {path}")


def _require_configuration_value(
    configuration: Mapping[str, object], key: str, expected: object
) -> None:
    actual = configuration.get(key)
    if actual != expected:
        raise ValueError(
            f"configuration {key!r} must be {expected!r}; got {actual!r}"
        )


def _configuration_integer(
    configuration: Mapping[str, object],
    key: str,
    *,
    minimum: int,
) -> int:
    value = configuration.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"configuration {key!r} must be an integer >= {minimum}; "
            f"got {value!r}"
        )
    return value


def _validate_dense_configuration(
    configuration: Mapping[str, object],
) -> None:
    model_id = configuration.get("model_id")
    revision = configuration.get("model_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("dense model_id must be a nonempty string")
    if not isinstance(revision, str) or re.fullmatch(
        r"[0-9a-f]{40}", revision
    ) is None:
        raise ValueError("dense model_revision must be a full commit SHA")
    weight_hash = configuration.get("model_weight_sha256")
    if not isinstance(weight_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", weight_hash
    ) is None:
        raise ValueError("dense model_weight_sha256 must be a SHA-256 digest")
    weight_size = configuration.get("model_weight_size_bytes")
    if (
        isinstance(weight_size, bool)
        or not isinstance(weight_size, int)
        or weight_size <= 0
    ):
        raise ValueError("dense model_weight_size_bytes must be positive")

    _require_configuration_value(configuration, "pooling", "cls")
    _require_configuration_value(
        configuration, "normalize_embeddings", True
    )
    _require_configuration_value(configuration, "query_prefix", "")
    _require_configuration_value(configuration, "dtype", "float32")
    _require_configuration_value(
        configuration,
        "sentence_transformers_version",
        _package_version("sentence-transformers"),
    )
    _require_configuration_value(
        configuration,
        "transformers_version",
        _package_version("transformers"),
    )


def _validate_common_configuration(
    config: Mapping[str, object],
    chunking: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> None:
    expected_root_keys = {
        "schema_version",
        "seed",
        "evaluation",
        "chunking",
        "bm25",
        "dense",
    }
    if set(config) != expected_root_keys:
        raise ValueError(
            "baseline configuration keys do not match the schema; "
            f"missing={sorted(expected_root_keys - set(config))!r}, "
            f"extra={sorted(set(config) - expected_root_keys)!r}"
        )
    _require_configuration_value(
        chunking, "unit", "normalized_whitespace_words"
    )
    _require_configuration_value(
        chunking, "page_aggregation", "maximum_chunk_score"
    )
    _require_configuration_value(
        evaluation,
        "tie_break",
        "score_descending_then_document_id_descending",
    )


def _validate_bm25_configuration(
    configuration: Mapping[str, object],
) -> None:
    _require_configuration_value(
        configuration, "formula", "rank_bm25_0.2.2_BM25Okapi"
    )
    _require_configuration_value(
        configuration,
        "tokenizer",
        "NFKC_casefold_unicode_alphanumeric",
    )
    _require_configuration_value(configuration, "stopwords", False)


def _prepare_model_artifact(
    configuration: Mapping[str, object], cache_folder: Path
) -> Dict[str, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required for model verification"
        ) from error

    model_path = Path(
        hf_hub_download(
            repo_id=str(configuration["model_id"]),
            filename="pytorch_model.bin",
            revision=str(configuration["model_revision"]),
            cache_dir=str(cache_folder),
        )
    )
    actual_size = model_path.stat().st_size
    actual_hash = _sha256_file(model_path)
    expected_size = int(configuration["model_weight_size_bytes"])
    expected_hash = str(configuration["model_weight_sha256"])
    if actual_size != expected_size or actual_hash != expected_hash:
        raise RuntimeError(
            "pinned model weight verification failed: "
            f"size={actual_size}, sha256={actual_hash}"
        )
    return {
        "filename": "pytorch_model.bin",
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "verified": True,
    }


def _token_length_summary(
    lengths: np.ndarray, *, truncation_limit: int
) -> Dict[str, object]:
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("token lengths must be a nonempty vector")
    if not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths <= 0):
        raise ValueError("token lengths must be positive integers")
    truncated = int(np.count_nonzero(lengths > truncation_limit))
    return {
        "count": int(lengths.size),
        "minimum": int(lengths.min()),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "maximum": int(lengths.max()),
        "truncation_limit": truncation_limit,
        "count_exceeding_limit": truncated,
        "fraction_exceeding_limit": truncated / int(lengths.size),
    }


def _select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    # Darwin reports bytes; Linux and most BSDs report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def _write_json_atomic(path: Path, value: Dict[str, object], overwrite: bool) -> None:
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
    args = _parse_args(argv)
    _preflight_output(args.output, args.overwrite)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    invocation_arguments = tuple(
        str(argument)
        for argument in (sys.argv[1:] if argv is None else argv)
    )
    config_path = args.config.resolve()
    manifest_path = args.dataset_manifest.resolve()
    config = _load_json(args.config)
    manifest = load_manifest(args.dataset_manifest)
    try:
        spec = manifest.datasets[args.task]
    except KeyError as error:
        raise ValueError(
            f"dataset manifest does not define task {args.task!r}"
        ) from error

    apple_cpu_brand = _sysctl("machdep.cpu.brand_string")
    physical_memory = _sysctl("hw.memsize")
    git_state = _git_state()
    pip_check = _pip_check()
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in SOURCE_FILES
    }
    source_fingerprint = _source_fingerprint(SOURCE_FILES)
    config_sha256 = _sha256_file(config_path)
    manifest_sha256 = _sha256_file(manifest_path)

    chunking = config["chunking"]
    evaluation_config = config["evaluation"]
    if not isinstance(chunking, dict) or not isinstance(
        evaluation_config, dict
    ):
        raise ValueError("chunking and evaluation must be JSON objects")
    _validate_common_configuration(config, chunking, evaluation_config)

    seed = _configuration_integer(config, "seed", minimum=0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    max_words = _configuration_integer(
        chunking, "max_words", minimum=1
    )
    overlap = _configuration_integer(
        chunking, "overlap_words", minimum=0
    )
    k = _configuration_integer(evaluation_config, "k", minimum=1)
    warmup_queries = _configuration_integer(
        evaluation_config, "warmup_queries", minimum=0
    )

    load_started = perf_counter()
    dataset = load_vidore_text_dataset(spec)
    dataset_load_seconds = perf_counter() - load_started

    model_load_seconds = 0.0
    model_artifact_prepare_seconds = 0.0
    model_artifact: Optional[Dict[str, object]] = None
    device: Optional[str] = None
    encoder_metadata: Optional[Dict[str, object]] = None
    tokenization_analysis: Optional[Dict[str, object]] = None

    if args.baseline == "bm25":
        baseline_config = config["bm25"]
        if not isinstance(baseline_config, dict):
            raise ValueError("bm25 configuration must be a JSON object")
        _validate_bm25_configuration(baseline_config)
        index_started = perf_counter()
        index = BM25ChunkMaxIndex(
            dataset.documents,
            max_words=max_words,
            overlap=overlap,
            k1=float(baseline_config["k1"]),
            b=float(baseline_config["b"]),
            epsilon=float(baseline_config["epsilon"]),
        )
        indexing_seconds = perf_counter() - index_started
        effective_configuration: Dict[str, object] = {
            "name": baseline_config["name"],
            "formula": baseline_config["formula"],
            "k1": index.k1,
            "b": index.b,
            "epsilon": index.epsilon,
            "tokenizer": baseline_config["tokenizer"],
            "stopwords": False,
        }
    else:
        baseline_config = config["dense"]
        if not isinstance(baseline_config, dict):
            raise ValueError("dense configuration must be a JSON object")
        _validate_dense_configuration(baseline_config)
        device = _select_device(args.device)
        batch_sizes = baseline_config["batch_size"]
        if not isinstance(batch_sizes, dict):
            raise ValueError("dense batch_size must be a JSON object")
        if set(batch_sizes) != {"cpu", "mps", "cuda"}:
            raise ValueError(
                "dense batch_size must define exactly cpu, mps, and cuda"
            )
        batch_size = _configuration_integer(
            batch_sizes, device, minimum=1
        )
        args.model_cache.mkdir(parents=True, exist_ok=True)
        artifact_started = perf_counter()
        model_artifact = _prepare_model_artifact(
            baseline_config, args.model_cache
        )
        model_artifact_prepare_seconds = perf_counter() - artifact_started
        encoder = SentenceTransformerTextEncoder(
            str(baseline_config["model_id"]),
            revision=str(baseline_config["model_revision"]),
            device=device,
            batch_size=batch_size,
            max_seq_length=int(baseline_config["max_seq_length"]),
            cache_folder=str(args.model_cache),
        )
        model_started = perf_counter()
        encoder.load()
        model_load_seconds = perf_counter() - model_started
        effective_dimension = encoder.embedding_dimension
        effective_max_seq_length = encoder.effective_max_seq_length
        effective_dtype = encoder.effective_model_dtype
        effective_pooling = encoder.effective_pooling_mode
        has_normalization_module = encoder.has_normalization_module

        _require_configuration_value(
            baseline_config, "embedding_dimension", effective_dimension
        )
        _require_configuration_value(
            baseline_config, "max_seq_length", effective_max_seq_length
        )
        _require_configuration_value(
            baseline_config, "dtype", effective_dtype
        )
        _require_configuration_value(
            baseline_config, "pooling", effective_pooling
        )
        _require_configuration_value(
            baseline_config,
            "normalize_embeddings",
            has_normalization_module,
        )

        encoder_metadata = {
            "embedding_dimension": effective_dimension,
            "effective_max_seq_length": effective_max_seq_length,
            "model_dtype": effective_dtype,
            "pooling": effective_pooling,
            "has_normalization_module": has_normalization_module,
            "batch_size": batch_size,
        }
        effective_configuration = {
            "name": baseline_config["name"],
            "model_id": baseline_config["model_id"],
            "model_revision": baseline_config["model_revision"],
            "model_artifact": model_artifact,
            "embedding_dimension": effective_dimension,
            "pooling": effective_pooling,
            "normalize_embeddings": True,
            "has_normalization_module": has_normalization_module,
            "query_prefix": "",
            "dtype": effective_dtype,
            "max_seq_length": effective_max_seq_length,
            "batch_size": batch_size,
            "sentence_transformers_version": _package_version(
                "sentence-transformers"
            ),
            "transformers_version": _package_version("transformers"),
        }
        index_started = perf_counter()
        index = DenseChunkMaxIndex(
            dataset.documents,
            encoder,
            max_words=max_words,
            overlap=overlap,
        )
        indexing_seconds = perf_counter() - index_started
        chunk_token_lengths = encoder.token_lengths(index.chunks)
        query_token_lengths = encoder.token_lengths(
            tuple(query.text for query in dataset.queries)
        )
        tokenization_analysis = {
            "method": (
                "model tokenizer with special tokens, padding disabled, "
                "and truncation disabled"
            ),
            "document_chunks": _token_length_summary(
                chunk_token_lengths,
                truncation_limit=effective_max_seq_length,
            ),
            "queries": _token_length_summary(
                query_token_lengths,
                truncation_limit=effective_max_seq_length,
            ),
            "warning": (
                "Inputs above the configured limit are truncated by the "
                "encoder; whitespace-word windows do not guarantee complete "
                "subword-token coverage."
            ),
        }

    evaluation = evaluate_retrieval(
        index,
        dataset.queries,
        k=k,
        warmup_count=warmup_queries,
    )
    peak_rss_bytes = _peak_process_rss_bytes()
    num_documents = len(dataset.documents)
    constraints_metadata: Optional[Dict[str, object]]
    if DEFAULT_CONSTRAINTS.exists():
        constraints_metadata = {
            "path": str(DEFAULT_CONSTRAINTS.relative_to(PROJECT_ROOT)),
            "sha256": _sha256_file(DEFAULT_CONSTRAINTS),
        }
    else:
        constraints_metadata = None

    comparability_notes = [
        "Uses revision-pinned page-level Tesseract OCR artifacts.",
        "Uses deterministic fixed whitespace-word windows, not the paper's "
        "Unstructured plus Claude-captioned chunks.",
        "Local timings are not comparable to the paper's NVIDIA L4 timings.",
        "Recall@5 and MRR@5 are project extensions; the paper's headline "
        "retrieval metric is nDCG@5.",
    ]
    if args.baseline == "dense":
        comparability_notes.append(
            "The dense encoder runs on the selected accelerator, while "
            "NumPy similarity aggregation and PyTorch ranking run on CPU."
        )
        if (
            tokenization_analysis is not None
            and tokenization_analysis["document_chunks"][
                "count_exceeding_limit"
            ]
        ):
            comparability_notes.append(
                "Some fixed word windows exceed the 512-token encoder limit; "
                "the exact count is recorded in tokenization_analysis."
            )

    record: Dict[str, object] = {
        "schema_version": 2,
        "result_kind": "measured",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": __version__,
        "seed": seed,
        "task": {
            "name": dataset.name,
            "split": spec.split,
            "documents": num_documents,
            "queries": len(dataset.queries),
            "qrels": sum(
                len(query.relevant_document_ids) for query in dataset.queries
            ),
            "empty_ocr_documents": sum(
                not document.text.strip() for document in dataset.documents
            ),
            "semantic_content_verification": {
                "expected_fingerprint": spec.expected_content_fingerprint,
                "actual_fingerprint": dataset.fingerprint,
                "verified": (
                    spec.expected_content_fingerprint == dataset.fingerprint
                ),
            },
            "dataset_spec_fingerprint": spec.fingerprint,
            "source_manifest": spec.to_dict(),
            "upstream_file_metadata": {
                "status": "expected_metadata_not_byte_verified",
                "explanation": (
                    "Revision-pinned Parquet content is projected to text and "
                    "verified through the semantic dataset fingerprint; the "
                    "remote LFS objects are not downloaded again solely to "
                    "rehash every complete file."
                ),
            },
        },
        "baseline": {
            "kind": args.baseline,
            "requested_configuration": baseline_config,
            "effective_configuration": effective_configuration,
            "chunking": {
                "requested": chunking,
                "effective": {
                    "unit": "normalized_whitespace_words",
                    "max_words": max_words,
                    "overlap_words": overlap,
                    "page_aggregation": "maximum_chunk_score",
                },
            },
            "num_chunks": len(index.chunks),
            "execution_devices": {
                "encoder": device,
                "similarity_and_page_aggregation": "cpu_numpy",
                "score_validation_and_ranking": "cpu_pytorch",
            },
            "encoder": encoder_metadata,
            "tokenization_analysis": tokenization_analysis,
        },
        "quality_and_query_performance": evaluation.to_dict(),
        "indexing": {
            "dataset_load_seconds": dataset_load_seconds,
            "model_artifact_prepare_and_verification_seconds": (
                model_artifact_prepare_seconds
            ),
            "model_load_seconds": model_load_seconds,
            "indexing_seconds": indexing_seconds,
            "pages_per_second": (
                num_documents / indexing_seconds
                if indexing_seconds > 0.0
                else 0.0
            ),
            "logical_index_payload_bytes": index.index_size_bytes,
            "logical_index_payload_bytes_per_page": (
                index.index_size_bytes / num_documents
            ),
            "payload_definition": (
                "BM25 terms, IDFs, postings, chunk lengths, and page mapping"
                if args.baseline == "bm25"
                else "float32 chunk embeddings and int64 page mapping"
            ),
            "excludes": (
                "Python object overhead, source text, identifiers, model "
                "weights, and runtime allocator overhead"
            ),
        },
        "memory": {
            "peak_process_rss_bytes": peak_rss_bytes,
            "measurement": "process high-water mark for the complete run",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "apple_cpu_brand": apple_cpu_brand,
            "physical_memory_bytes": (
                int(physical_memory) if physical_memory is not None else None
            ),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "numpy",
                    "torch",
                    "pyarrow",
                    "fsspec",
                    "sentence-transformers",
                    "transformers",
                    "huggingface-hub",
                    "tokenizers",
                    "safetensors",
                    "scipy",
                    "scikit-learn",
                    "rank-bm25",
                )
            },
            "pip_check": pip_check,
            "constraints": constraints_metadata,
        },
        "provenance": {
            "invocation": [
                sys.executable,
                str(Path(__file__).resolve()),
                *invocation_arguments,
            ],
            "git": git_state,
            "source_fingerprint_sha256": source_fingerprint,
            "source_file_sha256": source_hashes,
            "baseline_configuration": {
                "path": str(config_path),
                "sha256": config_sha256,
            },
            "dataset_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "semantic_fingerprint_sha256": manifest.fingerprint,
            },
        },
        "comparability": {
            "paper_result": False,
            "notes": comparability_notes,
        },
    }

    if args.output is not None:
        _write_json_atomic(args.output, record, args.overwrite)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
