#!/usr/bin/env python3
"""Execute one representative Triton MaxSim call for an external profiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from colpali_triton.triton_maxsim import (
    TritonMaxSimConfig,
    maxsim_triton,
)


LAUNCH_KEYS = (
    "block_query_tokens",
    "block_document_tokens",
    "block_embedding_dimension",
    "num_warps",
    "num_stages",
)


def _load_winner(path: Path) -> tuple[dict[str, object], TritonMaxSimConfig]:
    winner = json.loads(path.read_text())
    if not isinstance(winner, dict) or winner.get("format") != (
        "colpali-triton-phase11-tuning-winner-v1"
    ):
        raise ValueError("invalid Phase 11 tuning winner record")
    source_commit = os.environ.get("COLPALI_SOURCE_COMMIT")
    if winner.get("source_commit") != source_commit:
        raise ValueError("winner source commit differs from container source")
    launch = winner.get("launch_configuration")
    if not isinstance(launch, dict) or set(launch) != set(LAUNCH_KEYS):
        raise ValueError("winner launch configuration is invalid")
    return winner, TritonMaxSimConfig(
        **{key: launch[key] for key in LAUNCH_KEYS}
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-record", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the Phase 11 profiling probe requires CUDA")
    winner, launch_config = _load_winner(args.winner_record)

    device = torch.device("cuda", 0)
    # Construct and normalize on CPU so the first launched CUDA kernel under
    # ``ncu --launch-count 1`` is the Triton MaxSim kernel, not an input
    # generation, normalization, or mask-fill kernel.
    generator = torch.Generator().manual_seed(20250725)
    queries_cpu = torch.randn(
        (1, 15, 128),
        dtype=torch.float16,
        generator=generator,
    )
    documents_cpu = torch.randn(
        (32, 1030, 128),
        dtype=torch.float16,
        generator=generator,
    )
    queries = torch.nn.functional.normalize(queries_cpu, dim=-1).to(device)
    documents = torch.nn.functional.normalize(
        documents_cpu, dim=-1
    ).to(device)
    query_mask = torch.ones((1, 15), dtype=torch.bool).to(device)
    document_mask = torch.ones((32, 1030), dtype=torch.bool).to(device)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        scores = maxsim_triton(
            queries,
            documents,
            query_mask=query_mask,
            document_mask=document_mask,
            config=launch_config,
        )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    winner_sha = hashlib.sha256(args.winner_record.read_bytes()).hexdigest()
    record = {
        "format": "colpali-triton-phase11-profile-probe-v2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "source_commit": winner["source_commit"],
        "winner_candidate_id": winner["candidate_id"],
        "winner_record_sha256": winner_sha,
        "launch_configuration": {
            key: getattr(launch_config, key) for key in LAUNCH_KEYS
        },
        "dtype": "float16",
        "query_shape": list(queries.shape),
        "document_shape": list(documents.shape),
        "output_shape": list(scores.shape),
        "output_sum": float(scores.float().sum().item()),
        "wall_time_ms_including_cold_compile": elapsed_ms,
    }
    with args.metadata_output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
