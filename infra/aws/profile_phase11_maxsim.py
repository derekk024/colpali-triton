#!/usr/bin/env python3
"""Execute one representative Triton MaxSim call for an external profiler."""

from __future__ import annotations

import json
import platform
import time

import torch

from colpali_triton.triton_maxsim import (
    DEFAULT_TRITON_MAXSIM_CONFIG,
    maxsim_triton,
)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the Phase 11 profiling probe requires CUDA")

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
            config=DEFAULT_TRITON_MAXSIM_CONFIG,
        )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(
        json.dumps(
            {
                "format": "colpali-triton-phase11-profile-probe-v1",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "dtype": "float16",
                "query_shape": list(queries.shape),
                "document_shape": list(documents.shape),
                "output_shape": list(scores.shape),
                "output_sum": float(scores.float().sum().item()),
                "wall_time_ms_including_cold_compile": elapsed_ms,
                "triton_launch": {
                    "block_query_tokens": (
                        DEFAULT_TRITON_MAXSIM_CONFIG.block_query_tokens
                    ),
                    "block_document_tokens": (
                        DEFAULT_TRITON_MAXSIM_CONFIG.block_document_tokens
                    ),
                    "block_embedding_dimension": (
                        DEFAULT_TRITON_MAXSIM_CONFIG
                        .block_embedding_dimension
                    ),
                    "num_warps": DEFAULT_TRITON_MAXSIM_CONFIG.num_warps,
                    "num_stages": DEFAULT_TRITON_MAXSIM_CONFIG.num_stages,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
