#!/usr/bin/env python3
"""Require the exact NVIDIA L4 hardware contract before paid Phase 11 work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


MINIMUM_MEMORY_BYTES = 21 * 1024**3
MAXIMUM_MEMORY_BYTES = 25 * 1024**3


def inspect_cuda_l4() -> dict[str, object]:
    blockers = []
    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if available else 0
    device = None
    if not available:
        blockers.append("torch.cuda.is_available() is false")
    elif device_count < 1:
        blockers.append("CUDA reports no devices")
    else:
        properties = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        total_memory = int(properties.total_memory)
        capability = [int(properties.major), int(properties.minor)]
        device = {
            "index": 0,
            "name": name,
            "compute_capability": capability,
            "total_memory_bytes": total_memory,
        }
        if name.strip() != "NVIDIA L4":
            blockers.append(f"cuda:0 is {name!r}, expected 'NVIDIA L4'")
        if capability != [8, 9]:
            blockers.append(
                f"cuda:0 compute capability is {capability}, expected [8, 9]"
            )
        if not MINIMUM_MEMORY_BYTES <= total_memory <= MAXIMUM_MEMORY_BYTES:
            blockers.append(
                "cuda:0 memory is outside the 21-25 GiB full-L4 range: "
                f"{total_memory} bytes"
            )
    return {
        "format": "colpali-triton-phase11-l4-hardware-gate-v1",
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "cuda_available": available,
        "device_count": device_count,
        "required": {
            "name": "NVIDIA L4",
            "compute_capability": [8, 9],
            "minimum_memory_bytes": MINIMUM_MEMORY_BYTES,
            "maximum_memory_bytes": MAXIMUM_MEMORY_BYTES,
        },
        "device": device,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = inspect_cuda_l4()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
