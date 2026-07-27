#!/usr/bin/env python3
"""Assert that every required CUDA/Triton parity case really executed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


REQUIRED_COUNTS = {
    "test_cuda_kernel_matches_vectorized_reference": 3,
    "test_cuda_kernel_handles_noncontiguous_inputs_and_empty_masks": 1,
    "test_cuda_kernel_propagates_active_nan_across_document_tiles": 1,
}


def inspect_junit(path: Path) -> dict[str, object]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {
            "status": "failed",
            "error": f"cannot parse CUDA JUnit record: {exc}",
            "required_counts": REQUIRED_COUNTS,
        }

    observed = {name: 0 for name in REQUIRED_COUNTS}
    skipped: list[str] = []
    failed: list[str] = []
    for case in root.iter("testcase"):
        name = case.attrib.get("name", "")
        matched = next(
            (
                required
                for required in REQUIRED_COUNTS
                if name.startswith(required)
            ),
            None,
        )
        if matched is None:
            continue
        observed[matched] += 1
        identifier = (
            f"{case.attrib.get('classname', '')}::{name}".lstrip(":")
        )
        if case.find("skipped") is not None:
            skipped.append(identifier)
        if (
            case.find("failure") is not None
            or case.find("error") is not None
        ):
            failed.append(identifier)

    errors = []
    for name, expected in REQUIRED_COUNTS.items():
        if observed[name] != expected:
            errors.append(
                f"{name}: expected {expected} cases, observed {observed[name]}"
            )
    if skipped:
        errors.append(
            "required CUDA cases skipped: " + ", ".join(sorted(skipped))
        )
    if failed:
        errors.append(
            "required CUDA cases failed: " + ", ".join(sorted(failed))
        )
    return {
        "format": "colpali-triton-phase11-cuda-test-assertion-v1",
        "status": "passed" if not errors else "failed",
        "required_counts": REQUIRED_COUNTS,
        "observed_counts": observed,
        "skipped": sorted(skipped),
        "failed": sorted(failed),
        "errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = inspect_junit(args.junit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
