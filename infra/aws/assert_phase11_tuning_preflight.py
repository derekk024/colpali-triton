#!/usr/bin/env python3
"""Require the complete ready-to-tune Phase 11 preflight contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "candidates": 12,
    "dtypes": 2,
    "cases": 4,
}
IDENTIFIER_FIELDS = {
    "candidates": "candidate_id",
    "dtypes": "name",
    "cases": "name",
}
EXPECTED_CONFIG_CLASS = (
    "colpali_triton.triton_maxsim.TritonMaxSimConfig"
)


def _load_mapping(
    path: Path,
    *,
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot parse {label}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def _expected_identifiers(
    config: dict[str, Any],
    *,
    kind: str,
    errors: list[str],
) -> list[str]:
    raw_items = config.get(kind)
    expected_count = EXPECTED_COUNTS[kind]
    identifier_field = IDENTIFIER_FIELDS[kind]
    if not isinstance(raw_items, list):
        errors.append(f"config {kind} must be a JSON array")
        return []
    if len(raw_items) != expected_count:
        errors.append(
            f"config must define exactly {expected_count} {kind}; "
            f"observed {len(raw_items)}"
        )
    identifiers = []
    for index, item in enumerate(raw_items):
        identifier = (
            item.get(identifier_field) if isinstance(item, dict) else None
        )
        if not isinstance(identifier, str) or not identifier:
            errors.append(
                f"config {kind}[{index}].{identifier_field} is invalid"
            )
            continue
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        errors.append(f"config {kind} identifiers must be unique")
    return identifiers


def inspect_tuning_preflight(
    preflight_path: Path,
    *,
    config_path: Path,
) -> dict[str, object]:
    errors: list[str] = []
    config = _load_mapping(
        config_path,
        label="tuning config",
        errors=errors,
    )
    record = _load_mapping(
        preflight_path,
        label="tuning preflight",
        errors=errors,
    )

    expected = {
        kind: _expected_identifiers(config, kind=kind, errors=errors)
        for kind in EXPECTED_COUNTS
    }
    tuning_id = config.get("tuning_id")
    if not isinstance(tuning_id, str) or not tuning_id:
        errors.append("config tuning_id is invalid")
    if record.get("tuning_id") != tuning_id:
        errors.append("preflight tuning_id does not match the tuning config")

    try:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as exc:
        config_sha256 = None
        errors.append(f"cannot hash tuning config: {exc}")
    if record.get("config_file_sha256") != config_sha256:
        errors.append("preflight config file hash does not match")

    semantic_sha256 = record.get("semantic_config_sha256")
    if (
        not isinstance(semantic_sha256, str)
        or len(semantic_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in semantic_sha256
        )
    ):
        errors.append("preflight semantic config hash is invalid")

    selection = record.get("selection")
    if not isinstance(selection, dict):
        errors.append("preflight selection must be a JSON object")
        selection = {}
    observed_selection: dict[str, object] = {}
    for kind, expected_identifiers in expected.items():
        observed = selection.get(kind)
        observed_selection[kind] = observed
        if observed != expected_identifiers:
            errors.append(
                f"preflight must select the exact {len(expected_identifiers)} "
                f"configured {kind}"
            )

    preflight = record.get("preflight")
    if not isinstance(preflight, dict):
        errors.append("preflight result must be a JSON object")
        preflight = {}
    if preflight.get("status") != "ready":
        errors.append("preflight status is not ready")
    if preflight.get("blockers") != []:
        errors.append("preflight blockers must be empty")
    if preflight.get("candidate_count") != EXPECTED_COUNTS["candidates"]:
        errors.append("preflight candidate_count must be exactly 12")

    candidate_validation = preflight.get("candidate_validation")
    if not isinstance(candidate_validation, dict):
        errors.append("preflight candidate_validation must be an object")
        candidate_validation = {}
    if candidate_validation.get("valid") is not True:
        errors.append("candidate validation did not pass")
    if (
        candidate_validation.get("validated_candidate_count")
        != EXPECTED_COUNTS["candidates"]
    ):
        errors.append("candidate validation must cover exactly 12 candidates")
    if candidate_validation.get("class") != EXPECTED_CONFIG_CLASS:
        errors.append("candidate validation used an unexpected config class")
    if candidate_validation.get("error") is not None:
        errors.append("candidate validation reported an error")

    cuda = preflight.get("cuda")
    if not isinstance(cuda, dict):
        errors.append("preflight cuda result must be an object")
        cuda = {}
    for field in (
        "pytorch_build_includes_cuda",
        "available",
        "configured_device_in_range",
    ):
        if cuda.get(field) is not True:
            errors.append(f"preflight cuda.{field} must be true")

    triton = preflight.get("triton")
    if not isinstance(triton, dict):
        errors.append("preflight triton result must be an object")
        triton = {}
    if triton.get("available") is not True:
        errors.append("preflight Triton provider is unavailable")
    if triton.get("error") is not None:
        errors.append("preflight Triton provider reported an error")

    return {
        "format": (
            "colpali-triton-phase11-tuning-preflight-assertion-v1"
        ),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "required_counts": EXPECTED_COUNTS,
        "observed_selection": observed_selection,
        "candidate_validation": candidate_validation,
        "config_file_sha256": config_sha256,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = inspect_tuning_preflight(
        args.preflight,
        config_path=args.config,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
