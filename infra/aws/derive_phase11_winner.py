#!/usr/bin/env python3
"""Derive the canonical Phase 11 benchmark/profile contract from tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from colpali_triton.benchmarking import (
    BenchmarkConfig,
    TritonLaunchSpec,
)


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LAUNCH_KEYS = (
    "block_query_tokens",
    "block_document_tokens",
    "block_embedding_dimension",
    "num_warps",
    "num_stages",
)


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    value = json.loads(
        path.read_text(),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_winner_contract(
    tuning: dict[str, Any],
    base_benchmark: dict[str, Any],
    *,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not COMMIT_PATTERN.fullmatch(source_commit):
        raise ValueError("source_commit must be 40 lowercase hexadecimal digits")
    if tuning.get("status") != "completed":
        raise ValueError("tuning result did not complete with a winner")
    selection = tuning.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("canonical_full_matrix") is not True
    ):
        raise ValueError("tuning result is not the canonical full matrix")
    provenance = tuning.get("provenance")
    source_record = (
        provenance.get("source_commit")
        if isinstance(provenance, dict)
        else None
    )
    sources = (
        source_record.get("sources")
        if isinstance(source_record, dict)
        else None
    )
    present_sources = (
        [value for value in sources.values() if value is not None]
        if isinstance(sources, dict)
        else []
    )
    if (
        not isinstance(source_record, dict)
        or source_record.get("effective_commit") != source_commit
        or source_record.get("consistent") is not True
        or not isinstance(sources, dict)
        or not present_sources
        or any(value != source_commit for value in present_sources)
    ):
        raise ValueError("tuning source commit does not match deployed source")
    winner = tuning.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("tuning result lacks a winner")
    candidate_id = winner.get("candidate_id")
    configuration = winner.get("configuration")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("tuning winner candidate ID is invalid")
    if not isinstance(configuration, dict) or (
        configuration.get("candidate_id") != candidate_id
    ):
        raise ValueError("tuning winner configuration is invalid")
    overall = winner.get("overall")
    if (
        not isinstance(overall, dict)
        or overall.get("status") != "selected"
        or overall.get("winner_candidate_id") != candidate_id
    ):
        raise ValueError("tuning winner selection record is inconsistent")
    launch = TritonLaunchSpec.from_mapping(
        {key: configuration.get(key) for key in LAUNCH_KEYS}
    ).to_dict()

    required_base = {
        "schema_version",
        "benchmark_id",
        "seed",
        "device",
        "timing",
        "providers",
        "triton_launch",
        "dtypes",
        "cases",
        "normalize_embeddings",
    }
    if set(base_benchmark) != required_base:
        raise ValueError("base benchmark keys differ from the contract")
    if base_benchmark.get("schema_version") != 1:
        raise ValueError("base benchmark schema version is unsupported")
    BenchmarkConfig.from_mapping(base_benchmark)
    providers = base_benchmark.get("providers")
    cases = base_benchmark.get("cases")
    dtypes = base_benchmark.get("dtypes")
    if providers != ["torch", "triton"]:
        raise ValueError("base benchmark must compare torch then triton")
    if not isinstance(cases, list) or len(cases) != 6:
        raise ValueError("base benchmark must retain all six canonical cases")
    if not isinstance(dtypes, list) or len(dtypes) != 2:
        raise ValueError("base benchmark must retain both canonical dtypes")

    derived = json.loads(json.dumps(base_benchmark))
    derived["benchmark_id"] = (
        f"{base_benchmark['benchmark_id']}-tuned-{candidate_id}"
    )
    derived["triton_launch"] = launch
    BenchmarkConfig.from_mapping(derived)
    record = {
        "format": "colpali-triton-phase11-tuning-winner-v1",
        "source_commit": source_commit,
        "candidate_id": candidate_id,
        "launch_configuration": launch,
        "base_benchmark_id": base_benchmark["benchmark_id"],
        "derived_benchmark_id": derived["benchmark_id"],
        "canonical_case_count": len(cases),
        "canonical_dtype_count": len(dtypes),
    }
    return derived, record


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-result", type=Path, required=True)
    parser.add_argument("--base-benchmark-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--winner-record", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tuning = _load(args.tuning_result)
    base = _load(args.base_benchmark_config)
    derived, winner = derive_winner_contract(
        tuning,
        base,
        source_commit=args.source_commit,
    )
    winner.update(
        {
            "tuning_result_sha256": _sha256(args.tuning_result),
            "base_benchmark_config_sha256": _sha256(
                args.base_benchmark_config
            ),
        }
    )
    _write_exclusive(args.output_config, derived)
    winner["derived_benchmark_config_sha256"] = _sha256(
        args.output_config
    )
    _write_exclusive(args.winner_record, winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
