#!/usr/bin/env python3
"""Cross-check Phase 11 tuning, winner, config, and benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    result = json.loads(
        path.read_text(),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(result, dict):
        raise TypeError(f"{path.name} is not a JSON object")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_source(value: object) -> tuple[str | None, bool]:
    if not isinstance(value, dict):
        return None, False
    sources = value.get("sources")
    present = (
        [item for item in sources.values() if item is not None]
        if isinstance(sources, dict)
        else []
    )
    effective = value.get("effective_commit")
    valid = (
        value.get("consistent") is True
        and isinstance(effective, str)
        and bool(present)
        and all(item == effective for item in present)
    )
    return effective if isinstance(effective, str) else None, valid


def validate_result_contract(
    root: Path,
    *,
    source_commit: str,
) -> dict[str, Any]:
    errors = []
    observed: dict[str, Any] = {}
    if not COMMIT_PATTERN.fullmatch(source_commit):
        errors.append("expected source commit is invalid")
    paths = {
        "tuning": root / "triton_tuning.json",
        "winner": root / "tuning_winner.json",
        "derived_config": root / "maxsim_benchmark_tuned_config.json",
        "benchmark": root / "maxsim_cuda_benchmark.json",
    }
    values = {}
    for name, path in paths.items():
        try:
            values[name] = _load(path)
            observed[f"{name}_sha256"] = _sha256(path)
        except OSError as exc:
            errors.append(
                f"{name}: OSError errno={getattr(exc, 'errno', None)}"
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if len(values) != len(paths):
        return {
            "format": "colpali-triton-phase11-result-contract-v1",
            "status": "failed",
            "source_commit": source_commit,
            "errors": errors,
            "observed": observed,
        }

    tuning = values["tuning"]
    winner = values["winner"]
    derived = values["derived_config"]
    benchmark = values["benchmark"]
    winner_id = winner.get("candidate_id")
    launch = winner.get("launch_configuration")
    if winner.get("format") != (
        "colpali-triton-phase11-tuning-winner-v1"
    ):
        errors.append("winner record format is invalid")

    tuning_source, tuning_source_valid = _effective_source(
        tuning.get("provenance", {}).get("source_commit")
        if isinstance(tuning.get("provenance"), dict)
        else None
    )
    tuning_winner = tuning.get("winner")
    tuning_configuration = (
        tuning_winner.get("configuration")
        if isinstance(tuning_winner, dict)
        else None
    )
    tuning_launch = (
        {
            key: tuning_configuration.get(key)
            for key in launch
        }
        if isinstance(tuning_configuration, dict)
        and isinstance(launch, dict)
        else None
    )
    if tuning.get("status") != "completed":
        errors.append("tuning status is not completed")
    if not isinstance(tuning.get("selection"), dict) or (
        tuning["selection"].get("canonical_full_matrix") is not True
    ):
        errors.append("tuning matrix is not canonical")
    if not tuning_source_valid or tuning_source != source_commit:
        errors.append("tuning source provenance does not match")
    if not isinstance(tuning_winner, dict) or (
        tuning_winner.get("candidate_id") != winner_id
    ):
        errors.append("tuning winner ID does not match winner record")
    if tuning_launch != launch:
        errors.append("tuning launch does not match winner record")
    if winner.get("source_commit") != source_commit:
        errors.append("winner source commit does not match")
    if winner.get("tuning_result_sha256") != observed["tuning_sha256"]:
        errors.append("winner tuning result hash does not match")
    if winner.get("derived_benchmark_config_sha256") != observed[
        "derived_config_sha256"
    ]:
        errors.append("winner derived config hash does not match")
    if derived.get("triton_launch") != launch:
        errors.append("derived config launch does not match winner")
    if (
        not isinstance(derived.get("cases"), list)
        or len(derived["cases"]) != 6
        or not isinstance(derived.get("dtypes"), list)
        or len(derived["dtypes"]) != 2
        or derived.get("providers") != ["torch", "triton"]
    ):
        errors.append("derived benchmark is not the canonical full matrix")

    benchmark_source, benchmark_source_valid = _effective_source(
        benchmark.get("provenance", {}).get("source_commit")
        if isinstance(benchmark.get("provenance"), dict)
        else None
    )
    benchmark_protocol = benchmark.get("protocol")
    benchmark_config = (
        benchmark_protocol.get("config")
        if isinstance(benchmark_protocol, dict)
        else None
    )
    benchmark_launch = (
        benchmark_config.get("triton_launch")
        if isinstance(benchmark_config, dict)
        else None
    )
    if benchmark.get("status") != "completed":
        errors.append("benchmark status is not completed")
    if not isinstance(benchmark.get("selection"), dict) or (
        benchmark["selection"].get("canonical_full_matrix") is not True
    ):
        errors.append("benchmark selection is not canonical")
    if not benchmark_source_valid or benchmark_source != source_commit:
        errors.append("benchmark source provenance does not match")
    if benchmark_launch != launch:
        errors.append("benchmark launch does not match tuning winner")
    if (
        not isinstance(benchmark.get("results"), list)
        or len(benchmark["results"]) != 12
    ):
        errors.append("benchmark does not contain all 12 case/dtype results")
    benchmark_provenance = benchmark.get("provenance")
    if not isinstance(benchmark_provenance, dict) or (
        benchmark_provenance.get("config_file_sha256")
        != observed["derived_config_sha256"]
    ):
        errors.append("benchmark config hash does not match derived config")

    observed.update(
        {
            "winner_candidate_id": winner_id,
            "winner_launch_configuration": launch,
            "tuning_source_commit": tuning_source,
            "benchmark_source_commit": benchmark_source,
            "benchmark_result_count": (
                len(benchmark["results"])
                if isinstance(benchmark.get("results"), list)
                else None
            ),
        }
    )
    return {
        "format": "colpali-triton-phase11-result-contract-v1",
        "status": "passed" if not errors else "failed",
        "source_commit": source_commit,
        "errors": errors,
        "observed": observed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = validate_result_contract(
        args.result_directory,
        source_commit=args.source_commit,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
