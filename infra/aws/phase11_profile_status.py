#!/usr/bin/env python3
"""Validate bounded Phase 11 profiler attempts and write their status."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


EXPECTED_SUFFIX = {
    "ncu": ".ncu-rep",
    "nsys": ".nsys-rep",
}


def evaluate_profile_attempts(
    result_directory: Path,
    attempts: Sequence[dict[str, Any]],
    *,
    winner_record: Path,
) -> dict[str, Any]:
    winner = None
    winner_sha = None
    winner_error = None
    try:
        winner = json.loads(winner_record.read_text())
        winner_sha = hashlib.sha256(winner_record.read_bytes()).hexdigest()
        if not isinstance(winner, dict) or winner.get("format") != (
            "colpali-triton-phase11-tuning-winner-v1"
        ):
            raise ValueError("unexpected winner record format")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        winner = None
        winner_error = f"{type(exc).__name__}: {exc}"

    normalized = []
    successful_attempt = None
    for attempt in attempts:
        profiler = attempt["profiler"]
        attempted = attempt["attempted"]
        exit_code = attempt["exit_code"] if attempted else None
        report = attempt["report"] or None
        metadata = attempt.get("metadata") or None
        report_valid = False
        if report is not None:
            path = PurePosixPath(report)
            report_valid = (
                not path.is_absolute()
                and ".." not in path.parts
                and report.endswith(EXPECTED_SUFFIX[profiler])
                and (result_directory / report).is_file()
                and (result_directory / report).stat().st_size > 0
            )
        metadata_valid = False
        if metadata is not None and winner is not None:
            metadata_path = PurePosixPath(metadata)
            metadata_is_safe = (
                not metadata_path.is_absolute()
                and ".." not in metadata_path.parts
            )
            if metadata_is_safe:
                try:
                    metadata_record = json.loads(
                        (result_directory / metadata).read_text()
                    )
                except (OSError, json.JSONDecodeError):
                    metadata_record = None
            else:
                metadata_record = None
            metadata_valid = (
                metadata_is_safe
                and isinstance(metadata_record, dict)
                and metadata_record.get("format")
                == "colpali-triton-phase11-profile-probe-v2"
                and metadata_record.get("source_commit")
                == winner.get("source_commit")
                and metadata_record.get("winner_candidate_id")
                == winner.get("candidate_id")
                and metadata_record.get("winner_record_sha256")
                == winner_sha
                and metadata_record.get("launch_configuration")
                == winner.get("launch_configuration")
            )
        succeeded = (
            attempted
            and exit_code == 0
            and report_valid
            and metadata_valid
        )
        record = {
            "profiler": profiler,
            "attempted": attempted,
            "exit_code": exit_code,
            "report": report,
            "report_valid": report_valid,
            "metadata": metadata,
            "metadata_matches_winner": metadata_valid,
            "succeeded": succeeded,
        }
        normalized.append(record)
        if successful_attempt is None and succeeded:
            successful_attempt = record
    return {
        "format": "colpali-triton-phase11-profiling-status-v1",
        "succeeded": successful_attempt is not None,
        "profiler": (
            successful_attempt["profiler"] if successful_attempt else None
        ),
        "report": (
            successful_attempt["report"] if successful_attempt else None
        ),
        "metadata": (
            successful_attempt["metadata"] if successful_attempt else None
        ),
        "winner_record": winner_record.name if winner is not None else None,
        "winner_record_sha256": winner_sha,
        "winner_record_error": winner_error,
        "timeout_seconds_per_attempt": 300,
        "container_capabilities": ["SYS_ADMIN"],
        "privileged_container": False,
        "attempts": normalized,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--winner-record", type=Path, required=True)
    for profiler in ("ncu", "nsys"):
        parser.add_argument(
            f"--{profiler}-attempted",
            type=int,
            choices=(0, 1),
            required=True,
        )
        parser.add_argument(
            f"--{profiler}-exit",
            type=int,
            required=True,
        )
        parser.add_argument(f"--{profiler}-report", required=True)
        parser.add_argument(f"--{profiler}-metadata", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    attempts = [
        {
            "profiler": profiler,
            "attempted": bool(getattr(args, f"{profiler}_attempted")),
            "exit_code": getattr(args, f"{profiler}_exit"),
            "report": getattr(args, f"{profiler}_report"),
            "metadata": getattr(args, f"{profiler}_metadata"),
        }
        for profiler in ("ncu", "nsys")
    ]
    result = evaluate_profile_attempts(
        args.result_directory,
        attempts,
        winner_record=args.winner_record,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
