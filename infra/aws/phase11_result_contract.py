#!/usr/bin/env python3
"""Cross-check Phase 11 tuning, winner, config, and benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Sequence


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TUNING_CONFIG = (
    PROJECT_ROOT / "configs" / "triton_tuning.json"
)
CANONICAL_BENCHMARK_CONFIG = (
    PROJECT_ROOT / "configs" / "maxsim_benchmark.json"
)
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


def _is_finite_number(value: object, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and (number > 0.0 if positive else True)


def _validate_memory_record(
    value: object,
    *,
    context: str,
    errors: list[str],
) -> None:
    required = (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    )
    if not isinstance(value, dict) or any(
        not isinstance(value.get(key), int)
        or isinstance(value.get(key), bool)
        or value[key] < 0
        for key in required
    ):
        errors.append(f"{context} memory record is invalid")


def _validated_timing(
    value: object,
    *,
    expected_trials: int,
    expected_iterations: int,
    tuning: bool,
    context: str,
    errors: list[str],
) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        errors.append(f"{context} timing record is missing")
        return None
    trial_count_key = "trial_count_completed" if tuning else "trial_count"
    if (
        value.get("measured_iterations_per_trial") != expected_iterations
        or value.get(trial_count_key) != expected_trials
        or (
            tuning
            and value.get("trial_count_requested") != expected_trials
        )
    ):
        errors.append(f"{context} timing dimensions do not match protocol")
        return None
    trials = value.get("trials")
    if not isinstance(trials, list) or len(trials) != expected_trials:
        errors.append(f"{context} does not retain every timing trial")
        return None

    samples: list[float] = []
    valid = True
    for trial_index, trial in enumerate(trials):
        if (
            not isinstance(trial, dict)
            or trial.get("trial_index") != trial_index
            or (tuning and trial.get("status") != "completed")
        ):
            valid = False
            break
        raw_samples = trial.get("cuda_event_latency_ms")
        if (
            not isinstance(raw_samples, list)
            or len(raw_samples) != expected_iterations
            or not all(
                _is_finite_number(item, positive=True)
                for item in raw_samples
            )
        ):
            valid = False
            break
        samples.extend(float(item) for item in raw_samples)
        _validate_memory_record(
            trial.get("memory"),
            context=f"{context} trial {trial_index}",
            errors=errors,
        )
    if not valid:
        errors.append(f"{context} timing samples are invalid")
        return None

    summary = value.get("summary")
    latency = summary.get("latency_ms") if isinstance(summary, dict) else None
    expected_sample_count = expected_trials * expected_iterations
    if (
        not isinstance(summary, dict)
        or summary.get("sample_count") != expected_sample_count
        or not isinstance(latency, dict)
        or not _is_finite_number(latency.get("p50"), positive=True)
    ):
        errors.append(f"{context} timing summary is invalid")
        return None
    pooled_p50 = median(samples)
    if not math.isclose(
        float(latency["p50"]), pooled_p50, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(f"{context} p50 does not match retained samples")
        return None
    total = sum(samples)
    if tuning and (
        not _is_finite_number(
            value.get("total_cuda_event_latency_ms"), positive=True
        )
        or not math.isclose(
            float(value["total_cuda_event_latency_ms"]),
            total,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        errors.append(
            f"{context} total latency does not match retained samples"
        )
        return None
    return pooled_p50, total


def _case_record(case: dict[str, Any]) -> dict[str, Any]:
    result = dict(case)
    result["query_document_pairs"] = (
        case["query_batch_size"] * case["document_batch_size"]
    )
    return result


def _validate_tuning_evidence(
    tuning: dict[str, Any],
    *,
    canonical: dict[str, Any],
    winner_id: object,
    launch: object,
    errors: list[str],
) -> None:
    if not isinstance(winner_id, str) or not isinstance(launch, dict):
        return
    protocol = tuning.get("protocol")
    protocol_config = (
        protocol.get("config") if isinstance(protocol, dict) else None
    )
    if protocol_config != canonical:
        errors.append("tuning protocol is not the committed canonical config")
        return

    candidates = canonical["candidates"]
    dtypes = canonical["dtypes"]
    cases = canonical["cases"]
    candidate_ids = [item["candidate_id"] for item in candidates]
    dtype_names = [item["name"] for item in dtypes]
    case_names = [item["name"] for item in cases]
    selection = tuning.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("candidates") != candidate_ids
        or selection.get("dtypes") != dtype_names
        or selection.get("cases") != case_names
    ):
        errors.append("tuning selection does not match canonical config")

    provenance = tuning.get("provenance")
    source_hashes = (
        provenance.get("source_sha256")
        if isinstance(provenance, dict)
        else None
    )
    canonical_sha = _sha256(CANONICAL_TUNING_CONFIG)
    if (
        not isinstance(provenance, dict)
        or provenance.get("config_file_sha256") != canonical_sha
        or not isinstance(source_hashes, dict)
        or source_hashes.get("configs/triton_tuning.json")
        != canonical_sha
    ):
        errors.append("tuning config provenance is not canonical")

    tuning_winner = tuning.get("winner")
    configuration = (
        tuning_winner.get("configuration")
        if isinstance(tuning_winner, dict)
        else None
    )
    expected_candidate = next(
        (
            item
            for item in candidates
            if item["candidate_id"] == winner_id
        ),
        None,
    )
    if configuration != expected_candidate:
        errors.append("tuning winner configuration is not canonical")
    expected_launch = (
        {key: expected_candidate[key] for key in LAUNCH_KEYS}
        if isinstance(expected_candidate, dict)
        else None
    )
    if expected_launch != launch:
        errors.append("winner launch differs from canonical candidate")

    results = tuning.get("results")
    expected_workloads = [
        (dtype_name, case["name"])
        for dtype_name in dtype_names
        for case in cases
    ]
    if not isinstance(results, list) or len(results) != len(
        expected_workloads
    ):
        errors.append("tuning does not contain every canonical workload")
        return

    candidate_by_id = {
        item["candidate_id"]: item for item in candidates
    }
    relative_by_candidate: dict[str, list[float]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    total_by_candidate = {candidate_id: 0.0 for candidate_id in candidate_ids}
    completed_by_candidate = {candidate_id: 0 for candidate_id in candidate_ids}
    observed_workloads: list[tuple[object, object]] = []
    winner_completed_every_workload = True
    timing_config = canonical["timing"]

    for workload_index, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append("tuning workload record is invalid")
            return
        case_value = result.get("case")
        case_name = (
            case_value.get("name") if isinstance(case_value, dict) else None
        )
        dtype_name = result.get("dtype")
        observed_workloads.append((dtype_name, case_name))
        expected_case = cases[case_names.index(case_name)] if (
            case_name in case_names
        ) else None
        if (
            expected_case is None
            or case_value != _case_record(expected_case)
            or dtype_name not in dtype_names
        ):
            errors.append(
                f"tuning workload {workload_index} shape/dtype is invalid"
            )
            continue

        records = result.get("candidates")
        if not isinstance(records, list) or [
            item.get("candidate_id") if isinstance(item, dict) else None
            for item in records
        ] != candidate_ids:
            errors.append(
                f"tuning workload {workload_index} candidate matrix is invalid"
            )
            continue

        latencies: dict[str, float] = {}
        totals: dict[str, float] = {}
        for record in records:
            assert isinstance(record, dict)
            candidate_id = record["candidate_id"]
            canonical_candidate = candidate_by_id[candidate_id]
            canonical_launch = {
                key: canonical_candidate[key] for key in LAUNCH_KEYS
            }
            if record.get("launch_configuration") != canonical_launch:
                errors.append(
                    f"tuning candidate {candidate_id} launch is inconsistent"
                )
            if record.get("status") != "completed":
                if candidate_id == winner_id:
                    winner_completed_every_workload = False
                continue
            numerical_gate = record.get("numerical_gate")
            agreement = (
                numerical_gate.get("agreement")
                if isinstance(numerical_gate, dict)
                else None
            )
            if (
                not isinstance(numerical_gate, dict)
                or numerical_gate.get("status") != "passed"
                or not isinstance(agreement, dict)
                or agreement.get("shape_matches") is not True
                or agreement.get("finite") is not True
                or agreement.get("agreement") is not True
                or agreement.get("reference_provider") != "torch"
            ):
                errors.append(
                    f"tuning candidate {candidate_id} lacks numerical parity"
                )
                continue
            validated = _validated_timing(
                record.get("timing"),
                expected_trials=timing_config["trials"],
                expected_iterations=timing_config["measured_iterations"],
                tuning=True,
                context=(
                    f"tuning workload {workload_index} candidate "
                    f"{candidate_id}"
                ),
                errors=errors,
            )
            if validated is None:
                continue
            latencies[candidate_id], totals[candidate_id] = validated

        if winner_id not in latencies:
            winner_completed_every_workload = False
        if not latencies:
            errors.append(
                f"tuning workload {workload_index} has no valid candidate"
            )
            continue
        workload_winner_latency = min(
            latencies.items(), key=lambda item: (item[1], item[0])
        )[1]
        for candidate_id, latency in latencies.items():
            relative_by_candidate[candidate_id].append(
                latency / workload_winner_latency
            )
            total_by_candidate[candidate_id] += totals[candidate_id]
            completed_by_candidate[candidate_id] += 1

    if observed_workloads != expected_workloads:
        errors.append("tuning workload order/identity is not canonical")
    if not winner_completed_every_workload:
        errors.append("tuning winner did not pass every workload")

    eligible = [
        candidate_id
        for candidate_id in candidate_ids
        if completed_by_candidate[candidate_id] == len(expected_workloads)
    ]
    recomputed_winner = (
        min(
            eligible,
            key=lambda candidate_id: (
                median(relative_by_candidate[candidate_id]),
                total_by_candidate[candidate_id],
                candidate_id,
            ),
        )
        if eligible
        else None
    )
    overall = (
        tuning_winner.get("overall")
        if isinstance(tuning_winner, dict)
        else None
    )
    if recomputed_winner != winner_id:
        errors.append("claimed tuning winner is not the measured winner")
    if (
        not isinstance(overall, dict)
        or overall.get("status") != "selected"
        or overall.get("winner_candidate_id") != winner_id
        or overall.get("workload_count") != len(expected_workloads)
    ):
        errors.append("tuning winner selection record is inconsistent")


def _validate_benchmark_evidence(
    benchmark: dict[str, Any],
    *,
    derived: dict[str, Any],
    canonical_base: dict[str, Any],
    winner_id: object,
    launch: object,
    errors: list[str],
) -> None:
    expected_derived = json.loads(json.dumps(canonical_base))
    expected_derived["benchmark_id"] = (
        f"{canonical_base['benchmark_id']}-tuned-{winner_id}"
    )
    expected_derived["triton_launch"] = launch
    if derived != expected_derived:
        errors.append(
            "derived benchmark differs from canonical base plus winner launch"
        )

    protocol = benchmark.get("protocol")
    benchmark_config = (
        protocol.get("config") if isinstance(protocol, dict) else None
    )
    if benchmark_config != derived:
        errors.append("benchmark protocol config differs from derived config")

    cases = canonical_base["cases"]
    dtypes = canonical_base["dtypes"]
    case_names = [item["name"] for item in cases]
    dtype_names = [item["name"] for item in dtypes]
    selection = benchmark.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("providers") != ["torch", "triton"]
        or selection.get("dtypes") != dtype_names
        or selection.get("cases") != case_names
    ):
        errors.append("benchmark selection does not match canonical config")

    results = benchmark.get("results")
    expected_workloads = [
        (dtype_name, case["name"])
        for dtype_name in dtype_names
        for case in cases
    ]
    if not isinstance(results, list) or len(results) != len(
        expected_workloads
    ):
        errors.append("benchmark does not contain every canonical workload")
        return

    dtype_by_name = {item["name"]: item for item in dtypes}
    timing_config = canonical_base["timing"]
    observed_workloads: list[tuple[object, object]] = []
    for workload_index, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append("benchmark workload record is invalid")
            continue
        case_value = result.get("case")
        case_name = (
            case_value.get("name") if isinstance(case_value, dict) else None
        )
        dtype_name = result.get("dtype")
        observed_workloads.append((dtype_name, case_name))
        expected_case = cases[case_names.index(case_name)] if (
            case_name in case_names
        ) else None
        if (
            expected_case is None
            or case_value != _case_record(expected_case)
            or dtype_name not in dtype_names
        ):
            errors.append(
                f"benchmark workload {workload_index} shape/dtype is invalid"
            )
            continue
        dtype_spec = dtype_by_name[dtype_name]
        providers = result.get("providers")
        if not isinstance(providers, list) or [
            item.get("provider") if isinstance(item, dict) else None
            for item in providers
        ] != ["torch", "triton"]:
            errors.append(
                f"benchmark workload {workload_index} provider matrix is invalid"
            )
            continue
        for provider in providers:
            assert isinstance(provider, dict)
            provider_name = provider["provider"]
            expected_launch = launch if provider_name == "triton" else None
            correctness = provider.get("correctness")
            if (
                provider.get("launch_configuration") != expected_launch
                or not isinstance(correctness, dict)
                or correctness.get("reference_provider") != "torch"
                or correctness.get("finite") is not True
                or correctness.get("agreement") is not True
                or correctness.get("absolute_tolerance")
                != dtype_spec["absolute_tolerance"]
                or correctness.get("relative_tolerance")
                != dtype_spec["relative_tolerance"]
            ):
                errors.append(
                    f"benchmark workload {workload_index} provider "
                    f"{provider_name} lacks numerical parity"
                )
            _validated_timing(
                provider.get("timing"),
                expected_trials=timing_config["trials"],
                expected_iterations=timing_config["measured_iterations"],
                tuning=False,
                context=(
                    f"benchmark workload {workload_index} provider "
                    f"{provider_name}"
                ),
                errors=errors,
            )
            memory = provider.get("memory")
            required_memory = (
                "maximum_peak_allocated_bytes",
                "maximum_peak_reserved_bytes",
                "maximum_incremental_peak_allocated_bytes",
                "maximum_incremental_peak_reserved_bytes",
            )
            if not isinstance(memory, dict) or any(
                not isinstance(memory.get(key), int)
                or isinstance(memory.get(key), bool)
                or memory[key] < 0
                for key in required_memory
            ):
                errors.append(
                    f"benchmark workload {workload_index} provider "
                    f"{provider_name} memory summary is invalid"
                )
    if observed_workloads != expected_workloads:
        errors.append("benchmark workload order/identity is not canonical")


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
                f"{name}: {type(exc).__name__} "
                f"errno={getattr(exc, 'errno', None)}"
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
    try:
        canonical_tuning = _load(CANONICAL_TUNING_CONFIG)
        canonical_benchmark = _load(CANONICAL_BENCHMARK_CONFIG)
        observed["canonical_tuning_config_sha256"] = _sha256(
            CANONICAL_TUNING_CONFIG
        )
        observed["canonical_benchmark_config_sha256"] = _sha256(
            CANONICAL_BENCHMARK_CONFIG
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(
            "canonical configs: "
            f"{type(exc).__name__}: {exc}"
        )
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
    if not isinstance(winner_id, str) or not winner_id:
        errors.append("winner candidate ID is invalid")
    if not isinstance(launch, dict) or set(launch) != set(LAUNCH_KEYS):
        errors.append("winner launch configuration is invalid")

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
    if winner.get("base_benchmark_config_sha256") != observed[
        "canonical_benchmark_config_sha256"
    ]:
        errors.append("winner base benchmark config hash does not match")
    if (
        winner.get("canonical_case_count") != 6
        or winner.get("canonical_dtype_count") != 2
    ):
        errors.append("winner canonical benchmark counts are invalid")
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
    _validate_tuning_evidence(
        tuning,
        canonical=canonical_tuning,
        winner_id=winner_id,
        launch=launch,
        errors=errors,
    )

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
    _validate_benchmark_evidence(
        benchmark,
        derived=derived,
        canonical_base=canonical_benchmark,
        winner_id=winner_id,
        launch=launch,
        errors=errors,
    )

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
