#!/usr/bin/env python3
"""Run the deterministic tiny ColPali objective overfit check."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from colpali_triton import (
    SyntheticOverfitConfig,
    SyntheticOverfitResult,
    run_synthetic_overfit,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "synthetic_overfit.json"
)


def _load_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")

    experiment = payload.get("experiment")
    acceptance = payload.get("acceptance")
    if not isinstance(experiment, dict) or not isinstance(acceptance, dict):
        raise ValueError(
            "configuration requires object-valued experiment and acceptance"
        )
    return experiment, acceptance


def _passes(
    result: SyntheticOverfitResult, acceptance: Dict[str, Any]
) -> bool:
    return (
        result.loss_reduction_fraction
        >= float(acceptance["minimum_loss_reduction_fraction"])
        and result.final_top1_accuracy
        >= float(acceptance["minimum_final_top1_accuracy"])
        and result.minimum_positive_margin
        >= float(acceptance["minimum_positive_margin"])
        and result.initial_query_gradient_norm
        > float(acceptance["minimum_initial_gradient_norm"])
        and result.initial_document_gradient_norm
        > float(acceptance["minimum_initial_gradient_norm"])
        and result.maximum_padding_gradient
        <= float(acceptance["maximum_padding_gradient"])
        and result.maximum_padding_parameter_change
        <= float(acceptance["maximum_padding_parameter_change"])
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON experiment configuration",
    )
    arguments = parser.parse_args(argv)

    experiment, acceptance = _load_config(arguments.config)
    config = SyntheticOverfitConfig(**experiment)
    started_at = time.perf_counter()
    result = run_synthetic_overfit(config)
    duration_seconds = time.perf_counter() - started_at
    output = asdict(result)
    output["duration_seconds"] = duration_seconds
    output["overfit_passed"] = _passes(result, acceptance)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["overfit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
