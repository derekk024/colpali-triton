import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from colpali_triton import (
    SyntheticOverfitConfig,
    run_synthetic_overfit,
)


def test_tiny_synthetic_dataset_is_overfit() -> None:
    result = run_synthetic_overfit()

    assert result.initial_top1_accuracy < 1.0
    assert result.final_top1_accuracy == 1.0
    assert result.loss_reduction_fraction >= 0.85
    assert result.minimum_positive_margin >= 1.5
    assert result.initial_query_gradient_norm > 0.0
    assert result.initial_document_gradient_norm > 0.0
    assert result.maximum_padding_gradient == 0.0
    assert result.maximum_padding_parameter_change == 0.0


def test_synthetic_experiment_is_deterministic() -> None:
    config = replace(SyntheticOverfitConfig(), steps=10)
    expected = run_synthetic_overfit(config)
    original_dtype = torch.get_default_dtype()

    try:
        torch.set_default_dtype(torch.float64)
        actual = run_synthetic_overfit(config)
    finally:
        torch.set_default_dtype(original_dtype)

    assert actual == expected


def test_canonical_config_and_cli_pass() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "synthetic_overfit.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        config_payload = json.load(config_file)

    assert config_payload["experiment"] == asdict(SyntheticOverfitConfig())

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "overfit_synthetic.py"),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["overfit_passed"] is True
    assert result["final_top1_accuracy"] == 1.0
    assert result["loss_reduction_fraction"] >= 0.85
    assert result["minimum_positive_margin"] >= 1.5


@pytest.mark.parametrize(
    "config",
    [
        replace(SyntheticOverfitConfig(), num_pairs=1),
        replace(SyntheticOverfitConfig(), query_tokens=1),
        replace(SyntheticOverfitConfig(), query_tokens=2),
        replace(SyntheticOverfitConfig(), document_tokens=1),
        replace(SyntheticOverfitConfig(), document_tokens=3),
        replace(SyntheticOverfitConfig(), embedding_dim=0),
        replace(SyntheticOverfitConfig(), embedding_dim=1),
        replace(SyntheticOverfitConfig(), steps=0),
        replace(SyntheticOverfitConfig(), learning_rate=0.0),
    ],
)
def test_invalid_synthetic_configuration_raises(
    config: SyntheticOverfitConfig,
) -> None:
    with pytest.raises(ValueError):
        run_synthetic_overfit(config)
