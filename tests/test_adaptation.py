import pytest
import torch
from torch import nn

from colpali_triton.adaptation import (
    AdaptationMode,
    LoRASpec,
    apply_lora,
    build_lora_config,
    configure_frozen_backbone_ablation,
    freeze_for_inference,
    resolve_lora_targets,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


class _LanguageLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _FakeColPali(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.config = {"model_type": "custom"}
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList(
            [_LanguageLayer() for _ in range(num_layers)]
        )
        self.model.vision_tower = _Attention()
        self.model.multi_modal_projector = nn.Linear(4, 4)
        self.custom_text_proj = nn.Linear(4, 2, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        **_: object,
    ) -> torch.Tensor:
        del attention_mask
        assert input_ids is not None
        return input_ids


def test_lora_spec_matches_paper_peft_contract() -> None:
    spec = LoRASpec()

    assert spec.to_dict() == {
        "r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "init_lora_weights": "gaussian",
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
        "target_modules": spec.target_pattern,
    }
    peft_config = build_lora_config(spec)
    assert peft_config.r == 32
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.1
    assert peft_config.init_lora_weights == "gaussian"
    assert peft_config.bias == "none"
    assert peft_config.target_modules == spec.target_pattern


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rank": 0},
        {"rank": True},
        {"alpha": -1},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": float("nan")},
        {"init": True},
        {"bias": "all"},
        {"task_type": "CAUSAL_LM"},
        {"target_pattern": "("},
    ),
)
def test_lora_spec_rejects_contract_drift(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        LoRASpec(**kwargs)


def test_resolve_lora_targets_is_exact_and_excludes_vision() -> None:
    model = _FakeColPali(num_layers=2)

    targets = resolve_lora_targets(model)

    assert len(targets) == 15
    assert targets == tuple(sorted(targets))
    assert "custom_text_proj" in targets
    assert all("vision_tower" not in target for target in targets)
    assert all("multi_modal_projector" not in target for target in targets)


def test_resolve_lora_targets_rejects_empty_or_vision_selection() -> None:
    model = _FakeColPali()

    with pytest.raises(ValueError, match="resolved no"):
        resolve_lora_targets(model, target_pattern=r".*not_present")
    with pytest.raises(ValueError, match="forbidden vision"):
        resolve_lora_targets(
            model, target_pattern=r".*vision_tower.*q_proj"
        )


def test_frozen_backbone_ablation_trains_full_projection_only() -> None:
    model = _FakeColPali()

    report = configure_frozen_backbone_ablation(model)

    assert report.mode is AdaptationMode.FROZEN_BACKBONE
    assert set(report.trainable_parameter_names) == {
        "custom_text_proj.weight",
        "custom_text_proj.bias",
    }
    assert report.trainable_parameters == 10
    assert all(
        not parameter.requires_grad
        for parameter in model.model.parameters()
    )


def test_freeze_for_inference_disables_every_gradient() -> None:
    model = _FakeColPali()

    report = freeze_for_inference(model)

    assert report.mode is AdaptationMode.INFERENCE
    assert report.trainable_parameters == 0
    assert report.trainable_parameter_names == ()


def test_apply_lora_exposes_only_approved_adapter_parameters() -> None:
    model = _FakeColPali(num_layers=2)

    adapted, report = apply_lora(
        model,
        expected_target_count=15,
        expected_trainable_parameters=4_544,
    )

    assert report.mode is AdaptationMode.LORA
    assert len(report.resolved_lora_targets) == 15
    assert len(report.trainable_parameter_names) == 30
    assert all(
        ".lora_A." in name or ".lora_B." in name
        for name in report.trainable_parameter_names
    )
    assert all(
        "vision_tower" not in name
        for name in report.trainable_parameter_names
    )
    base_projection = dict(adapted.named_parameters())[
        "base_model.model.custom_text_proj.base_layer.weight"
    ]
    assert not base_projection.requires_grad


def test_apply_lora_checks_expected_topology_before_installation() -> None:
    model = _FakeColPali(num_layers=1)

    with pytest.raises(ValueError, match="expected 127"):
        apply_lora(model, expected_target_count=127)

    assert not any(
        "lora_" in name for name, _ in model.named_parameters()
    )


def test_trainability_report_is_json_ready() -> None:
    report = configure_frozen_backbone_ablation(_FakeColPali())

    assert report.to_dict()["mode"] == "frozen_backbone"
    assert report.to_dict()["trainable_fraction"] == (
        report.trainable_parameters / report.total_parameters
    )
