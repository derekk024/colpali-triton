"""Paper-faithful LoRA topology and trainability audits for ColPali."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Dict, Optional, Tuple

from torch import nn


PAPER_LORA_TARGET_PATTERN = (
    r"(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|"
    r"v_proj|o_proj).*$|.*(custom_text_proj).*$)"
)
PAPER_LORA_TARGET_COUNT = 127
PAPER_LORA_TRAINABLE_PARAMETERS = 39_292_928

_APPROVED_LANGUAGE_SUFFIXES = frozenset(
    {
        "down_proj",
        "gate_proj",
        "up_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    }
)


class AdaptationMode(str, Enum):
    """Supported Phase 7 parameter-update regimes."""

    FROZEN_BACKBONE = "frozen_backbone"
    LORA = "lora"
    INFERENCE = "inference"


@dataclass(frozen=True)
class LoRASpec:
    """Validated PEFT settings matching the paper-era ColPali run."""

    rank: int = 32
    alpha: int = 32
    dropout: float = 0.1
    init: str = "gaussian"
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    target_pattern: str = PAPER_LORA_TARGET_PATTERN

    def __post_init__(self) -> None:
        for label, value in (("rank", self.rank), ("alpha", self.alpha)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.init != "gaussian":
            raise ValueError("paper-faithful LoRA init must be 'gaussian'")
        if self.bias != "none":
            raise ValueError("paper-faithful LoRA bias must be 'none'")
        if self.task_type != "FEATURE_EXTRACTION":
            raise ValueError(
                "paper-faithful task_type must be 'FEATURE_EXTRACTION'"
            )
        if not isinstance(self.target_pattern, str):
            raise TypeError("target_pattern must be a string")
        try:
            re.compile(self.target_pattern)
        except re.error as error:
            raise ValueError("target_pattern must be a valid regex") from error

    def to_dict(self) -> Dict[str, object]:
        """Return the PEFT-relevant settings in a JSON-ready form."""

        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": float(self.dropout),
            "init_lora_weights": self.init,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_pattern,
        }


@dataclass(frozen=True)
class TrainabilityReport:
    """Deterministic summary of a model's parameter-update surface."""

    mode: AdaptationMode
    total_parameters: int
    trainable_parameters: int
    trainable_parameter_names: Tuple[str, ...]
    resolved_lora_targets: Tuple[str, ...] = ()

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode.value,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_fraction": self.trainable_fraction,
            "trainable_parameter_names": list(
                self.trainable_parameter_names
            ),
            "resolved_lora_targets": list(self.resolved_lora_targets),
        }


def resolve_lora_targets(
    model: nn.Module,
    *,
    target_pattern: str = PAPER_LORA_TARGET_PATTERN,
) -> Tuple[str, ...]:
    """Resolve and audit every linear module selected by the LoRA regex."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(target_pattern, str):
        raise TypeError("target_pattern must be a string")
    try:
        pattern = re.compile(target_pattern)
    except re.error as error:
        raise ValueError("target_pattern must be a valid regex") from error

    targets = []
    for name, module in model.named_modules():
        if not name or pattern.fullmatch(name) is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoRA target {name!r} must be torch.nn.Linear; "
                f"got {type(module).__name__}"
            )
        if "vision_tower" in name or "multi_modal_projector" in name:
            raise ValueError(
                f"LoRA target {name!r} reaches a forbidden vision module"
            )
        if name.endswith("custom_text_proj"):
            targets.append(name)
            continue
        suffix = name.rsplit(".", 1)[-1]
        if "language_model" not in name or suffix not in (
            _APPROVED_LANGUAGE_SUFFIXES
        ):
            raise ValueError(f"LoRA target {name!r} is outside the contract")
        targets.append(name)

    if not targets:
        raise ValueError("LoRA target pattern resolved no linear modules")
    if len(set(targets)) != len(targets):
        raise RuntimeError("LoRA target resolution produced duplicates")
    return tuple(sorted(targets))


def build_lora_config(spec: LoRASpec = LoRASpec()) -> object:
    """Create the exact PEFT config, importing PEFT only when requested."""

    if not isinstance(spec, LoRASpec):
        raise TypeError("spec must be a LoRASpec")
    try:
        from peft import LoraConfig, TaskType
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "PEFT is required; install the 'multimodal' extra"
        ) from error

    return LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=float(spec.dropout),
        init_lora_weights=spec.init,
        bias=spec.bias,
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=spec.target_pattern,
    )


def _parameter_report(
    model: nn.Module,
    *,
    mode: AdaptationMode,
    resolved_lora_targets: Tuple[str, ...] = (),
) -> TrainabilityReport:
    named_parameters = tuple(model.named_parameters())
    return TrainabilityReport(
        mode=mode,
        total_parameters=sum(
            parameter.numel() for _, parameter in named_parameters
        ),
        trainable_parameters=sum(
            parameter.numel()
            for _, parameter in named_parameters
            if parameter.requires_grad
        ),
        trainable_parameter_names=tuple(
            name
            for name, parameter in named_parameters
            if parameter.requires_grad
        ),
        resolved_lora_targets=resolved_lora_targets,
    )


def freeze_for_inference(model: nn.Module) -> TrainabilityReport:
    """Freeze every parameter for released-checkpoint inference."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    model.requires_grad_(False)
    report = _parameter_report(model, mode=AdaptationMode.INFERENCE)
    if report.trainable_parameters != 0:
        raise RuntimeError("inference model still has trainable parameters")
    return report


def configure_frozen_backbone_ablation(
    model: nn.Module,
) -> TrainabilityReport:
    """Freeze the backbone and fully train only the retrieval projection.

    This is a useful ablation.  It is deliberately distinct from the paper's
    LoRA regime, which freezes the base projection and adapts it through LoRA.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    projection = getattr(model, "custom_text_proj", None)
    if not isinstance(projection, nn.Linear):
        raise ValueError("model must expose linear custom_text_proj")

    model.requires_grad_(False)
    projection.requires_grad_(True)
    report = _parameter_report(
        model, mode=AdaptationMode.FROZEN_BACKBONE
    )
    expected_names = {
        f"custom_text_proj.{name}"
        for name, _ in projection.named_parameters()
    }
    if set(report.trainable_parameter_names) != expected_names:
        raise RuntimeError(
            "frozen-backbone ablation exposed unexpected trainable parameters"
        )
    return report


def apply_lora(
    model: nn.Module,
    spec: LoRASpec = LoRASpec(),
    *,
    expected_target_count: Optional[int] = None,
    expected_trainable_parameters: Optional[int] = None,
) -> Tuple[nn.Module, TrainabilityReport]:
    """Install LoRA, then prove that only approved adapters are trainable."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(spec, LoRASpec):
        raise TypeError("spec must be a LoRASpec")
    targets = resolve_lora_targets(
        model, target_pattern=spec.target_pattern
    )
    if (
        expected_target_count is not None
        and len(targets) != expected_target_count
    ):
        raise ValueError(
            f"resolved {len(targets)} LoRA targets; "
            f"expected {expected_target_count}"
        )

    try:
        from peft import get_peft_model
    except ImportError as error:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "PEFT is required; install the 'multimodal' extra"
        ) from error
    adapted = get_peft_model(model, build_lora_config(spec))
    report = _parameter_report(
        adapted,
        mode=AdaptationMode.LORA,
        resolved_lora_targets=targets,
    )

    forbidden = [
        name
        for name in report.trainable_parameter_names
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if forbidden:
        raise RuntimeError(
            "LoRA regime exposed non-adapter trainables: "
            f"{forbidden!r}"
        )
    vision_trainables = [
        name
        for name in report.trainable_parameter_names
        if "vision_tower" in name or "multi_modal_projector" in name
    ]
    if vision_trainables:
        raise RuntimeError(
            f"LoRA regime exposed vision trainables: {vision_trainables!r}"
        )
    if (
        expected_trainable_parameters is not None
        and report.trainable_parameters != expected_trainable_parameters
    ):
        raise ValueError(
            f"LoRA has {report.trainable_parameters} trainable parameters; "
            f"expected {expected_trainable_parameters}"
        )
    return adapted, report


__all__ = [
    "AdaptationMode",
    "LoRASpec",
    "PAPER_LORA_TARGET_COUNT",
    "PAPER_LORA_TARGET_PATTERN",
    "PAPER_LORA_TRAINABLE_PARAMETERS",
    "TrainabilityReport",
    "apply_lora",
    "build_lora_config",
    "configure_frozen_backbone_ablation",
    "freeze_for_inference",
    "resolve_lora_targets",
]
