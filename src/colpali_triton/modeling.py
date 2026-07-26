"""Core tensor contracts for ColPali-style multi-vector encoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from colpali_triton.maxsim import maxsim


_SUPPORTED_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


class EncodingKind(str, Enum):
    """The retrieval role represented by a batch or encoding."""

    QUERY = "query"
    DOCUMENT = "document"


def _require_tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must use the torch.strided layout")
    return value


def _require_nonempty_matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, tokens]")
    if value.shape[0] == 0:
        raise ValueError(f"{name} batch dimension must be nonzero")
    if value.shape[1] == 0:
        raise ValueError(f"{name} token dimension must be nonzero")


def _has_true_token_in_every_row(mask: Tensor) -> bool:
    return bool(mask.any(dim=1).all().item())


def _valid_values_are_finite(values: Tensor, mask: Tensor) -> bool:
    valid_positions = mask.unsqueeze(-1)
    finite_or_padding = torch.isfinite(values) | ~valid_positions
    return bool(finite_or_padding.all().item())


@dataclass(frozen=True)
class ColPaliBatch:
    """Validated model inputs whose retrieval role is explicit."""

    input_ids: Tensor
    attention_mask: Tensor
    kind: EncodingKind
    pixel_values: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EncodingKind):
            raise TypeError("kind must be an EncodingKind")

        input_ids = _require_tensor(self.input_ids, "input_ids")
        _require_nonempty_matrix(input_ids, "input_ids")
        if input_ids.dtype != torch.int64:
            raise TypeError("input_ids must have dtype torch.int64")

        attention_mask = _require_tensor(
            self.attention_mask, "attention_mask"
        )
        _require_nonempty_matrix(attention_mask, "attention_mask")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must have dtype torch.bool")
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must have the same shape as input_ids; "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        if attention_mask.device != input_ids.device:
            raise ValueError(
                "attention_mask and input_ids must be on the same device"
            )
        if not _has_true_token_in_every_row(attention_mask):
            raise ValueError(
                "attention_mask must contain at least one valid token per row"
            )

        if self.kind is EncodingKind.QUERY:
            if self.pixel_values is not None:
                raise ValueError("query batches must not contain pixel_values")
            return

        if self.pixel_values is None:
            raise ValueError("document batches must contain pixel_values")
        pixel_values = _require_tensor(self.pixel_values, "pixel_values")
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape [batch, channels, height, width]"
            )
        if pixel_values.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                "pixel_values must have a supported floating-point dtype"
            )
        if pixel_values.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "pixel_values and input_ids must have the same batch size"
            )
        if any(size == 0 for size in pixel_values.shape[1:]):
            raise ValueError(
                "pixel_values channel and spatial dimensions must be nonzero"
            )
        if pixel_values.device != input_ids.device:
            raise ValueError(
                "pixel_values and input_ids must be on the same device"
            )

    def as_model_inputs(self) -> Dict[str, Tensor]:
        """Return a fresh mapping containing only backbone tensor inputs."""

        inputs = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        if self.pixel_values is not None:
            inputs["pixel_values"] = self.pixel_values
        return inputs

    def to(
        self,
        device: Union[str, torch.device],
        *,
        pixel_dtype: Optional[torch.dtype] = None,
        non_blocking: bool = False,
    ) -> "ColPaliBatch":
        """Return an equivalent batch transferred to ``device``.

        Token and mask dtypes are preserved. ``pixel_dtype`` affects only
        document pixels, which allows mixed-precision image encoding without
        converting integer token IDs.
        """

        if pixel_dtype is not None and pixel_dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                "pixel_dtype must be a supported floating-point dtype or None"
            )
        pixel_values = self.pixel_values
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=device,
                dtype=pixel_dtype or pixel_values.dtype,
                non_blocking=non_blocking,
            )
        return ColPaliBatch(
            input_ids=self.input_ids.to(
                device=device, non_blocking=non_blocking
            ),
            attention_mask=self.attention_mask.to(
                device=device, non_blocking=non_blocking
            ),
            kind=self.kind,
            pixel_values=pixel_values,
        )


@dataclass(frozen=True)
class MultiVectorEncoding:
    """Normalized multi-vector embeddings kept together with their mask."""

    embeddings: Tensor
    mask: Tensor
    kind: EncodingKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EncodingKind):
            raise TypeError("kind must be an EncodingKind")

        embeddings = _require_tensor(self.embeddings, "embeddings")
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape [batch, tokens, dimension]"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings batch dimension must be nonzero")
        if embeddings.shape[1] == 0:
            raise ValueError("embeddings token dimension must be nonzero")
        if embeddings.shape[2] != ColPaliEncoder.EMBEDDING_DIM:
            raise ValueError(
                "embeddings must use the ColPali projection dimension "
                f"{ColPaliEncoder.EMBEDDING_DIM}; got {embeddings.shape[2]}"
            )
        if embeddings.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                "embeddings must have a supported floating-point dtype"
            )

        mask = _require_tensor(self.mask, "mask")
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, tokens]")
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if tuple(mask.shape) != tuple(embeddings.shape[:2]):
            raise ValueError(
                "mask must match the batch and token dimensions of embeddings; "
                f"got {tuple(mask.shape)} and {tuple(embeddings.shape[:2])}"
            )
        if mask.device != embeddings.device:
            raise ValueError("mask and embeddings must be on the same device")
        if not _has_true_token_in_every_row(mask):
            raise ValueError("mask must contain at least one valid token per row")
        if not _valid_values_are_finite(embeddings, mask):
            raise ValueError("valid embeddings must contain only finite values")

        padded_values = embeddings.masked_select(~mask.unsqueeze(-1))
        if padded_values.numel() and bool(
            torch.count_nonzero(padded_values).item()
        ):
            raise ValueError("masked embeddings must be exactly zero")


class RetrievalBackbone(nn.Module, ABC):
    """Injected hidden-state backbone used by the retrieval projection."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the final hidden-state dimension."""

    @abstractmethod
    def forward_hidden(self, batch: ColPaliBatch) -> Tensor:
        """Return final contextual states with shape ``[batch, tokens, H]``."""


class ColPaliEncoder(nn.Module):
    """Project injected backbone states to normalized 128-D token vectors."""

    EMBEDDING_DIM = 128

    def __init__(
        self,
        backbone: RetrievalBackbone,
        *,
        freeze_backbone: bool = True,
        normalization_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, RetrievalBackbone):
            raise TypeError("backbone must be a RetrievalBackbone")
        if not isinstance(freeze_backbone, bool):
            raise TypeError("freeze_backbone must be a bool")
        if (
            isinstance(normalization_eps, bool)
            or not isinstance(normalization_eps, (int, float))
            or not torch.isfinite(torch.tensor(float(normalization_eps)))
            or normalization_eps <= 0
        ):
            raise ValueError("normalization_eps must be finite and positive")

        hidden_size = backbone.hidden_size
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("backbone.hidden_size must be an integer")
        if hidden_size <= 0:
            raise ValueError("backbone.hidden_size must be positive")

        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.normalization_eps = float(normalization_eps)
        self.projection = nn.Linear(
            hidden_size, self.EMBEDDING_DIM, bias=True
        )

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ColPaliEncoder":
        """Set training mode while keeping a frozen backbone deterministic."""

        super().train(mode)
        if mode and self.freeze_backbone:
            self.backbone.eval()
        return self

    def _validate_hidden_states(
        self, hidden_states: object, batch: ColPaliBatch
    ) -> Tensor:
        hidden = _require_tensor(hidden_states, "backbone hidden states")
        if hidden.dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError(
                "backbone hidden states must have a supported "
                "floating-point dtype"
            )
        expected_shape = (
            batch.input_ids.shape[0],
            batch.input_ids.shape[1],
            self.projection.in_features,
        )
        if tuple(hidden.shape) != expected_shape:
            raise ValueError(
                "backbone hidden states must have shape "
                f"{expected_shape}; got {tuple(hidden.shape)}"
            )
        if hidden.device != batch.input_ids.device:
            raise ValueError(
                "backbone hidden states and inputs must be on the same device"
            )
        if not _valid_values_are_finite(hidden, batch.attention_mask):
            raise ValueError(
                "valid backbone hidden states must contain only finite values"
            )
        return hidden

    def forward(self, batch: ColPaliBatch) -> MultiVectorEncoding:
        """Encode one query or document batch as masked 128-D vectors."""

        if not isinstance(batch, ColPaliBatch):
            raise TypeError("batch must be a ColPaliBatch")

        if self.freeze_backbone:
            with torch.no_grad():
                hidden_output = self.backbone.forward_hidden(batch)
        else:
            hidden_output = self.backbone.forward_hidden(batch)

        hidden_states = self._validate_hidden_states(hidden_output, batch)
        valid_positions = batch.attention_mask.unsqueeze(-1)
        safe_hidden_states = hidden_states.masked_fill(
            ~valid_positions, 0.0
        )
        projected = self.projection(safe_hidden_states)
        normalized = F.normalize(
            projected,
            p=2,
            dim=-1,
            eps=self.normalization_eps,
        )
        embeddings = normalized.masked_fill(~valid_positions, 0.0)
        return MultiVectorEncoding(
            embeddings=embeddings,
            mask=batch.attention_mask,
            kind=batch.kind,
        )


def late_interaction_scores(
    queries: MultiVectorEncoding,
    documents: MultiVectorEncoding,
) -> Tensor:
    """Score query and document encodings while always applying both masks."""

    if not isinstance(queries, MultiVectorEncoding):
        raise TypeError("queries must be a MultiVectorEncoding")
    if not isinstance(documents, MultiVectorEncoding):
        raise TypeError("documents must be a MultiVectorEncoding")
    if queries.kind is not EncodingKind.QUERY:
        raise ValueError("queries must have EncodingKind.QUERY")
    if documents.kind is not EncodingKind.DOCUMENT:
        raise ValueError("documents must have EncodingKind.DOCUMENT")

    return maxsim(
        queries.embeddings,
        documents.embeddings,
        query_mask=queries.mask,
        document_mask=documents.mask,
    )


__all__ = [
    "ColPaliBatch",
    "ColPaliEncoder",
    "EncodingKind",
    "MultiVectorEncoding",
    "RetrievalBackbone",
    "late_interaction_scores",
]
