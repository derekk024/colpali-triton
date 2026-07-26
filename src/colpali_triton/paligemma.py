"""PaliGemma-backed ColPali model with the paper-era state-dict layout."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from colpali_triton.modeling import (
    ColPaliBatch,
    EncodingKind,
    MultiVectorEncoding,
)

try:
    from transformers.models.paligemma.configuration_paligemma import (
        PaliGemmaConfig,
    )
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaPreTrainedModel,
    )
except ImportError as error:  # pragma: no cover - optional dependency path
    raise ImportError(
        "PaliGemma support requires the 'multimodal' project extra"
    ) from error


class ColPali(PaliGemmaPreTrainedModel):
    """Paper-faithful PaliGemma final-state retrieval projection.

    Attribute names intentionally match the released ``vidore/colpali``
    checkpoint: the full generative backbone is stored as ``model`` and the
    shared biased retrieval projection as ``custom_text_proj``.
    """

    config_class = PaliGemmaConfig
    main_input_name = "input_ids"
    RETRIEVAL_EMBEDDING_DIM = 128

    def __init__(
        self,
        config: PaliGemmaConfig,
        *,
        normalization_eps: float = 1e-12,
        model: Optional[PaliGemmaForConditionalGeneration] = None,
    ) -> None:
        super().__init__(config)
        if (
            isinstance(normalization_eps, bool)
            or not isinstance(normalization_eps, (int, float))
            or not math.isfinite(float(normalization_eps))
            or float(normalization_eps) <= 0.0
        ):
            raise ValueError("normalization_eps must be finite and positive")
        if model is not None and not isinstance(
            model, PaliGemmaForConditionalGeneration
        ):
            raise TypeError(
                "model must be PaliGemmaForConditionalGeneration or None"
            )

        retrieval_dimension = getattr(
            config,
            "retrieval_embedding_dim",
            self.RETRIEVAL_EMBEDDING_DIM,
        )
        if (
            isinstance(retrieval_dimension, bool)
            or not isinstance(retrieval_dimension, int)
            or retrieval_dimension != self.RETRIEVAL_EMBEDDING_DIM
        ):
            raise ValueError(
                "ColPali retrieval_embedding_dim must be exactly 128"
            )
        config.retrieval_embedding_dim = retrieval_dimension

        self.model = (
            model
            if model is not None
            else PaliGemmaForConditionalGeneration(config)
        )
        inner_tied_keys = getattr(self.model, "_tied_weights_keys", None)
        if inner_tied_keys:
            self._tied_weights_keys = [
                f"model.{key}" for key in inner_tied_keys
            ]
        self.custom_text_proj = nn.Linear(
            config.text_config.hidden_size,
            retrieval_dimension,
            bias=True,
        )
        self.normalization_eps = float(normalization_eps)

    @property
    def retrieval_embedding_dim(self) -> int:
        return self.custom_text_proj.out_features

    @property
    def hidden_size(self) -> int:
        return self.custom_text_proj.in_features

    def get_input_embeddings(self) -> nn.Module:
        """Delegate tied-token input embeddings to the PaliGemma backbone."""

        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the PaliGemma token embeddings through the outer model."""

        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        """Delegate the tied language-model head to the backbone."""

        return self.model.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Set the PaliGemma language-model head through the outer model."""

        self.model.set_output_embeddings(value)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Optional[Tensor] = None,
        **model_kwargs: object,
    ) -> MultiVectorEncoding:
        """Return normalized query or page token vectors and their mask."""

        kind = (
            EncodingKind.DOCUMENT
            if pixel_values is not None
            else EncodingKind.QUERY
        )
        batch = ColPaliBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            kind=kind,
            pixel_values=pixel_values,
        )
        for protected_name in (
            "output_hidden_states",
            "return_dict",
            "use_cache",
        ):
            protected_value = model_kwargs.pop(protected_name, None)
            if protected_value is not None:
                raise ValueError(
                    f"{protected_name} is controlled by ColPali"
                )

        outputs = self.model(
            **batch.as_model_inputs(),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **model_kwargs,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if (
            not isinstance(hidden_states, (tuple, list))
            or not hidden_states
        ):
            raise ValueError(
                "PaliGemma output must contain a nonempty hidden_states "
                "sequence"
            )
        final_hidden_state = hidden_states[-1]
        if not isinstance(final_hidden_state, Tensor):
            raise TypeError("final PaliGemma hidden state must be a tensor")
        if final_hidden_state.layout != torch.strided:
            raise TypeError(
                "final PaliGemma hidden state must use strided layout"
            )
        if not final_hidden_state.is_floating_point():
            raise TypeError(
                "final PaliGemma hidden state must be floating point"
            )
        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden_size,
        )
        if tuple(final_hidden_state.shape) != expected_shape:
            raise ValueError(
                "final PaliGemma hidden state must have shape "
                f"{expected_shape}; got {tuple(final_hidden_state.shape)}"
            )
        if final_hidden_state.device != attention_mask.device:
            raise ValueError(
                "final PaliGemma hidden state and mask must share a device"
            )

        valid_positions = attention_mask.unsqueeze(-1)
        finite_or_padding = (
            torch.isfinite(final_hidden_state) | ~valid_positions
        )
        if not bool(finite_or_padding.all().item()):
            raise ValueError(
                "valid PaliGemma hidden states must contain only finite values"
            )
        safe_hidden_states = final_hidden_state.masked_fill(
            ~valid_positions, 0.0
        )
        projected = self.custom_text_proj(safe_hidden_states)
        embeddings = F.normalize(
            projected,
            p=2,
            dim=-1,
            eps=self.normalization_eps,
        ).masked_fill(~valid_positions, 0.0)
        return MultiVectorEncoding(
            embeddings=embeddings,
            mask=attention_mask,
            kind=kind,
        )


__all__ = ["ColPali"]
