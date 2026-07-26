from typing import Any, Optional

import pytest
import torch
from torch import Tensor, nn

from colpali_triton.loss import hardest_in_batch_contrastive_loss
from colpali_triton.modeling import (
    ColPaliBatch,
    ColPaliEncoder,
    EncodingKind,
    MultiVectorEncoding,
    RetrievalBackbone,
    late_interaction_scores,
)


class EmbeddingBackbone(RetrievalBackbone):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        with torch.no_grad():
            self.embedding.weight.zero_()
            for index in range(1, min(16, hidden_size + 1)):
                self.embedding.weight[index, index - 1] = 1.0

    @property
    def hidden_size(self) -> int:
        return self.embedding.embedding_dim

    def forward_hidden(self, batch: ColPaliBatch) -> Tensor:
        return self.embedding(batch.input_ids)


class StaticBackbone(RetrievalBackbone):
    def __init__(self, hidden_states: Any, hidden_size: int) -> None:
        super().__init__()
        self.hidden_states = hidden_states
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward_hidden(self, batch: ColPaliBatch) -> Tensor:
        return self.hidden_states


def _query_batch(
    *,
    input_ids: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> ColPaliBatch:
    if input_ids is None:
        input_ids = torch.tensor([[1, 2, 0]], dtype=torch.int64)
    if mask is None:
        mask = torch.tensor([[True, True, False]])
    return ColPaliBatch(
        input_ids=input_ids,
        attention_mask=mask,
        kind=EncodingKind.QUERY,
    )


def _document_batch(
    *,
    input_ids: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    pixel_values: Optional[Tensor] = None,
) -> ColPaliBatch:
    if input_ids is None:
        input_ids = torch.tensor([[1, 2, 0]], dtype=torch.int64)
    if mask is None:
        mask = torch.tensor([[True, True, False]])
    if pixel_values is None:
        pixel_values = torch.zeros(
            input_ids.shape[0], 3, 2, 2, dtype=torch.float32
        )
    return ColPaliBatch(
        input_ids=input_ids,
        attention_mask=mask,
        kind=EncodingKind.DOCUMENT,
        pixel_values=pixel_values,
    )


def _set_identity_projection(encoder: ColPaliEncoder) -> None:
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.zero_()
        hidden_size = encoder.projection.in_features
        encoder.projection.weight[:hidden_size, :hidden_size] = torch.eye(
            hidden_size
        )


def _encoding(
    values: Tensor, mask: Tensor, kind: EncodingKind
) -> MultiVectorEncoding:
    embeddings = torch.zeros(
        *values.shape[:-1], ColPaliEncoder.EMBEDDING_DIM
    )
    embeddings[..., : values.shape[-1]] = values
    embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0.0)
    return MultiVectorEncoding(
        embeddings=embeddings,
        mask=mask,
        kind=kind,
    )


def test_projection_is_biased_128_dimensional_and_l2_normalized() -> None:
    hidden_states = torch.tensor(
        [[[3.0, 4.0], [0.0, 5.0], [float("nan"), float("nan")]]]
    )
    encoder = ColPaliEncoder(
        StaticBackbone(hidden_states, hidden_size=2)
    )
    _set_identity_projection(encoder)

    output = encoder(_query_batch())

    assert encoder.projection.bias is not None
    assert output.embeddings.shape == (1, 3, 128)
    expected = torch.zeros(1, 3, 128)
    expected[0, 0, :2] = torch.tensor([0.6, 0.8])
    expected[0, 1, 1] = 1.0
    torch.testing.assert_close(output.embeddings, expected)
    torch.testing.assert_close(
        output.embeddings[output.mask].norm(dim=-1),
        torch.ones(2),
    )
    assert torch.isfinite(output.embeddings).all()


def test_zero_projection_is_finite_and_masked_padding_is_exactly_zero() -> None:
    backbone = EmbeddingBackbone()
    encoder = ColPaliEncoder(backbone)
    with torch.no_grad():
        encoder.projection.weight.zero_()
        encoder.projection.bias.zero_()

    output = encoder(_query_batch())

    assert torch.isfinite(output.embeddings).all()
    torch.testing.assert_close(
        output.embeddings, torch.zeros_like(output.embeddings)
    )
    torch.testing.assert_close(
        output.embeddings[~output.mask],
        torch.zeros_like(output.embeddings[~output.mask]),
    )


def test_valid_nonfinite_hidden_state_is_rejected() -> None:
    hidden_states = torch.tensor(
        [[[float("nan"), 0.0], [0.0, 1.0], [0.0, 0.0]]]
    )
    encoder = ColPaliEncoder(
        StaticBackbone(hidden_states, hidden_size=2)
    )

    with pytest.raises(ValueError, match="valid backbone hidden"):
        encoder(_query_batch())


@pytest.mark.parametrize(
    ("hidden_states", "hidden_size", "error_type", "message"),
    [
        ("not a tensor", 2, TypeError, "torch.Tensor"),
        (
            torch.ones(1, 3, dtype=torch.int64),
            1,
            TypeError,
            "floating-point",
        ),
        (
            torch.ones(1, 3),
            2,
            ValueError,
            "must have shape",
        ),
        (
            torch.ones(1, 2, 2),
            2,
            ValueError,
            "must have shape",
        ),
        (
            torch.ones(1, 3, 3),
            2,
            ValueError,
            "must have shape",
        ),
    ],
)
def test_invalid_backbone_outputs_raise(
    hidden_states: Any,
    hidden_size: int,
    error_type: type,
    message: str,
) -> None:
    encoder = ColPaliEncoder(
        StaticBackbone(hidden_states, hidden_size=hidden_size)
    )

    with pytest.raises(error_type, match=message):
        encoder(_query_batch())


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"kind": "query"}, TypeError, "EncodingKind"),
        (
            {"input_ids": torch.ones(1, 2, dtype=torch.float32)},
            TypeError,
            "torch.int64",
        ),
        (
            {"input_ids": torch.ones(2, dtype=torch.int64)},
            ValueError,
            r"\[batch, tokens\]",
        ),
        (
            {"input_ids": torch.empty(0, 2, dtype=torch.int64)},
            ValueError,
            "batch dimension",
        ),
        (
            {"attention_mask": torch.ones(1, 3, dtype=torch.int64)},
            TypeError,
            "torch.bool",
        ),
        (
            {"attention_mask": torch.ones(1, 2, dtype=torch.bool)},
            ValueError,
            "same shape",
        ),
        (
            {"attention_mask": torch.zeros(1, 3, dtype=torch.bool)},
            ValueError,
            "at least one valid token",
        ),
        (
            {
                "kind": EncodingKind.QUERY,
                "pixel_values": torch.zeros(1, 3, 2, 2),
            },
            ValueError,
            "must not contain",
        ),
    ],
)
def test_query_batch_validation(
    changes: dict, error_type: type, message: str
) -> None:
    arguments = {
        "input_ids": torch.ones(1, 3, dtype=torch.int64),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "kind": EncodingKind.QUERY,
        "pixel_values": None,
    }
    arguments.update(changes)

    with pytest.raises(error_type, match=message):
        ColPaliBatch(**arguments)


@pytest.mark.parametrize(
    ("pixel_values", "error_type", "message"),
    [
        (None, ValueError, "must contain pixel_values"),
        (torch.ones(1, 3, 2, dtype=torch.float32), ValueError, "shape"),
        (torch.ones(1, 3, 2, 2, dtype=torch.int64), TypeError, "floating"),
        (
            torch.ones(2, 3, 2, 2, dtype=torch.float32),
            ValueError,
            "same batch size",
        ),
        (
            torch.ones(1, 3, 0, 2, dtype=torch.float32),
            ValueError,
            "must be nonzero",
        ),
    ],
)
def test_document_batch_validation(
    pixel_values: Optional[Tensor], error_type: type, message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ColPaliBatch(
            input_ids=torch.ones(1, 2, dtype=torch.int64),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            kind=EncodingKind.DOCUMENT,
            pixel_values=pixel_values,
        )


def test_valid_variable_length_document_batch_is_accepted() -> None:
    batch = _document_batch(
        input_ids=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        mask=torch.tensor([[True, True, False], [True, False, False]]),
        pixel_values=torch.zeros(2, 3, 2, 2),
    )

    assert batch.kind is EncodingKind.DOCUMENT
    assert batch.pixel_values is not None
    assert batch.pixel_values.shape[0] == 2


def test_batch_model_inputs_exclude_role_and_absent_pixels() -> None:
    query = _query_batch()
    document = _document_batch()

    query_inputs = query.as_model_inputs()
    document_inputs = document.as_model_inputs()

    assert query_inputs == {
        "input_ids": query.input_ids,
        "attention_mask": query.attention_mask,
    }
    assert document_inputs == {
        "input_ids": document.input_ids,
        "attention_mask": document.attention_mask,
        "pixel_values": document.pixel_values,
    }
    assert query_inputs is not query.as_model_inputs()


def test_batch_to_preserves_token_dtypes_and_can_cast_pixels() -> None:
    batch = _document_batch()

    transferred = batch.to("cpu", pixel_dtype=torch.float64)

    assert transferred is not batch
    assert transferred.kind is EncodingKind.DOCUMENT
    assert transferred.input_ids.dtype == torch.int64
    assert transferred.attention_mask.dtype == torch.bool
    assert transferred.pixel_values is not None
    assert transferred.pixel_values.dtype == torch.float64


def test_batch_to_rejects_nonfloating_pixel_dtype() -> None:
    with pytest.raises(TypeError, match="pixel_dtype"):
        _document_batch().to("cpu", pixel_dtype=torch.int64)


@pytest.mark.parametrize(
    ("embeddings", "mask", "kind", "error_type", "message"),
    [
        (
            torch.zeros(1, 2, 127),
            torch.ones(1, 2, dtype=torch.bool),
            EncodingKind.QUERY,
            ValueError,
            "projection dimension",
        ),
        (
            torch.zeros(1, 2, 128, dtype=torch.int64),
            torch.ones(1, 2, dtype=torch.bool),
            EncodingKind.QUERY,
            TypeError,
            "floating-point",
        ),
        (
            torch.zeros(1, 2, 128),
            torch.ones(1, 2, dtype=torch.int64),
            EncodingKind.QUERY,
            TypeError,
            "torch.bool",
        ),
        (
            torch.zeros(1, 2, 128),
            torch.ones(1, 1, dtype=torch.bool),
            EncodingKind.QUERY,
            ValueError,
            "must match",
        ),
        (
            torch.zeros(1, 2, 128),
            torch.zeros(1, 2, dtype=torch.bool),
            EncodingKind.QUERY,
            ValueError,
            "at least one valid",
        ),
        (
            torch.full((1, 1, 128), float("nan")),
            torch.ones(1, 1, dtype=torch.bool),
            EncodingKind.QUERY,
            ValueError,
            "finite",
        ),
        (
            torch.zeros(1, 1, 128),
            torch.ones(1, 1, dtype=torch.bool),
            "query",
            TypeError,
            "EncodingKind",
        ),
    ],
)
def test_multivector_encoding_validation(
    embeddings: Tensor,
    mask: Tensor,
    kind: Any,
    error_type: type,
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        MultiVectorEncoding(
            embeddings=embeddings,
            mask=mask,
            kind=kind,
        )


def test_multivector_encoding_rejects_nonzero_masked_values() -> None:
    embeddings = torch.zeros(1, 2, 128)
    embeddings[0, 1, 0] = 1.0

    with pytest.raises(ValueError, match="exactly zero"):
        MultiVectorEncoding(
            embeddings=embeddings,
            mask=torch.tensor([[True, False]]),
            kind=EncodingKind.QUERY,
        )


def test_late_interaction_always_applies_document_mask() -> None:
    query = _encoding(
        torch.tensor([[[-1.0, 0.0]]]),
        torch.tensor([[True]]),
        EncodingKind.QUERY,
    )
    document = _encoding(
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        torch.tensor([[True, False]]),
        EncodingKind.DOCUMENT,
    )

    score = late_interaction_scores(query, document)

    torch.testing.assert_close(score, torch.tensor([[-1.0]]))


def test_late_interaction_rejects_reversed_roles() -> None:
    query = _encoding(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[True]]),
        EncodingKind.QUERY,
    )
    document = _encoding(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[True]]),
        EncodingKind.DOCUMENT,
    )

    with pytest.raises(ValueError, match="EncodingKind.QUERY"):
        late_interaction_scores(document, query)
    with pytest.raises(ValueError, match="EncodingKind.DOCUMENT"):
        late_interaction_scores(query, query)


def test_frozen_backbone_stays_in_eval_and_receives_no_gradients() -> None:
    backbone = EmbeddingBackbone()
    encoder = ColPaliEncoder(backbone, freeze_backbone=True)
    encoder.train()
    _set_identity_projection(encoder)

    output = encoder(_query_batch())
    output.embeddings.sum().backward()

    assert encoder.training
    assert encoder.projection.training
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()


def test_externally_configured_mode_preserves_backbone_trainability() -> None:
    backbone = EmbeddingBackbone()
    encoder = ColPaliEncoder(backbone, freeze_backbone=False)

    output = encoder(_query_batch())
    output.embeddings.sum().backward()

    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert backbone.embedding.weight.grad is not None


def test_fake_backbone_integrates_with_maxsim_and_contrastive_loss() -> None:
    backbone = EmbeddingBackbone(hidden_size=4)
    encoder = ColPaliEncoder(backbone, freeze_backbone=True)
    _set_identity_projection(encoder)
    queries = _query_batch(
        input_ids=torch.tensor([[1, 3], [2, 0]]),
        mask=torch.tensor([[True, True], [True, False]]),
    )
    documents = _document_batch(
        input_ids=torch.tensor([[1, 3, 0], [2, 0, 0]]),
        mask=torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
        pixel_values=torch.zeros(2, 3, 2, 2),
    )

    query_encoding = encoder(queries)
    document_encoding = encoder(documents)
    scores = late_interaction_scores(query_encoding, document_encoding)
    loss = hardest_in_batch_contrastive_loss(scores)
    loss.backward()

    torch.testing.assert_close(
        scores, torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    )
    assert torch.isfinite(loss)
    assert scores.diagonal().gt(scores.flip(dims=(1,)).diagonal()).all()
    assert encoder.projection.weight.grad is not None
    assert torch.count_nonzero(encoder.projection.weight.grad) > 0
    assert encoder.projection.bias.grad is not None
    assert torch.isfinite(encoder.projection.bias.grad).all()
    assert backbone.embedding.weight.grad is None


@pytest.mark.parametrize(
    ("backbone", "freeze_backbone", "normalization_eps", "error_type", "message"),
    [
        (
            nn.Linear(2, 2),
            True,
            1e-12,
            TypeError,
            "RetrievalBackbone",
        ),
        (
            EmbeddingBackbone(),
            "frozen_backbone",
            1e-12,
            TypeError,
            "freeze_backbone",
        ),
        (
            EmbeddingBackbone(),
            True,
            0.0,
            ValueError,
            "finite and positive",
        ),
        (
            EmbeddingBackbone(),
            True,
            float("inf"),
            ValueError,
            "finite and positive",
        ),
    ],
)
def test_encoder_constructor_validation(
    backbone: nn.Module,
    freeze_backbone: Any,
    normalization_eps: float,
    error_type: type,
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ColPaliEncoder(
            backbone,  # type: ignore[arg-type]
            freeze_backbone=freeze_backbone,
            normalization_eps=normalization_eps,
        )


def test_encoder_rejects_non_batch_input() -> None:
    encoder = ColPaliEncoder(EmbeddingBackbone())

    with pytest.raises(TypeError, match="ColPaliBatch"):
        encoder(torch.ones(1, 2, dtype=torch.int64))  # type: ignore[arg-type]
