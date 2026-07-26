from typing import Callable, Optional

import pytest
import torch
from torch import Tensor

from colpali_triton import maxsim, maxsim_nested, maxsim_vectorized


MaxSimImplementation = Callable[..., Tensor]
IMPLEMENTATIONS = (maxsim_nested, maxsim_vectorized)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_hand_computed_unbatched_score(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    documents = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])

    score = implementation(queries, documents)

    assert score.shape == torch.Size([])
    torch.testing.assert_close(score, torch.tensor(3.0))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_all_pairs_batching(implementation: MaxSimImplementation) -> None:
    queries = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    documents = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 2.0], [0.0, -2.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )

    scores = implementation(queries, documents)

    expected = torch.tensor([[1.0, 0.0, 2.0], [0.0, 2.0, 2.0]])
    assert scores.shape == (2, 3)
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_unbatched_and_batched_output_shapes(
    implementation: MaxSimImplementation,
) -> None:
    query = torch.randn(3, 4)
    query_batch = torch.randn(2, 3, 4)
    document = torch.randn(5, 4)
    document_batch = torch.randn(6, 5, 4)

    assert implementation(query, document).shape == torch.Size([])
    assert implementation(query, document_batch).shape == (6,)
    assert implementation(query_batch, document).shape == (2,)
    assert implementation(query_batch, document_batch).shape == (2, 6)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_query_mask_omits_padded_tokens(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]])
    documents = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    query_mask = torch.tensor([True, True, False])

    score = implementation(queries, documents, query_mask=query_mask)

    torch.testing.assert_close(score, torch.tensor(3.0))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_document_mask_uses_negative_infinity_not_zero(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor([[1.0, 0.0]])
    documents = torch.tensor([[-2.0, 0.0], [100.0, 0.0]])
    document_mask = torch.tensor([True, False])

    score = implementation(
        queries, documents, document_mask=document_mask
    )

    torch.testing.assert_close(score, torch.tensor(-2.0))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_query_and_document_masks_broadcast_per_batch(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [10.0, 10.0]],
        ]
    )
    documents = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 3.0], [100.0, 100.0]],
            [[-1.0, 0.0], [0.0, -2.0], [50.0, 50.0]],
        ]
    )
    query_mask = torch.tensor([[True, True], [True, False]])
    document_mask = torch.tensor(
        [[True, True, False], [True, True, False]]
    )

    scores = implementation(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    expected = torch.tensor([[5.0, 0.0], [3.0, -1.0]])
    torch.testing.assert_close(scores, expected)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_left_and_interior_padding_are_ignored(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [100.0, 100.0], [0.0, 1.0]],
        ]
    )
    documents = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [100.0, 100.0], [0.0, 2.0]],
        ]
    )
    query_mask = torch.tensor(
        [[False, True, True], [True, False, True]]
    )
    document_mask = torch.tensor(
        [[False, True, True], [True, False, True]]
    )

    scores = implementation(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    torch.testing.assert_close(scores, torch.full((2, 2), 3.0))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fully_masked_query_scores_zero(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.randn(2, 3)
    documents = torch.randn(4, 3)
    query_mask = torch.zeros(2, dtype=torch.bool)
    document_mask = torch.zeros(4, dtype=torch.bool)

    score = implementation(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    torch.testing.assert_close(score, torch.tensor(0.0))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fully_masked_document_scores_negative_infinity(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.randn(2, 3)
    documents = torch.randn(4, 3)
    document_mask = torch.zeros(4, dtype=torch.bool)

    score = implementation(
        queries, documents, document_mask=document_mask
    )

    assert torch.isneginf(score)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_mixed_empty_rows_have_defined_values_and_gradients(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor(
        [
            [[9.0, 9.0], [8.0, 8.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        requires_grad=True,
    )
    documents = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[7.0, 7.0], [6.0, 6.0]],
        ],
        requires_grad=True,
    )
    query_mask = torch.tensor([[False, False], [True, True]])
    document_mask = torch.tensor([[True, True], [False, False]])

    scores = implementation(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    torch.testing.assert_close(scores[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(scores[1, 0], torch.tensor(5.0))
    assert torch.isneginf(scores[1, 1])

    scores.sum().backward()
    assert queries.grad is not None
    assert documents.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(documents.grad).all()
    torch.testing.assert_close(queries.grad[0], torch.zeros(2, 2))
    torch.testing.assert_close(documents.grad[1], torch.zeros(2, 2))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_masked_nan_padding_is_ignored_in_forward_and_backward(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.tensor(
        [[1.0, 0.0], [float("nan"), float("nan")]], requires_grad=True
    )
    documents = torch.tensor(
        [[-2.0, 0.0], [float("nan"), float("nan")]], requires_grad=True
    )
    query_mask = torch.tensor([True, False])
    document_mask = torch.tensor([True, False])

    score = implementation(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    torch.testing.assert_close(score, torch.tensor(-2.0))
    score.backward()
    assert queries.grad is not None
    assert documents.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(documents.grad).all()
    torch.testing.assert_close(queries.grad[1], torch.zeros(2))
    torch.testing.assert_close(documents.grad[1], torch.zeros(2))


def _random_mask(
    shape: tuple, *, generator: torch.Generator, ensure_last_dim_valid: bool
) -> Tensor:
    mask = torch.rand(shape, generator=generator) > 0.3
    if ensure_last_dim_valid:
        if mask.ndim == 1:
            mask[0] = True
        else:
            mask[:, 0] = True
    return mask


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
@pytest.mark.parametrize(
    ("query_shape", "document_shape"),
    [
        ((4, 7), (6, 7)),
        ((4, 7), (3, 6, 7)),
        ((2, 4, 7), (6, 7)),
        ((2, 4, 7), (3, 6, 7)),
        ((2, 5, 128), (3, 8, 128)),
    ],
)
def test_randomized_nested_and_vectorized_parity(
    dtype: torch.dtype, query_shape: tuple, document_shape: tuple
) -> None:
    generator = torch.Generator().manual_seed(17)
    queries = torch.randn(query_shape, dtype=dtype, generator=generator)
    documents = torch.randn(document_shape, dtype=dtype, generator=generator)
    query_mask = _random_mask(
        query_shape[:-1],
        generator=generator,
        ensure_last_dim_valid=False,
    )
    document_mask = _random_mask(
        document_shape[:-1],
        generator=generator,
        ensure_last_dim_valid=True,
    )

    actual = maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )
    expected = maxsim_nested(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    elif dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_noncontiguous_embeddings_match_nested_reference() -> None:
    generator = torch.Generator().manual_seed(23)
    queries = torch.randn(2, 5, 3, generator=generator).transpose(1, 2)
    documents = torch.randn(4, 5, 7, generator=generator).transpose(1, 2)

    assert not queries.is_contiguous()
    assert not documents.is_contiguous()
    torch.testing.assert_close(
        maxsim_vectorized(queries, documents),
        maxsim_nested(queries, documents),
    )


def test_default_maxsim_uses_vectorized_contract() -> None:
    generator = torch.Generator().manual_seed(29)
    queries = torch.randn(2, 3, 5, generator=generator)
    documents = torch.randn(4, 6, 5, generator=generator)

    torch.testing.assert_close(
        maxsim(queries, documents),
        maxsim_vectorized(queries, documents),
    )


def test_default_maxsim_forwards_masks() -> None:
    queries = torch.tensor([[[1.0, 0.0], [50.0, 50.0]]])
    documents = torch.tensor([[[2.0, 0.0], [100.0, 100.0]]])
    query_mask = torch.tensor([[True, False]])
    document_mask = torch.tensor([[True, False]])

    actual = maxsim(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )
    expected = maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.tensor([[2.0]]))


def test_gradient_parity_with_masks() -> None:
    generator = torch.Generator().manual_seed(31)
    query_data = torch.randn(
        2, 3, 4, dtype=torch.float64, generator=generator
    )
    document_data = torch.randn(
        3, 5, 4, dtype=torch.float64, generator=generator
    )
    query_mask = torch.tensor([[True, True, False], [True, False, True]])
    document_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
            [True, True, False, False, True],
        ]
    )
    weights = torch.randn(2, 3, dtype=torch.float64, generator=generator)

    nested_queries = query_data.clone().requires_grad_(True)
    nested_documents = document_data.clone().requires_grad_(True)
    nested_loss = (
        maxsim_nested(
            nested_queries,
            nested_documents,
            query_mask=query_mask,
            document_mask=document_mask,
        )
        * weights
    ).sum()
    nested_gradients = torch.autograd.grad(
        nested_loss, (nested_queries, nested_documents)
    )

    vectorized_queries = query_data.clone().requires_grad_(True)
    vectorized_documents = document_data.clone().requires_grad_(True)
    vectorized_loss = (
        maxsim_vectorized(
            vectorized_queries,
            vectorized_documents,
            query_mask=query_mask,
            document_mask=document_mask,
        )
        * weights
    ).sum()
    vectorized_gradients = torch.autograd.grad(
        vectorized_loss, (vectorized_queries, vectorized_documents)
    )

    torch.testing.assert_close(
        vectorized_loss, nested_loss, rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        vectorized_gradients[0], nested_gradients[0], rtol=1e-10, atol=1e-12
    )
    torch.testing.assert_close(
        vectorized_gradients[1], nested_gradients[1], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_gradcheck(implementation: MaxSimImplementation) -> None:
    generator = torch.Generator().manual_seed(37)
    queries = torch.randn(
        1, 2, 3, dtype=torch.float64, generator=generator, requires_grad=True
    )
    documents = torch.randn(
        2, 3, 3, dtype=torch.float64, generator=generator, requires_grad=True
    )
    query_mask = torch.tensor([[True, False]])
    document_mask = torch.tensor(
        [[True, True, False], [True, False, True]]
    )

    def score(query_values: Tensor, document_values: Tensor) -> Tensor:
        return implementation(
            query_values,
            document_values,
            query_mask=query_mask,
            document_mask=document_mask,
        )

    assert torch.autograd.gradcheck(score, (queries, documents), fast_mode=True)


def test_masked_positions_receive_zero_gradient() -> None:
    queries = torch.tensor(
        [[[1.0, 0.0], [9.0, 9.0]]], requires_grad=True
    )
    documents = torch.tensor(
        [[[2.0, 0.0], [0.0, 1.0], [8.0, 8.0]]], requires_grad=True
    )
    query_mask = torch.tensor([[True, False]])
    document_mask = torch.tensor([[True, True, False]])

    maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    ).sum().backward()

    torch.testing.assert_close(queries.grad[:, 1], torch.zeros(1, 2))
    torch.testing.assert_close(documents.grad[:, 2], torch.zeros(1, 2))


def test_tied_document_maxima_have_matching_gradients() -> None:
    query_data = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    document_data = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64
    )

    nested_queries = query_data.clone().requires_grad_(True)
    nested_documents = document_data.clone().requires_grad_(True)
    maxsim_nested(nested_queries, nested_documents).backward()

    vectorized_queries = query_data.clone().requires_grad_(True)
    vectorized_documents = document_data.clone().requires_grad_(True)
    maxsim_vectorized(vectorized_queries, vectorized_documents).backward()

    torch.testing.assert_close(vectorized_queries.grad, nested_queries.grad)
    torch.testing.assert_close(vectorized_documents.grad, nested_documents.grad)
    torch.testing.assert_close(
        vectorized_documents.grad,
        torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_result_preserves_dtype_and_device(
    implementation: MaxSimImplementation,
) -> None:
    queries = torch.randn(2, 3, dtype=torch.float64)
    documents = torch.randn(4, 3, dtype=torch.float64)

    result = implementation(queries, documents)

    assert result.dtype == torch.float64
    assert result.device == queries.device


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_accumulation_dtype_contract(
    implementation: MaxSimImplementation,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> None:
    queries = torch.randn(2, 3, dtype=input_dtype)
    documents = torch.randn(4, 3, dtype=input_dtype)

    assert implementation(queries, documents).dtype == output_dtype


@pytest.mark.parametrize(
    "bad_queries",
    [
        torch.randn(3),
        torch.randn(1, 2, 3, 4),
        torch.randn(0, 2, 3),
        torch.randn(2, 0, 3),
        torch.randn(2, 3, 0),
    ],
)
def test_invalid_query_shapes_raise(bad_queries: Tensor) -> None:
    with pytest.raises(ValueError):
        maxsim_vectorized(bad_queries, torch.randn(4, 3))


def test_embedding_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="embedding dimension"):
        maxsim_vectorized(torch.randn(2, 3), torch.randn(4, 5))


def test_non_floating_embeddings_raise() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        maxsim_vectorized(torch.ones(2, 3, dtype=torch.int64), torch.randn(4, 3))


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"), reason="Float8 is not defined"
)
def test_unsupported_floating_dtype_raises() -> None:
    queries = torch.empty(2, 3, dtype=torch.float8_e4m3fn)
    documents = torch.empty(4, 3, dtype=torch.float8_e4m3fn)

    with pytest.raises(TypeError, match="supported floating-point dtype"):
        maxsim_vectorized(queries, documents)


def test_non_strided_embeddings_raise() -> None:
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1.0, 2.0])
    sparse_queries = torch.sparse_coo_tensor(indices, values, (2, 3))

    with pytest.raises(TypeError, match="torch.strided"):
        maxsim_vectorized(sparse_queries, torch.randn(4, 3))


def test_embedding_dtype_mismatch_raises() -> None:
    with pytest.raises(TypeError, match="same dtype"):
        maxsim_vectorized(
            torch.randn(2, 3, dtype=torch.float32),
            torch.randn(4, 3, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (torch.ones(2), "dtype torch.bool"),
        (torch.ones(3, dtype=torch.bool), "shape"),
    ],
)
def test_invalid_query_mask_raises(mask: Tensor, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        maxsim_vectorized(
            torch.randn(2, 4),
            torch.randn(3, 4),
            query_mask=mask,
        )


def test_invalid_batched_document_mask_shape_raises() -> None:
    with pytest.raises(ValueError, match="document_mask"):
        maxsim_vectorized(
            torch.randn(2, 3, 4),
            torch.randn(5, 6, 4),
            document_mask=torch.ones(6, dtype=torch.bool),
        )


def test_mask_device_mismatch_raises() -> None:
    query_mask = torch.ones(2, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="device"):
        maxsim_vectorized(
            torch.randn(2, 4),
            torch.randn(3, 4),
            query_mask=query_mask,
        )


def test_embedding_device_mismatch_raises() -> None:
    meta_documents = torch.empty(3, 4, device="meta")
    with pytest.raises(ValueError, match="same device"):
        maxsim_vectorized(torch.randn(2, 4), meta_documents)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_vectorized_mps_matches_cpu_and_backpropagates() -> None:
    generator = torch.Generator().manual_seed(41)
    cpu_queries = torch.randn(2, 4, 8, generator=generator)
    cpu_documents = torch.randn(3, 6, 8, generator=generator)
    query_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]]
    )
    document_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, False, True, True, False, True],
            [True, True, False, True, True, False],
        ]
    )
    expected = maxsim_vectorized(
        cpu_queries,
        cpu_documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    mps_queries = cpu_queries.to("mps").requires_grad_(True)
    mps_documents = cpu_documents.to("mps").requires_grad_(True)
    actual = maxsim_vectorized(
        mps_queries,
        mps_documents,
        query_mask=query_mask.to("mps"),
        document_mask=document_mask.to("mps"),
    )
    actual.sum().backward()

    torch.testing.assert_close(
        actual.detach().cpu(), expected, rtol=2e-4, atol=2e-5
    )
    assert mps_queries.grad is not None
    assert mps_documents.grad is not None
    assert torch.isfinite(mps_queries.grad).all().item()
    assert torch.isfinite(mps_documents.grad).all().item()
