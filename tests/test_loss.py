from typing import Callable

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from colpali_triton import (
    hardest_in_batch_contrastive_loss,
    maxsim,
)


def test_hand_computed_loss_uses_hardest_off_diagonal_scores() -> None:
    scores = torch.tensor(
        [
            [4.0, 1.0, 3.0],
            [2.0, 5.0, 4.0],
            [0.0, 2.0, 1.0],
        ]
    )

    losses = hardest_in_batch_contrastive_loss(
        scores, reduction="none"
    )

    expected = F.softplus(torch.tensor([-1.0, -1.0, 1.0]))
    torch.testing.assert_close(losses, expected)


def test_reductions_match_per_example_losses() -> None:
    scores = torch.tensor([[2.0, 1.0], [3.0, 4.0]])
    losses = hardest_in_batch_contrastive_loss(
        scores, reduction="none"
    )

    torch.testing.assert_close(
        hardest_in_batch_contrastive_loss(scores), losses.mean()
    )
    torch.testing.assert_close(
        hardest_in_batch_contrastive_loss(scores, reduction="sum"),
        losses.sum(),
    )


def test_loss_integrates_with_masked_maxsim() -> None:
    queries = torch.tensor(
        [
            [[1.0, 0.0], [50.0, 50.0]],
            [[0.0, 1.0], [50.0, 50.0]],
        ]
    )
    documents = torch.tensor(
        [
            [[2.0, 0.0], [100.0, 100.0]],
            [[0.0, 2.0], [100.0, 100.0]],
        ]
    )
    query_mask = torch.tensor([[True, False], [True, False]])
    document_mask = torch.tensor([[True, False], [True, False]])

    scores = maxsim(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )
    loss = hardest_in_batch_contrastive_loss(scores)

    torch.testing.assert_close(scores, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    torch.testing.assert_close(loss, F.softplus(torch.tensor(-2.0)))


def test_loss_rejects_all_empty_documents_from_maxsim() -> None:
    queries = torch.randn(2, 3, 4)
    documents = torch.randn(2, 5, 4)
    document_mask = torch.zeros(2, 5, dtype=torch.bool)
    scores = maxsim(
        queries, documents, document_mask=document_mask
    )

    assert torch.isneginf(scores).all()
    with pytest.raises(ValueError, match="finite positives"):
        hardest_in_batch_contrastive_loss(scores)


def test_gradient_matches_direct_formula() -> None:
    score_data = torch.tensor(
        [[1.0, 0.5, -1.0], [0.2, 2.0, 0.7], [0.4, 0.1, 1.5]],
        dtype=torch.float64,
    )
    actual_scores = score_data.clone().requires_grad_(True)
    actual_loss = hardest_in_batch_contrastive_loss(actual_scores)
    actual_gradient = torch.autograd.grad(actual_loss, actual_scores)[0]

    expected_scores = score_data.clone().requires_grad_(True)
    diagonal_mask = torch.eye(3, dtype=torch.bool)
    positives = expected_scores.diagonal()
    negatives = expected_scores.masked_fill(
        diagonal_mask, float("-inf")
    ).amax(dim=1)
    expected_loss = F.softplus(negatives - positives).mean()
    expected_gradient = torch.autograd.grad(expected_loss, expected_scores)[0]

    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_gradcheck() -> None:
    scores = torch.tensor(
        [[2.0, 0.3, 0.1], [0.2, 1.8, 0.4], [0.6, 0.5, 2.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    function: Callable[[Tensor], Tensor] = (
        lambda values: hardest_in_batch_contrastive_loss(values)
    )

    assert torch.autograd.gradcheck(function, (scores,), fast_mode=True)


def test_joint_pair_permutation_leaves_loss_unchanged() -> None:
    generator = torch.Generator().manual_seed(53)
    scores = torch.randn(5, 5, generator=generator)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = scores[permutation][:, permutation]

    torch.testing.assert_close(
        hardest_in_batch_contrastive_loss(scores),
        hardest_in_batch_contrastive_loss(permuted),
    )


def test_noncontiguous_scores_match_contiguous_copy() -> None:
    generator = torch.Generator().manual_seed(59)
    scores = torch.randn(4, 4, generator=generator).transpose(0, 1)

    assert not scores.is_contiguous()
    torch.testing.assert_close(
        hardest_in_batch_contrastive_loss(scores),
        hardest_in_batch_contrastive_loss(scores.contiguous()),
    )


def test_tied_hardest_negatives_split_the_gradient() -> None:
    scores = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    hardest_in_batch_contrastive_loss(
        scores, reduction="none"
    )[0].backward()

    expected_first_row = torch.tensor(
        [-0.5, 0.25, 0.25], dtype=torch.float64
    )
    torch.testing.assert_close(scores.grad[0], expected_first_row)
    torch.testing.assert_close(
        scores.grad[1:], torch.zeros(2, 3, dtype=torch.float64)
    )


def test_softplus_is_stable_for_large_score_differences() -> None:
    easy_scores = torch.tensor(
        [[10_000.0, -10_000.0], [-10_000.0, 10_000.0]]
    )
    hard_scores = -easy_scores

    easy_loss = hardest_in_batch_contrastive_loss(easy_scores)
    hard_loss = hardest_in_batch_contrastive_loss(hard_scores)

    assert torch.isfinite(easy_loss)
    assert torch.isfinite(hard_loss)
    torch.testing.assert_close(easy_loss, torch.tensor(0.0))
    torch.testing.assert_close(hard_loss, torch.tensor(20_000.0))


def test_losing_negative_infinity_is_ignored() -> None:
    scores = torch.tensor(
        [
            [3.0, 1.0, float("-inf")],
            [0.0, 2.0, float("-inf")],
            [float("-inf"), 1.0, 4.0],
        ],
        requires_grad=True,
    )

    loss = hardest_in_batch_contrastive_loss(scores)
    loss.backward()

    expected = F.softplus(torch.tensor([-2.0, -2.0, -3.0])).mean()
    torch.testing.assert_close(loss, expected)
    assert scores.grad is not None
    assert scores.grad[0, 2] == 0.0
    assert scores.grad[1, 2] == 0.0
    assert scores.grad[2, 0] == 0.0


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([[float("nan"), 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, float("inf")], [0.0, 1.0]]),
        torch.tensor([[float("-inf"), 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, float("-inf")], [float("-inf"), 1.0]]),
        torch.full((2, 2), float("-inf")),
    ],
)
def test_invalid_nonfinite_score_patterns_raise(scores: Tensor) -> None:
    with pytest.raises(ValueError, match="finite positives"):
        hardest_in_batch_contrastive_loss(scores)


@pytest.mark.parametrize(
    "scores",
    [
        torch.randn(3),
        torch.randn(2, 3, 4),
        torch.randn(2, 3),
        torch.randn(1, 1),
    ],
)
def test_invalid_shapes_raise(scores: Tensor) -> None:
    with pytest.raises(ValueError):
        hardest_in_batch_contrastive_loss(scores)


def test_non_floating_scores_raise() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        hardest_in_batch_contrastive_loss(
            torch.ones(2, 2, dtype=torch.int64)
        )


def test_non_strided_scores_raise() -> None:
    indices = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 2.0])
    scores = torch.sparse_coo_tensor(indices, values, (2, 2))

    with pytest.raises(TypeError, match="torch.strided"):
        hardest_in_batch_contrastive_loss(scores)


def test_invalid_reduction_raises() -> None:
    with pytest.raises(ValueError, match="reduction"):
        hardest_in_batch_contrastive_loss(
            torch.eye(2), reduction="median"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_loss_dtype_contract(
    input_dtype: torch.dtype, output_dtype: torch.dtype
) -> None:
    scores = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]], dtype=input_dtype
    )

    assert hardest_in_batch_contrastive_loss(scores).dtype == output_dtype


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_loss_runs_forward_and_backward_on_mps() -> None:
    scores = torch.tensor(
        [[2.0, 0.2], [0.1, 2.0]], device="mps", requires_grad=True
    )

    loss = hardest_in_batch_contrastive_loss(scores)
    loss.backward()

    assert loss.device.type == "mps"
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all().item()
