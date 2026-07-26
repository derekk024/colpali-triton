"""Training objectives for ColPali-style late-interaction retrieval."""

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


Reduction = Literal["none", "mean", "sum"]
_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def hardest_in_batch_contrastive_loss(
    scores: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Compute the paper's row-wise hardest-negative pairwise loss.

    ``scores[k, k]`` is the positive score for query ``k``. The negative is
    the largest score elsewhere in row ``k``. Per-example losses are
    ``softplus(hardest_negative - positive)``.

    This is not full in-batch InfoNCE and is not symmetrized in the
    document-to-query direction. Positive scores must be finite, and every row
    must contain at least one finite negative. Negative infinity is permitted
    only for additional, losing negative candidates.
    """

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if scores.ndim != 2:
        raise ValueError(
            f"scores must have shape [batch, batch]; got {tuple(scores.shape)}"
        )
    if scores.layout != torch.strided:
        raise TypeError("scores must use the torch.strided layout")
    if scores.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise TypeError(
            "scores must have a supported floating-point dtype "
            f"({supported}); got {scores.dtype}"
        )
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(
            "scores must be square so diagonal entries identify positives; "
            f"got {tuple(scores.shape)}"
        )
    if scores.shape[0] < 2:
        raise ValueError("scores must contain at least two pairs")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of 'none', 'mean', or 'sum'; "
            f"got {reduction!r}"
        )

    working_scores = (
        scores.float()
        if scores.dtype in (torch.float16, torch.bfloat16)
        else scores
    )
    diagonal_mask = torch.eye(
        working_scores.shape[0],
        dtype=torch.bool,
        device=working_scores.device,
    )
    finite_scores = torch.isfinite(working_scores)
    allowed_scores = finite_scores | torch.isneginf(working_scores)
    valid_scores = (
        allowed_scores.all()
        & finite_scores.diagonal().all()
        & (finite_scores & ~diagonal_mask).any(dim=1).all()
    )
    if not bool(valid_scores.item()):
        raise ValueError(
            "scores must have finite positives, no NaN or positive infinity, "
            "and at least one finite off-diagonal negative per row"
        )

    positives = working_scores.diagonal()
    hardest_negatives = working_scores.masked_fill(
        diagonal_mask, float("-inf")
    ).amax(dim=1)
    losses = F.softplus(hardest_negatives - positives)

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


__all__ = ["hardest_in_batch_contrastive_loss"]
