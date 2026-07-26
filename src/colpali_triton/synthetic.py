"""Deterministic synthetic training checks for the retrieval objective."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from colpali_triton.loss import hardest_in_batch_contrastive_loss
from colpali_triton.maxsim import maxsim


@dataclass(frozen=True)
class SyntheticOverfitConfig:
    """Configuration for the tiny CPU-only overfit experiment."""

    seed: int = 17
    num_pairs: int = 6
    query_tokens: int = 3
    document_tokens: int = 5
    embedding_dim: int = 8
    steps: int = 60
    learning_rate: float = 0.1

    def validate(self) -> None:
        if self.num_pairs < 2:
            raise ValueError("num_pairs must be at least 2")
        if self.query_tokens < 3:
            raise ValueError(
                "query_tokens must be at least 3 to exercise padding"
            )
        if self.document_tokens < 4:
            raise ValueError(
                "document_tokens must be at least 4 to exercise padding"
            )
        if self.embedding_dim < 2:
            raise ValueError(
                "embedding_dim must be at least 2 after L2 normalization"
            )
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class SyntheticOverfitResult:
    """Measured before-and-after statistics from the tiny experiment."""

    seed: int
    num_pairs: int
    steps: int
    initial_loss: float
    final_loss: float
    loss_reduction_fraction: float
    initial_top1_accuracy: float
    final_top1_accuracy: float
    minimum_positive_margin: float
    initial_query_gradient_norm: float
    initial_document_gradient_norm: float
    maximum_padding_gradient: float
    maximum_padding_parameter_change: float


def _masks(config: SyntheticOverfitConfig) -> Tuple[Tensor, Tensor]:
    pair_indices = torch.arange(
        config.num_pairs, dtype=torch.int64, device="cpu"
    )

    query_span = config.query_tokens - 1
    query_lengths = 2 + pair_indices.remainder(query_span)
    query_mask = (
        torch.arange(
            config.query_tokens, dtype=torch.int64, device="cpu"
        )[None, :]
        < query_lengths[:, None]
    )

    minimum_document_tokens = min(3, config.document_tokens)
    document_span = config.document_tokens - minimum_document_tokens + 1
    document_lengths = minimum_document_tokens + pair_indices.remainder(
        document_span
    )
    document_mask = (
        torch.arange(
            config.document_tokens, dtype=torch.int64, device="cpu"
        )[None, :]
        < document_lengths[:, None]
    )
    return query_mask, document_mask


def _scores(
    query_parameters: Tensor,
    document_parameters: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
) -> Tensor:
    queries = F.normalize(query_parameters, p=2, dim=-1)
    documents = F.normalize(document_parameters, p=2, dim=-1)
    return maxsim(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )


def _ranking_statistics(scores: Tensor) -> Tuple[float, float]:
    batch_size = scores.shape[0]
    diagonal_mask = torch.eye(
        batch_size, dtype=torch.bool, device=scores.device
    )
    positives = scores.diagonal()
    hardest_negatives = scores.masked_fill(
        diagonal_mask, float("-inf")
    ).amax(dim=1)
    top1_accuracy = (
        scores.argmax(dim=1)
        == torch.arange(batch_size, dtype=torch.int64, device=scores.device)
    ).float().mean()
    minimum_margin = (positives - hardest_negatives).amin()
    return float(top1_accuracy.item()), float(minimum_margin.item())


def _maximum_absolute(values: Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.abs().amax().item())


def run_synthetic_overfit(
    config: SyntheticOverfitConfig = SyntheticOverfitConfig(),
) -> SyntheticOverfitResult:
    """Train tiny normalized multi-vector tables on identity pair labels.

    The experiment intentionally runs on CPU and has no external data. Query
    and document tables begin independently random, so their matching identity
    must be learned through MaxSim and the hardest-in-batch objective.
    """

    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    query_parameters = nn.Parameter(
        0.25
        * torch.randn(
            config.num_pairs,
            config.query_tokens,
            config.embedding_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
    )
    document_parameters = nn.Parameter(
        0.25
        * torch.randn(
            config.num_pairs,
            config.document_tokens,
            config.embedding_dim,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
    )
    initial_query_parameters = query_parameters.detach().clone()
    initial_document_parameters = document_parameters.detach().clone()
    query_mask, document_mask = _masks(config)
    optimizer = torch.optim.Adam(
        (query_parameters, document_parameters),
        lr=config.learning_rate,
    )

    with torch.no_grad():
        initial_scores = _scores(
            query_parameters,
            document_parameters,
            query_mask,
            document_mask,
        )
        initial_loss = hardest_in_batch_contrastive_loss(initial_scores)
        initial_accuracy, _ = _ranking_statistics(initial_scores)

    initial_query_gradient_norm = 0.0
    initial_document_gradient_norm = 0.0
    maximum_padding_gradient = 0.0
    query_valid = query_mask[..., None].expand_as(query_parameters)
    document_valid = document_mask[..., None].expand_as(
        document_parameters
    )
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        scores = _scores(
            query_parameters,
            document_parameters,
            query_mask,
            document_mask,
        )
        loss = hardest_in_batch_contrastive_loss(scores)
        loss.backward()
        if query_parameters.grad is None or document_parameters.grad is None:
            raise RuntimeError("synthetic parameters did not receive gradients")
        if step == 0:
            initial_query_gradient_norm = float(
                query_parameters.grad[query_valid].norm().item()
            )
            initial_document_gradient_norm = float(
                document_parameters.grad[document_valid].norm().item()
            )
        query_padding_gradient = query_parameters.grad[~query_valid]
        document_padding_gradient = document_parameters.grad[~document_valid]
        maximum_padding_gradient = max(
            maximum_padding_gradient,
            _maximum_absolute(query_padding_gradient),
            _maximum_absolute(document_padding_gradient),
        )
        optimizer.step()

    with torch.no_grad():
        final_scores = _scores(
            query_parameters,
            document_parameters,
            query_mask,
            document_mask,
        )
        final_loss = hardest_in_batch_contrastive_loss(final_scores)
        final_accuracy, minimum_margin = _ranking_statistics(final_scores)

    initial_loss_value = float(initial_loss.item())
    final_loss_value = float(final_loss.item())
    loss_reduction = 1.0 - final_loss_value / initial_loss_value
    query_padding = ~query_mask[..., None].expand_as(query_parameters)
    document_padding = ~document_mask[..., None].expand_as(
        document_parameters
    )
    maximum_padding_parameter_change = max(
        _maximum_absolute(
            query_parameters.detach()[query_padding]
            - initial_query_parameters[query_padding]
        ),
        _maximum_absolute(
            document_parameters.detach()[document_padding]
            - initial_document_parameters[document_padding]
        ),
    )
    return SyntheticOverfitResult(
        seed=config.seed,
        num_pairs=config.num_pairs,
        steps=config.steps,
        initial_loss=initial_loss_value,
        final_loss=final_loss_value,
        loss_reduction_fraction=loss_reduction,
        initial_top1_accuracy=initial_accuracy,
        final_top1_accuracy=final_accuracy,
        minimum_positive_margin=minimum_margin,
        initial_query_gradient_norm=initial_query_gradient_norm,
        initial_document_gradient_norm=initial_document_gradient_norm,
        maximum_padding_gradient=maximum_padding_gradient,
        maximum_padding_parameter_change=maximum_padding_parameter_change,
    )


__all__ = [
    "SyntheticOverfitConfig",
    "SyntheticOverfitResult",
    "run_synthetic_overfit",
]
