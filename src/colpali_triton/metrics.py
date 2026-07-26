"""Deterministic ranking and retrieval metrics for page retrieval."""

from collections.abc import Sequence as SequenceABC
from collections.abc import Set as SetABC
from dataclasses import dataclass
from math import log2
from typing import AbstractSet, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalMetrics:
    """Macro-averaged retrieval metrics at one cutoff."""

    k: int
    num_queries: int
    ndcg_at_k: float
    recall_at_k: float
    mrr_at_k: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-ready metric names with the cutoff embedded."""

        values: Dict[str, object] = {
            "k": self.k,
            "num_queries": self.num_queries,
            f"ndcg@{self.k}": self.ndcg_at_k,
            f"recall@{self.k}": self.recall_at_k,
        }
        if self.mrr_at_k is not None:
            values[f"mrr@{self.k}"] = self.mrr_at_k
        return values


def _validate_identifier_sequence(
    identifiers: Sequence[str], name: str
) -> Tuple[str, ...]:
    if isinstance(identifiers, (str, bytes)) or not isinstance(
        identifiers, SequenceABC
    ):
        raise TypeError(f"{name} must be a sequence of strings")

    prepared = tuple(identifiers)
    if not prepared:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(identifier, str) for identifier in prepared):
        raise TypeError(f"{name} must contain only strings")
    if any(not identifier for identifier in prepared):
        raise ValueError(f"{name} must not contain empty strings")
    if len(set(prepared)) != len(prepared):
        raise ValueError(f"{name} must not contain duplicates")
    return prepared


def rank_scores(
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    scores: Tensor,
) -> Dict[str, Tuple[str, ...]]:
    """Convert an all-pairs score matrix into complete, deterministic rankings.

    Higher scores rank first. Equal scores use descending lexicographic
    document ID order, matching ``trec_eval``/``pytrec_eval`` tie behavior.
    """

    prepared_query_ids = _validate_identifier_sequence(query_ids, "query_ids")
    prepared_document_ids = _validate_identifier_sequence(
        document_ids, "document_ids"
    )

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if scores.layout != torch.strided:
        raise TypeError("scores must use the torch.strided layout")
    if scores.ndim != 2:
        raise ValueError(
            "scores must have shape [queries, documents]; "
            f"got {tuple(scores.shape)}"
        )
    if not scores.is_floating_point():
        raise TypeError("scores must have a floating-point dtype")
    expected_shape = (len(prepared_query_ids), len(prepared_document_ids))
    if tuple(scores.shape) != expected_shape:
        raise ValueError(
            f"scores must have shape {expected_shape}; got {tuple(scores.shape)}"
        )
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("scores must contain only finite values")

    cpu_scores = scores.detach().to(device="cpu", dtype=torch.float64)
    rankings: Dict[str, Tuple[str, ...]] = {}
    for query_index, query_id in enumerate(prepared_query_ids):
        score_row = cpu_scores[query_index].tolist()
        scored_documents = zip(score_row, prepared_document_ids)
        ranked = sorted(
            scored_documents,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        rankings[query_id] = tuple(document_id for _, document_id in ranked)
    return rankings


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant_document_ids: Mapping[str, AbstractSet[str]],
    *,
    k: int = 5,
    include_mrr: bool = False,
) -> RetrievalMetrics:
    """Compute macro binary nDCG, recall, and optionally reciprocal rank."""

    _validate_k(k)
    if not isinstance(include_mrr, bool):
        raise TypeError("include_mrr must be a bool")
    if not rankings:
        raise ValueError("rankings must not be empty")
    if set(rankings) != set(relevant_document_ids):
        raise ValueError(
            "rankings and relevant_document_ids must have identical query IDs"
        )
    if any(not isinstance(query_id, str) or not query_id for query_id in rankings):
        raise ValueError("query IDs must be nonempty strings")

    prepared_rankings: Dict[str, Tuple[str, ...]] = {}
    shared_corpus: Optional[AbstractSet[str]] = None
    for query_id, ranking in rankings.items():
        prepared = _validate_identifier_sequence(
            ranking, f"ranking for query {query_id!r}"
        )
        corpus = frozenset(prepared)
        if shared_corpus is None:
            shared_corpus = corpus
        elif corpus != shared_corpus:
            raise ValueError(
                "every ranking must be a complete permutation of one corpus"
            )
        prepared_rankings[query_id] = prepared

    assert shared_corpus is not None
    ndcg_values = []
    recall_values = []
    reciprocal_rank_values = []

    for query_id, ranking in prepared_rankings.items():
        raw_relevant = relevant_document_ids[query_id]
        if not isinstance(raw_relevant, SetABC):
            raise TypeError(
                f"relevance set for query {query_id!r} must be a set of strings"
            )
        relevant = frozenset(raw_relevant)
        if not relevant:
            raise ValueError(
                f"relevance set for query {query_id!r} must not be empty"
            )
        if any(
            not isinstance(document_id, str) or not document_id
            for document_id in relevant
        ):
            raise ValueError("relevant document IDs must be nonempty strings")
        missing = relevant - shared_corpus
        if missing:
            raise ValueError(
                f"relevant documents for query {query_id!r} are absent "
                f"from the corpus: {sorted(missing)!r}"
            )

        retrieved = ranking[:k]
        gains = [
            1.0 / log2(rank + 1)
            for rank, document_id in enumerate(retrieved, start=1)
            if document_id in relevant
        ]
        ideal_count = min(k, len(relevant))
        ideal_dcg = sum(
            1.0 / log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        ndcg_values.append(sum(gains) / ideal_dcg)
        recall_values.append(
            len(set(retrieved).intersection(relevant)) / len(relevant)
        )

        if include_mrr:
            first_relevant_rank = next(
                (
                    rank
                    for rank, document_id in enumerate(retrieved, start=1)
                    if document_id in relevant
                ),
                None,
            )
            reciprocal_rank_values.append(
                0.0
                if first_relevant_rank is None
                else 1.0 / first_relevant_rank
            )

    count = len(prepared_rankings)
    return RetrievalMetrics(
        k=k,
        num_queries=count,
        ndcg_at_k=sum(ndcg_values) / count,
        recall_at_k=sum(recall_values) / count,
        mrr_at_k=(
            sum(reciprocal_rank_values) / count if include_mrr else None
        ),
    )


__all__ = ["RetrievalMetrics", "rank_scores", "retrieval_metrics"]
