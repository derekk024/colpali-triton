"""Measured evaluation for complete-corpus text retrieval indexes."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Dict, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray
import torch

from colpali_triton.metrics import rank_scores, retrieval_metrics
from colpali_triton.vidore import TextQuery


@runtime_checkable
class RetrievalIndex(Protocol):
    """Minimal index contract required by :func:`evaluate_retrieval`."""

    @property
    def document_ids(self) -> Sequence[str]:
        """IDs in the same order as values returned by :meth:`score`."""

    @property
    def index_size_bytes(self) -> int:
        """Logical index size in bytes."""

    def score(self, query_text: str) -> NDArray[np.floating]:
        """Return one finite floating-point score per document."""


@dataclass(frozen=True)
class EvaluationResult:
    """JSON-ready retrieval quality and measured scoring performance."""

    k: int
    num_queries: int
    num_documents: int
    warmup_count: int
    ndcg_at_k: float
    recall_at_k: float
    mrr_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float
    query_scoring_seconds: float
    postprocessing_and_ranking_seconds: float
    total_scoring_seconds: float
    scoring_query_document_pairs_per_second: float
    end_to_end_query_document_pairs_per_second: float
    logical_index_payload_bytes: int

    @property
    def ranking_seconds(self) -> float:
        """Backward-compatible alias for postprocessing and ranking time."""

        return self.postprocessing_and_ranking_seconds

    @property
    def documents_per_second(self) -> float:
        """Backward-compatible alias for end-to-end candidate throughput."""

        return self.end_to_end_query_document_pairs_per_second

    @property
    def index_size_bytes(self) -> int:
        """Backward-compatible alias for the logical index payload."""

        return self.logical_index_payload_bytes

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable mapping with explicit metric cutoffs."""

        return {
            "k": self.k,
            "num_queries": self.num_queries,
            "num_documents": self.num_documents,
            "warmup_count": self.warmup_count,
            f"ndcg@{self.k}": self.ndcg_at_k,
            f"recall@{self.k}": self.recall_at_k,
            f"mrr@{self.k}": self.mrr_at_k,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "query_scoring_seconds": self.query_scoring_seconds,
            "postprocessing_and_ranking_seconds": (
                self.postprocessing_and_ranking_seconds
            ),
            "total_scoring_seconds": self.total_scoring_seconds,
            "scoring_query_document_pairs_per_second": (
                self.scoring_query_document_pairs_per_second
            ),
            "end_to_end_query_document_pairs_per_second": (
                self.end_to_end_query_document_pairs_per_second
            ),
            "logical_index_payload_bytes": (
                self.logical_index_payload_bytes
            ),
        }


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_warmup_count(warmup_count: int) -> None:
    if isinstance(warmup_count, bool) or not isinstance(warmup_count, int):
        raise TypeError("warmup_count must be an integer")
    if warmup_count < 0:
        raise ValueError("warmup_count must be nonnegative")


def _prepare_document_ids(index: RetrievalIndex) -> Tuple[str, ...]:
    try:
        raw_document_ids = index.document_ids
    except AttributeError as error:
        raise TypeError("index must expose document_ids") from error

    if isinstance(raw_document_ids, (str, bytes)) or not isinstance(
        raw_document_ids, SequenceABC
    ):
        raise TypeError("index document_ids must be a sequence of strings")
    document_ids = tuple(raw_document_ids)
    if not document_ids:
        raise ValueError("index document_ids must not be empty")
    if any(not isinstance(document_id, str) for document_id in document_ids):
        raise TypeError("index document_ids must contain only strings")
    if any(not document_id for document_id in document_ids):
        raise ValueError("index document_ids must not contain empty strings")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("index document_ids must not contain duplicates")

    score = getattr(index, "score", None)
    if not callable(score):
        raise TypeError("index must expose a callable score method")
    return document_ids


def _prepare_index_size(index: RetrievalIndex) -> int:
    try:
        index_size_bytes = index.index_size_bytes
    except AttributeError as error:
        raise TypeError("index must expose index_size_bytes") from error
    if isinstance(index_size_bytes, bool) or not isinstance(
        index_size_bytes, int
    ):
        raise TypeError("index_size_bytes must be an integer")
    if index_size_bytes < 0:
        raise ValueError("index_size_bytes must be nonnegative")
    return index_size_bytes


def _prepare_queries(
    queries: Sequence[TextQuery], document_ids: Tuple[str, ...]
) -> Tuple[TextQuery, ...]:
    if isinstance(queries, (str, bytes)) or not isinstance(
        queries, SequenceABC
    ):
        raise TypeError("queries must be a sequence of TextQuery values")
    prepared = tuple(queries)
    if not prepared:
        raise ValueError("queries must not be empty")
    if any(not isinstance(query, TextQuery) for query in prepared):
        raise TypeError("queries must contain only TextQuery values")

    query_ids = tuple(query.query_id for query in prepared)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query IDs must be unique")

    corpus = frozenset(document_ids)
    for query in prepared:
        missing = query.relevant_document_ids - corpus
        if missing:
            raise ValueError(
                f"query {query.query_id!r} references documents absent "
                f"from the index: {sorted(missing)!r}"
            )
    return prepared


def _validate_scores(
    raw_scores: object, *, expected_documents: int, query_id: str
) -> NDArray[np.float64]:
    if not isinstance(raw_scores, np.ndarray):
        raise TypeError(
            f"scores for query {query_id!r} must be a NumPy array"
        )
    if raw_scores.ndim != 1:
        raise ValueError(
            f"scores for query {query_id!r} must be one-dimensional"
        )
    if raw_scores.shape != (expected_documents,):
        raise ValueError(
            f"scores for query {query_id!r} must have shape "
            f"({expected_documents},); got {raw_scores.shape}"
        )
    if not np.issubdtype(raw_scores.dtype, np.floating):
        raise TypeError(
            f"scores for query {query_id!r} must have a floating dtype"
        )
    if not np.isfinite(raw_scores).all():
        raise ValueError(
            f"scores for query {query_id!r} must contain only finite values"
        )
    return np.ascontiguousarray(raw_scores, dtype=np.float64)


def _score_and_rank(
    index: RetrievalIndex,
    query: TextQuery,
    document_ids: Tuple[str, ...],
) -> Tuple[str, ...]:
    scores = _validate_scores(
        index.score(query.text),
        expected_documents=len(document_ids),
        query_id=query.query_id,
    )
    score_tensor = torch.from_numpy(scores).unsqueeze(0)
    return rank_scores(
        (query.query_id,), document_ids, score_tensor
    )[query.query_id]


def evaluate_retrieval(
    index: RetrievalIndex,
    queries: Sequence[TextQuery],
    *,
    k: int = 5,
    warmup_count: int = 0,
) -> EvaluationResult:
    """Evaluate every query against a complete-corpus retrieval index.

    Warmups cycle over the provided queries and are excluded from all timings
    and metrics.  Each measured latency starts immediately before
    ``index.score`` and ends after score validation and the canonical complete
    ranking from :func:`colpali_triton.metrics.rank_scores`.
    """

    _validate_positive_integer(k, "k")
    _validate_warmup_count(warmup_count)
    document_ids = _prepare_document_ids(index)
    index_size_bytes = _prepare_index_size(index)
    prepared_queries = _prepare_queries(queries, document_ids)

    for warmup_index in range(warmup_count):
        query = prepared_queries[warmup_index % len(prepared_queries)]
        _score_and_rank(index, query, document_ids)

    rankings: Dict[str, Tuple[str, ...]] = {}
    query_scoring_seconds = 0.0
    ranking_seconds = 0.0
    latencies_seconds = []

    for query in prepared_queries:
        started_at = perf_counter()
        raw_scores = index.score(query.text)
        scoring_finished_at = perf_counter()
        scores = _validate_scores(
            raw_scores,
            expected_documents=len(document_ids),
            query_id=query.query_id,
        )
        score_tensor = torch.from_numpy(scores).unsqueeze(0)
        ranking = rank_scores(
            (query.query_id,), document_ids, score_tensor
        )[query.query_id]
        finished_at = perf_counter()

        scoring_elapsed = scoring_finished_at - started_at
        ranking_elapsed = finished_at - scoring_finished_at
        latency = finished_at - started_at
        if (
            scoring_elapsed < 0.0
            or ranking_elapsed < 0.0
            or latency < 0.0
            or not all(
                math.isfinite(value)
                for value in (scoring_elapsed, ranking_elapsed, latency)
            )
        ):
            raise RuntimeError("performance clock returned invalid timestamps")

        query_scoring_seconds += scoring_elapsed
        ranking_seconds += ranking_elapsed
        latencies_seconds.append(latency)
        rankings[query.query_id] = ranking

    qrels = {
        query.query_id: query.relevant_document_ids
        for query in prepared_queries
    }
    quality = retrieval_metrics(
        rankings,
        qrels,
        k=k,
        include_mrr=True,
    )
    assert quality.mrr_at_k is not None

    latencies_ms = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    total_scoring_seconds = query_scoring_seconds + ranking_seconds
    scored_pairs = len(prepared_queries) * len(document_ids)
    scoring_pairs_per_second = (
        scored_pairs / query_scoring_seconds
        if query_scoring_seconds > 0.0
        else 0.0
    )
    end_to_end_pairs_per_second = (
        scored_pairs / total_scoring_seconds
        if total_scoring_seconds > 0.0
        else 0.0
    )

    return EvaluationResult(
        k=k,
        num_queries=len(prepared_queries),
        num_documents=len(document_ids),
        warmup_count=warmup_count,
        ndcg_at_k=quality.ndcg_at_k,
        recall_at_k=quality.recall_at_k,
        mrr_at_k=quality.mrr_at_k,
        latency_p50_ms=float(np.percentile(latencies_ms, 50)),
        latency_p95_ms=float(np.percentile(latencies_ms, 95)),
        query_scoring_seconds=query_scoring_seconds,
        postprocessing_and_ranking_seconds=ranking_seconds,
        total_scoring_seconds=total_scoring_seconds,
        scoring_query_document_pairs_per_second=scoring_pairs_per_second,
        end_to_end_query_document_pairs_per_second=(
            end_to_end_pairs_per_second
        ),
        logical_index_payload_bytes=index_size_bytes,
    )


__all__ = ["EvaluationResult", "RetrievalIndex", "evaluate_retrieval"]
