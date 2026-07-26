"""Measured ColPali indexing, MaxSim retrieval, and mean-pooled ablation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from time import perf_counter
from typing import Dict, FrozenSet, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from colpali_triton.maxsim import maxsim
from colpali_triton.metrics import rank_scores, retrieval_metrics
from colpali_triton.modeling import (
    EncodingKind,
    MultiVectorEncoding,
)
from colpali_triton.vidore_images import MultimodalRetrievalDataset


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def masked_mean_pool(encoding: MultiVectorEncoding) -> Tensor:
    """Return one float32, L2-normalized vector per multi-vector item."""

    if not isinstance(encoding, MultiVectorEncoding):
        raise TypeError("encoding must be a MultiVectorEncoding")
    mask = encoding.mask.unsqueeze(-1)
    values = encoding.embeddings.to(dtype=torch.float32)
    counts = mask.sum(dim=1, dtype=torch.float32)
    means = values.masked_fill(~mask, 0.0).sum(dim=1) / counts
    norms = torch.linalg.vector_norm(means, ord=2, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all().item()):
        raise RuntimeError("mean-pooled vector norms are not finite")
    if bool((norms <= 0.0).any().item()):
        raise RuntimeError("mean-pooled vectors must have nonzero norms")
    pooled = F.normalize(means, p=2, dim=-1)
    if not bool(torch.isfinite(pooled).all().item()):
        raise RuntimeError("mean-pooled vectors are not finite")
    return pooled


@dataclass(frozen=True)
class DocumentIndex:
    """Device-resident document representations for both Phase 9 scorers."""

    document_ids: Tuple[str, ...]
    embeddings: Tensor
    mask: Tensor
    mean_embeddings: Tensor
    indexing_seconds: float
    per_page_seconds: Tuple[float, ...]

    def __post_init__(self) -> None:
        document_ids = tuple(self.document_ids)
        per_page_seconds = tuple(self.per_page_seconds)
        if not document_ids or len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must be nonempty and unique")
        if (
            not isinstance(self.embeddings, Tensor)
            or self.embeddings.ndim != 3
            or self.embeddings.shape[0] != len(document_ids)
            or self.embeddings.shape[2] != 128
        ):
            raise ValueError("embeddings must have shape [documents, tokens, 128]")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype != torch.bool
            or tuple(self.mask.shape) != tuple(self.embeddings.shape[:2])
        ):
            raise ValueError("mask must match document embedding positions")
        if (
            not isinstance(self.mean_embeddings, Tensor)
            or tuple(self.mean_embeddings.shape) != (len(document_ids), 128)
            or self.mean_embeddings.dtype != torch.float32
        ):
            raise ValueError(
                "mean_embeddings must have shape [documents, 128] in float32"
            )
        if (
            self.embeddings.device != self.mask.device
            or self.embeddings.device != self.mean_embeddings.device
        ):
            raise ValueError("all index tensors must share one device")
        if (
            not math.isfinite(self.indexing_seconds)
            or self.indexing_seconds <= 0.0
        ):
            raise ValueError("indexing_seconds must be finite and positive")
        if (
            len(per_page_seconds) != len(document_ids)
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in per_page_seconds
            )
        ):
            raise ValueError(
                "per_page_seconds must contain one positive value per page"
            )
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "per_page_seconds", per_page_seconds)

    @property
    def device(self) -> torch.device:
        return self.embeddings.device

    @property
    def multi_vector_payload_bytes(self) -> int:
        return _tensor_bytes(self.embeddings) + _tensor_bytes(self.mask)

    @property
    def mean_vector_payload_bytes(self) -> int:
        return _tensor_bytes(self.mean_embeddings)

    def indexing_record(self) -> Dict[str, object]:
        latencies = np.asarray(self.per_page_seconds) * 1000.0
        count = len(self.document_ids)
        return {
            "documents": count,
            "seconds": self.indexing_seconds,
            "pages_per_second": count / self.indexing_seconds,
            "per_page_latency_p50_ms": float(
                np.percentile(latencies, 50)
            ),
            "per_page_latency_p95_ms": float(
                np.percentile(latencies, 95)
            ),
            "multi_vector_payload_bytes": (
                self.multi_vector_payload_bytes
            ),
            "multi_vector_payload_bytes_per_page": (
                self.multi_vector_payload_bytes / count
            ),
            "mean_vector_payload_bytes": self.mean_vector_payload_bytes,
            "mean_vector_payload_bytes_per_page": (
                self.mean_vector_payload_bytes / count
            ),
            "payload_excludes": (
                "model weights, source images, Python object overhead, "
                "and accelerator allocator overhead"
            ),
        }


@dataclass(frozen=True)
class ScorerResult:
    """Quality and measured query performance for one scoring rule."""

    name: str
    k: int
    num_queries: int
    num_documents: int
    warmup_count: int
    ndcg_at_k: float
    recall_at_k: float
    mrr_at_k: float
    scoring_seconds: float
    ranking_seconds: float
    shared_query_encoding_seconds: float
    latency_p50_ms: float
    latency_p95_ms: float
    scoring_pairs_per_second: float
    end_to_end_pairs_per_second: float
    logical_index_payload_bytes: int
    score_matrix_sha256: str
    top_k_rankings: Mapping[str, Tuple[str, ...]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "k": self.k,
            "num_queries": self.num_queries,
            "num_documents": self.num_documents,
            "warmup_count": self.warmup_count,
            f"ndcg@{self.k}": self.ndcg_at_k,
            f"recall@{self.k}": self.recall_at_k,
            f"mrr@{self.k}": self.mrr_at_k,
            "scoring_seconds": self.scoring_seconds,
            "postprocessing_and_ranking_seconds": self.ranking_seconds,
            "shared_query_encoding_seconds": (
                self.shared_query_encoding_seconds
            ),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "scoring_query_document_pairs_per_second": (
                self.scoring_pairs_per_second
            ),
            "end_to_end_query_document_pairs_per_second": (
                self.end_to_end_pairs_per_second
            ),
            "logical_index_payload_bytes": (
                self.logical_index_payload_bytes
            ),
            "logical_index_payload_bytes_per_page": (
                self.logical_index_payload_bytes / self.num_documents
            ),
            "score_matrix_sha256": self.score_matrix_sha256,
            "top_k_rankings": {
                query_id: list(ranking)
                for query_id, ranking in self.top_k_rankings.items()
            },
        }


@dataclass(frozen=True)
class MultimodalEvaluationResult:
    """The paired late-interaction and mean-vector evaluation."""

    indexing: Mapping[str, object]
    shared_query_encoding_seconds: float
    late_interaction: ScorerResult
    mean_pooling: ScorerResult

    def to_dict(self) -> Dict[str, object]:
        return {
            "indexing": dict(self.indexing),
            "shared_query_encoding_seconds": (
                self.shared_query_encoding_seconds
            ),
            "scorers": {
                self.late_interaction.name: self.late_interaction.to_dict(),
                self.mean_pooling.name: self.mean_pooling.to_dict(),
            },
        }


def encode_document_index(
    model: nn.Module,
    processor: object,
    dataset: MultimodalRetrievalDataset,
    *,
    device: torch.device,
    pixel_dtype: torch.dtype,
    document_batch_size: int = 1,
) -> DocumentIndex:
    """Encode every selected page once and build both logical indexes."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(dataset, MultimodalRetrievalDataset):
        raise TypeError("dataset must be a MultimodalRetrievalDataset")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if document_batch_size != 1:
        raise ValueError("document_batch_size must be exactly 1")
    process_documents = getattr(processor, "process_documents", None)
    if not callable(process_documents):
        raise TypeError("processor must expose process_documents")

    embeddings = []
    masks = []
    per_page_seconds = []
    token_count = None
    _synchronize(device)
    indexing_started = perf_counter()
    with torch.inference_mode():
        for document in dataset.documents:
            page_started = perf_counter()
            source_image = document.open_image()
            try:
                batch = process_documents([source_image]).to(
                    device,
                    pixel_dtype=pixel_dtype,
                )
            finally:
                source_image.close()
            encoding = model(**batch.as_model_inputs())
            _synchronize(device)
            elapsed = perf_counter() - page_started
            if not isinstance(encoding, MultiVectorEncoding):
                raise TypeError("model must return MultiVectorEncoding")
            if encoding.kind is not EncodingKind.DOCUMENT:
                raise RuntimeError("document model output has the wrong kind")
            if token_count is None:
                token_count = encoding.embeddings.shape[1]
            elif encoding.embeddings.shape[1] != token_count:
                raise RuntimeError(
                    "document token counts must be equal for a dense index"
                )
            embeddings.append(encoding.embeddings.detach())
            masks.append(encoding.mask.detach())
            per_page_seconds.append(elapsed)

    stacked_embeddings = torch.cat(embeddings, dim=0)
    stacked_mask = torch.cat(masks, dim=0)
    complete_encoding = MultiVectorEncoding(
        embeddings=stacked_embeddings,
        mask=stacked_mask,
        kind=EncodingKind.DOCUMENT,
    )
    means = masked_mean_pool(complete_encoding)
    _synchronize(device)
    indexing_seconds = perf_counter() - indexing_started
    return DocumentIndex(
        document_ids=tuple(
            document.document_id for document in dataset.documents
        ),
        embeddings=stacked_embeddings,
        mask=stacked_mask,
        mean_embeddings=means,
        indexing_seconds=indexing_seconds,
        per_page_seconds=tuple(per_page_seconds),
    )


def _encode_query(
    model: nn.Module,
    processor: object,
    query_text: str,
    device: torch.device,
) -> MultiVectorEncoding:
    batch = processor.process_queries([query_text]).to(device)
    output = model(**batch.as_model_inputs())
    if not isinstance(output, MultiVectorEncoding):
        raise TypeError("model must return MultiVectorEncoding")
    if output.kind is not EncodingKind.QUERY:
        raise RuntimeError("query model output has the wrong kind")
    return output


def _late_interaction_scores(
    query: MultiVectorEncoding,
    index: DocumentIndex,
    *,
    document_batch_size: int,
) -> Tensor:
    score_chunks = []
    for start in range(0, len(index.document_ids), document_batch_size):
        stop = min(start + document_batch_size, len(index.document_ids))
        score_chunks.append(
            maxsim(
                query.embeddings,
                index.embeddings[start:stop],
                query_mask=query.mask,
                document_mask=index.mask[start:stop],
            )
        )
    scores = torch.cat(score_chunks, dim=1)
    if tuple(scores.shape) != (1, len(index.document_ids)):
        raise RuntimeError("late-interaction score shape drifted")
    return scores


def _score_matrix_fingerprint(
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    scores: Tensor,
) -> str:
    cpu_scores = scores.detach().to(device="cpu", dtype=torch.float32)
    digest = sha256()
    digest.update(b"colpali-triton-score-matrix-v1\0")
    for identifier in query_ids:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"\xff")
    for identifier in document_ids:
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"\xff")
    digest.update(cpu_scores.contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _build_scorer_result(
    *,
    name: str,
    scores: Tensor,
    rankings: Mapping[str, Tuple[str, ...]],
    relevant: Mapping[str, FrozenSet[str]],
    k: int,
    warmup_count: int,
    scoring_times: Sequence[float],
    ranking_times: Sequence[float],
    query_encoding_times: Sequence[float],
    index_payload_bytes: int,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
) -> ScorerResult:
    quality = retrieval_metrics(
        rankings,
        relevant,
        k=k,
        include_mrr=True,
    )
    assert quality.mrr_at_k is not None
    scoring_seconds = float(sum(scoring_times))
    ranking_seconds = float(sum(ranking_times))
    query_encoding_seconds = float(sum(query_encoding_times))
    latencies = (
        np.asarray(scoring_times)
        + np.asarray(ranking_times)
        + np.asarray(query_encoding_times)
    ) * 1000.0
    pairs = len(query_ids) * len(document_ids)
    end_to_end_seconds = (
        scoring_seconds + ranking_seconds + query_encoding_seconds
    )
    return ScorerResult(
        name=name,
        k=k,
        num_queries=len(query_ids),
        num_documents=len(document_ids),
        warmup_count=warmup_count,
        ndcg_at_k=quality.ndcg_at_k,
        recall_at_k=quality.recall_at_k,
        mrr_at_k=quality.mrr_at_k,
        scoring_seconds=scoring_seconds,
        ranking_seconds=ranking_seconds,
        shared_query_encoding_seconds=query_encoding_seconds,
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        scoring_pairs_per_second=(
            pairs / scoring_seconds if scoring_seconds > 0.0 else 0.0
        ),
        end_to_end_pairs_per_second=(
            pairs / end_to_end_seconds if end_to_end_seconds > 0.0 else 0.0
        ),
        logical_index_payload_bytes=index_payload_bytes,
        score_matrix_sha256=_score_matrix_fingerprint(
            query_ids, document_ids, scores
        ),
        top_k_rankings={
            query_id: ranking[:k]
            for query_id, ranking in rankings.items()
        },
    )


def evaluate_document_index(
    model: nn.Module,
    processor: object,
    dataset: MultimodalRetrievalDataset,
    index: DocumentIndex,
    *,
    k: int = 5,
    warmup_count: int = 1,
    score_document_batch_size: int = 16,
) -> MultimodalEvaluationResult:
    """Encode each query once and evaluate both scorers over the full subset."""

    if not isinstance(dataset, MultimodalRetrievalDataset):
        raise TypeError("dataset must be a MultimodalRetrievalDataset")
    if not isinstance(index, DocumentIndex):
        raise TypeError("index must be a DocumentIndex")
    if tuple(
        document.document_id for document in dataset.documents
    ) != index.document_ids:
        raise ValueError("dataset and index document orders do not match")
    for label, value, minimum in (
        ("k", k, 1),
        ("warmup_count", warmup_count, 0),
        (
            "score_document_batch_size",
            score_document_batch_size,
            1,
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(f"{label} must be an integer >= {minimum}")
    process_queries = getattr(processor, "process_queries", None)
    if not callable(process_queries):
        raise TypeError("processor must expose process_queries")

    with torch.inference_mode():
        for warmup_index in range(warmup_count):
            query = dataset.queries[warmup_index % len(dataset.queries)]
            encoded = _encode_query(
                model, processor, query.text, index.device
            )
            _late_interaction_scores(
                encoded,
                index,
                document_batch_size=score_document_batch_size,
            )
            query_mean = masked_mean_pool(encoded)
            torch.matmul(query_mean, index.mean_embeddings.T)
            _synchronize(index.device)

    query_ids = tuple(query.query_id for query in dataset.queries)
    relevant = {
        query.query_id: query.relevant_document_ids
        for query in dataset.queries
    }
    late_rows = []
    mean_rows = []
    query_encoding_times = []
    late_scoring_times = []
    mean_scoring_times = []
    late_ranking_times = []
    mean_ranking_times = []
    late_rankings: Dict[str, Tuple[str, ...]] = {}
    mean_rankings: Dict[str, Tuple[str, ...]] = {}

    with torch.inference_mode():
        for query in dataset.queries:
            _synchronize(index.device)
            query_started = perf_counter()
            encoded = _encode_query(
                model, processor, query.text, index.device
            )
            _synchronize(index.device)
            query_elapsed = perf_counter() - query_started

            late_started = perf_counter()
            late_device = _late_interaction_scores(
                encoded,
                index,
                document_batch_size=score_document_batch_size,
            )
            late_cpu = late_device.to(device="cpu", dtype=torch.float32)
            _synchronize(index.device)
            late_elapsed = perf_counter() - late_started

            mean_started = perf_counter()
            query_mean = masked_mean_pool(encoded)
            mean_device = torch.matmul(
                query_mean, index.mean_embeddings.T
            )
            mean_cpu = mean_device.to(device="cpu", dtype=torch.float32)
            _synchronize(index.device)
            mean_elapsed = perf_counter() - mean_started

            late_ranking_started = perf_counter()
            late_ranking = rank_scores(
                (query.query_id,), index.document_ids, late_cpu
            )[query.query_id]
            late_ranking_elapsed = (
                perf_counter() - late_ranking_started
            )
            mean_ranking_started = perf_counter()
            mean_ranking = rank_scores(
                (query.query_id,), index.document_ids, mean_cpu
            )[query.query_id]
            mean_ranking_elapsed = (
                perf_counter() - mean_ranking_started
            )

            if not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    query_elapsed,
                    late_elapsed,
                    mean_elapsed,
                    late_ranking_elapsed,
                    mean_ranking_elapsed,
                )
            ):
                raise RuntimeError("performance clock returned invalid values")
            query_encoding_times.append(query_elapsed)
            late_scoring_times.append(late_elapsed)
            mean_scoring_times.append(mean_elapsed)
            late_ranking_times.append(late_ranking_elapsed)
            mean_ranking_times.append(mean_ranking_elapsed)
            late_rows.append(late_cpu.squeeze(0))
            mean_rows.append(mean_cpu.squeeze(0))
            late_rankings[query.query_id] = late_ranking
            mean_rankings[query.query_id] = mean_ranking

    late_scores = torch.stack(late_rows)
    mean_scores = torch.stack(mean_rows)
    late_result = _build_scorer_result(
        name="late_interaction_maxsim",
        scores=late_scores,
        rankings=late_rankings,
        relevant=relevant,
        k=k,
        warmup_count=warmup_count,
        scoring_times=late_scoring_times,
        ranking_times=late_ranking_times,
        query_encoding_times=query_encoding_times,
        index_payload_bytes=index.multi_vector_payload_bytes,
        query_ids=query_ids,
        document_ids=index.document_ids,
    )
    mean_result = _build_scorer_result(
        name="l2_normalized_masked_mean",
        scores=mean_scores,
        rankings=mean_rankings,
        relevant=relevant,
        k=k,
        warmup_count=warmup_count,
        scoring_times=mean_scoring_times,
        ranking_times=mean_ranking_times,
        query_encoding_times=query_encoding_times,
        index_payload_bytes=index.mean_vector_payload_bytes,
        query_ids=query_ids,
        document_ids=index.document_ids,
    )
    return MultimodalEvaluationResult(
        indexing=index.indexing_record(),
        shared_query_encoding_seconds=float(sum(query_encoding_times)),
        late_interaction=late_result,
        mean_pooling=mean_result,
    )


__all__ = [
    "DocumentIndex",
    "MultimodalEvaluationResult",
    "ScorerResult",
    "encode_document_index",
    "evaluate_document_index",
    "masked_mean_pool",
]
