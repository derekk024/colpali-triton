"""Reference implementations of ColPali's late-interaction MaxSim score."""

from typing import Optional, Tuple

import torch
from torch import Tensor


_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _as_batched_embeddings(embeddings: Tensor, name: str) -> Tuple[Tensor, bool]:
    if not isinstance(embeddings, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if embeddings.ndim not in (2, 3):
        raise ValueError(
            f"{name} must have shape [tokens, dim] or [batch, tokens, dim]; "
            f"got {tuple(embeddings.shape)}"
        )
    if embeddings.layout != torch.strided:
        raise TypeError(f"{name} must use the torch.strided layout")
    if embeddings.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise TypeError(
            f"{name} must have a supported floating-point dtype "
            f"({supported}); got {embeddings.dtype}"
        )
    if embeddings.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonzero embedding dimension")
    if embeddings.shape[-2] == 0:
        raise ValueError(f"{name} must contain at least one padded token slot")

    was_unbatched = embeddings.ndim == 2
    batched = embeddings.unsqueeze(0) if was_unbatched else embeddings
    if batched.shape[0] == 0:
        raise ValueError(f"{name} batch dimension must be nonzero")
    return batched, was_unbatched


def _as_batched_mask(
    mask: Optional[Tensor],
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
    was_unbatched: bool,
    device: torch.device,
) -> Tensor:
    if mask is None:
        return torch.ones(
            (batch_size, sequence_length), dtype=torch.bool, device=device
        )
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != device:
        raise ValueError(f"{name} must be on device {device}; got {mask.device}")

    if was_unbatched and tuple(mask.shape) == (sequence_length,):
        return mask.unsqueeze(0)

    expected_shape = (batch_size, sequence_length)
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {tuple(mask.shape)}"
        )
    return mask


def _prepare_inputs(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    query_mask: Optional[Tensor],
    document_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, bool, bool]:
    queries, query_was_unbatched = _as_batched_embeddings(
        query_embeddings, "query_embeddings"
    )
    documents, document_was_unbatched = _as_batched_embeddings(
        document_embeddings, "document_embeddings"
    )

    if queries.shape[-1] != documents.shape[-1]:
        raise ValueError(
            "query_embeddings and document_embeddings must have the same "
            f"embedding dimension; got {queries.shape[-1]} and {documents.shape[-1]}"
        )
    if queries.dtype != documents.dtype:
        raise TypeError(
            "query_embeddings and document_embeddings must have the same dtype; "
            f"got {queries.dtype} and {documents.dtype}"
        )
    if queries.device != documents.device:
        raise ValueError(
            "query_embeddings and document_embeddings must be on the same device; "
            f"got {queries.device} and {documents.device}"
        )

    prepared_query_mask = _as_batched_mask(
        query_mask,
        name="query_mask",
        batch_size=queries.shape[0],
        sequence_length=queries.shape[1],
        was_unbatched=query_was_unbatched,
        device=queries.device,
    )
    prepared_document_mask = _as_batched_mask(
        document_mask,
        name="document_mask",
        batch_size=documents.shape[0],
        sequence_length=documents.shape[1],
        was_unbatched=document_was_unbatched,
        device=documents.device,
    )
    return (
        queries,
        documents,
        prepared_query_mask,
        prepared_document_mask,
        query_was_unbatched,
        document_was_unbatched,
    )


def _restore_batch_shape(
    scores: Tensor, query_was_unbatched: bool, document_was_unbatched: bool
) -> Tensor:
    if query_was_unbatched and document_was_unbatched:
        return scores[0, 0]
    if query_was_unbatched:
        return scores[0]
    if document_was_unbatched:
        return scores[:, 0]
    return scores


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def maxsim_nested(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    query_mask: Optional[Tensor] = None,
    document_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute all-pairs MaxSim scores with explicit Python and token loops.

    This implementation is intentionally slow and transparent. It is the
    correctness oracle for optimized implementations and remains fully
    differentiable for finite scores.
    """

    (
        queries,
        documents,
        prepared_query_mask,
        prepared_document_mask,
        query_was_unbatched,
        document_was_unbatched,
    ) = _prepare_inputs(
        query_embeddings, document_embeddings, query_mask, document_mask
    )

    query_batch_scores = []
    accumulation_dtype = _accumulation_dtype(queries.dtype)
    negative_infinity = queries.new_tensor(
        float("-inf"), dtype=accumulation_dtype
    )

    for query_index in range(queries.shape[0]):
        document_batch_scores = []
        for document_index in range(documents.shape[0]):
            query_token_scores = []
            for query_token_index in range(queries.shape[1]):
                if not bool(
                    prepared_query_mask[query_index, query_token_index].item()
                ):
                    continue

                dot_products = []
                for document_token_index in range(documents.shape[1]):
                    if bool(
                        prepared_document_mask[
                            document_index, document_token_index
                        ].item()
                    ):
                        dot_products.append(
                            torch.dot(
                                queries[query_index, query_token_index],
                                documents[document_index, document_token_index],
                            )
                        )

                if dot_products:
                    query_token_scores.append(
                        torch.amax(torch.stack(dot_products), dim=0)
                    )
                else:
                    graph_zero = (
                        queries[query_index, :0].sum(dtype=accumulation_dtype)
                        + documents[document_index, :0].sum(
                            dtype=accumulation_dtype
                        )
                    )
                    query_token_scores.append(graph_zero + negative_infinity)

            if query_token_scores:
                document_batch_scores.append(
                    torch.stack(query_token_scores).sum(
                        dtype=accumulation_dtype
                    )
                )
            else:
                document_batch_scores.append(
                    queries[query_index, :0].sum(dtype=accumulation_dtype)
                    + documents[document_index, :0].sum(
                        dtype=accumulation_dtype
                    )
                )
        query_batch_scores.append(torch.stack(document_batch_scores))

    scores = torch.stack(query_batch_scores)
    return _restore_batch_shape(
        scores, query_was_unbatched, document_was_unbatched
    )


def maxsim_vectorized(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    query_mask: Optional[Tensor] = None,
    document_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute all-pairs MaxSim scores with vectorized PyTorch operations.

    Queries use shape ``[Bq, Nq, D]`` and documents use
    ``[Bd, Nd, D]``. The returned matrix has shape ``[Bq, Bd]``.
    Either input may omit its batch dimension; the corresponding output
    dimension is then removed.

    This reference materializes the ``[Bq, Bd, Nq, Nd]`` similarity tensor.
    A later Triton kernel will fuse the document-token reduction to avoid that
    allocation.
    """

    (
        queries,
        documents,
        prepared_query_mask,
        prepared_document_mask,
        query_was_unbatched,
        document_was_unbatched,
    ) = _prepare_inputs(
        query_embeddings, document_embeddings, query_mask, document_mask
    )

    masked_queries = queries.masked_fill(
        ~prepared_query_mask[..., None], 0.0
    )
    masked_documents = documents.masked_fill(
        ~prepared_document_mask[..., None], 0.0
    )
    similarities = torch.matmul(
        masked_queries.unsqueeze(1),
        masked_documents.transpose(-2, -1).unsqueeze(0),
    )
    similarities = similarities.masked_fill(
        ~prepared_document_mask[None, :, None, :], float("-inf")
    )
    maxima = similarities.amax(dim=-1)
    maxima = maxima.masked_fill(~prepared_query_mask[:, None, :], 0.0)
    scores = maxima.sum(
        dim=-1, dtype=_accumulation_dtype(queries.dtype)
    )

    return _restore_batch_shape(
        scores, query_was_unbatched, document_was_unbatched
    )


def maxsim(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    query_mask: Optional[Tensor] = None,
    document_mask: Optional[Tensor] = None,
) -> Tensor:
    """Compute MaxSim with the default vectorized PyTorch implementation."""

    return maxsim_vectorized(
        query_embeddings,
        document_embeddings,
        query_mask=query_mask,
        document_mask=document_mask,
    )


__all__ = ["maxsim", "maxsim_nested", "maxsim_vectorized"]
