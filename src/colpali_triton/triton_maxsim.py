"""Triton forward kernel for ColPali late-interaction MaxSim.

Triton is an optional dependency. Importing this module is safe in CPU-only
environments; :func:`maxsim_triton` reports a focused error if execution is
requested without CUDA or Triton.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from colpali_triton.maxsim import (
    _prepare_inputs,
    _restore_batch_shape,
    maxsim_vectorized,
)

try:
    import triton
    import triton.language as tl
except Exception as error:  # pragma: no cover - depends on the host install
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: Optional[BaseException] = error
else:
    _TRITON_IMPORT_ERROR = None


_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK_SIZES = frozenset((16, 32, 64, 128))


def _validate_block_size(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in _BLOCK_SIZES:
        allowed = ", ".join(str(block) for block in sorted(_BLOCK_SIZES))
        raise ValueError(f"{name} must be one of {allowed}; got {value}")


@dataclass(frozen=True)
class TritonMaxSimConfig:
    """Launch-time tuning knobs for the MaxSim forward kernels.

    Blocks are deliberately explicit rather than autotuned so benchmark
    records can identify the exact configuration that produced every result.
    The embedding block may exceed the logical embedding width; masked loads
    fill the remaining columns with zero.
    """

    block_query_tokens: int = 16
    block_document_tokens: int = 64
    block_embedding_dimension: int = 128
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        _validate_block_size(
            self.block_query_tokens, "block_query_tokens"
        )
        _validate_block_size(
            self.block_document_tokens, "block_document_tokens"
        )
        _validate_block_size(
            self.block_embedding_dimension, "block_embedding_dimension"
        )
        if (
            isinstance(self.num_warps, bool)
            or not isinstance(self.num_warps, int)
        ):
            raise TypeError("num_warps must be an integer")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1, 2, 4, or 8")
        if (
            isinstance(self.num_stages, bool)
            or not isinstance(self.num_stages, int)
        ):
            raise TypeError("num_stages must be an integer")
        if not 1 <= self.num_stages <= 5:
            raise ValueError("num_stages must be between 1 and 5")


DEFAULT_TRITON_MAXSIM_CONFIG = TritonMaxSimConfig()


def triton_is_available() -> bool:
    """Return whether the optional Triton Python package imported cleanly."""

    return triton is not None and tl is not None


def _require_triton() -> None:
    if triton_is_available():
        return
    detail = (
        f" ({type(_TRITON_IMPORT_ERROR).__name__}: "
        f"{_TRITON_IMPORT_ERROR})"
        if _TRITON_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(
        "maxsim_triton requires the optional Triton dependency; install the "
        f"project's CUDA dependencies in the NVIDIA environment{detail}"
    ) from _TRITON_IMPORT_ERROR


if triton is not None and tl is not None:

    @triton.jit
    def _maxsim_partial_kernel(
        query_pointer,
        document_pointer,
        query_mask_pointer,
        document_mask_pointer,
        partial_pointer,
        query_batch_size,
        document_batch_size,
        query_token_count,
        document_token_count,
        embedding_dimension,
        query_batch_stride,
        query_token_stride,
        query_dimension_stride,
        document_batch_stride,
        document_token_stride,
        document_dimension_stride,
        query_mask_batch_stride,
        query_mask_token_stride,
        document_mask_batch_stride,
        document_mask_token_stride,
        partial_pair_stride,
        partial_query_block_stride,
        BLOCK_QUERY_TOKENS: tl.constexpr,
        BLOCK_DOCUMENT_TOKENS: tl.constexpr,
        BLOCK_EMBEDDING_DIMENSION: tl.constexpr,
    ):
        pair_index = tl.program_id(axis=0)
        query_block_index = tl.program_id(axis=1)
        query_batch_index = pair_index // document_batch_size
        document_batch_index = pair_index % document_batch_size

        query_offsets = (
            query_block_index * BLOCK_QUERY_TOKENS
            + tl.arange(0, BLOCK_QUERY_TOKENS)
        )
        embedding_offsets = tl.arange(
            0, BLOCK_EMBEDDING_DIMENSION
        )
        query_in_bounds = query_offsets < query_token_count
        query_mask_values = tl.load(
            query_mask_pointer
            + query_batch_index * query_mask_batch_stride
            + query_offsets * query_mask_token_stride,
            mask=query_in_bounds,
            other=False,
        )
        query_is_active = query_in_bounds & query_mask_values
        query_values = tl.load(
            query_pointer
            + query_batch_index * query_batch_stride
            + query_offsets[:, None] * query_token_stride
            + embedding_offsets[None, :] * query_dimension_stride,
            mask=(
                query_is_active[:, None]
                & (embedding_offsets[None, :] < embedding_dimension)
            ),
            other=0.0,
        )

        maxima = tl.full(
            (BLOCK_QUERY_TOKENS,), -float("inf"), tl.float32
        )
        for document_start in range(
            0, document_token_count, BLOCK_DOCUMENT_TOKENS
        ):
            document_offsets = (
                document_start
                + tl.arange(0, BLOCK_DOCUMENT_TOKENS)
            )
            document_in_bounds = (
                document_offsets < document_token_count
            )
            document_mask_values = tl.load(
                document_mask_pointer
                + document_batch_index * document_mask_batch_stride
                + document_offsets * document_mask_token_stride,
                mask=document_in_bounds,
                other=False,
            )
            document_is_active = (
                document_in_bounds & document_mask_values
            )
            document_values = tl.load(
                document_pointer
                + document_batch_index * document_batch_stride
                + embedding_offsets[:, None]
                * document_dimension_stride
                + document_offsets[None, :] * document_token_stride,
                mask=(
                    (
                        embedding_offsets[:, None]
                        < embedding_dimension
                    )
                    & document_is_active[None, :]
                ),
                other=0.0,
            )
            similarities = tl.dot(
                query_values,
                document_values,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            similarities = tl.where(
                document_is_active[None, :],
                similarities,
                -float("inf"),
            )
            maxima = tl.maximum(
                maxima, tl.max(similarities, axis=1)
            )

        maxima = tl.where(query_is_active, maxima, 0.0)
        partial_score = tl.sum(maxima, axis=0)
        tl.store(
            partial_pointer
            + pair_index * partial_pair_stride
            + query_block_index * partial_query_block_stride,
            partial_score,
        )

    @triton.jit
    def _maxsim_reduce_kernel(
        partial_pointer,
        output_pointer,
        query_block_count,
        partial_pair_stride,
        partial_query_block_stride,
        BLOCK_QUERY_BLOCKS: tl.constexpr,
    ):
        pair_index = tl.program_id(axis=0)
        query_block_offsets = tl.arange(0, BLOCK_QUERY_BLOCKS)
        partial_values = tl.load(
            partial_pointer
            + pair_index * partial_pair_stride
            + query_block_offsets * partial_query_block_stride,
            mask=query_block_offsets < query_block_count,
            other=0.0,
        )
        tl.store(
            output_pointer + pair_index,
            tl.sum(partial_values, axis=0),
        )

else:
    _maxsim_partial_kernel = None
    _maxsim_reduce_kernel = None


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_execution(
    queries: Tensor,
    documents: Tensor,
    config: TritonMaxSimConfig,
) -> None:
    if config.block_embedding_dimension < queries.shape[-1]:
        raise ValueError(
            "block_embedding_dimension must be at least the logical "
            f"embedding dimension {queries.shape[-1]}; got "
            f"{config.block_embedding_dimension}"
        )
    if not queries.is_cuda:
        raise RuntimeError(
            "maxsim_triton requires query and document embeddings on a "
            "CUDA device"
        )
    if torch.version.cuda is None:
        raise RuntimeError(
            "maxsim_triton requires a CUDA-enabled PyTorch build"
        )
    if queries.dtype not in _TRITON_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _TRITON_DTYPES)
        raise TypeError(
            "maxsim_triton supports CUDA embedding dtypes "
            f"{supported}; got {queries.dtype}"
        )


def _launch_maxsim(
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    config: TritonMaxSimConfig,
) -> Tensor:
    _require_triton()
    if _maxsim_partial_kernel is None or _maxsim_reduce_kernel is None:
        raise AssertionError("Triton kernels were not initialized")

    query_batch_size, query_token_count, embedding_dimension = queries.shape
    document_batch_size, document_token_count, _ = documents.shape
    pair_count = query_batch_size * document_batch_size
    query_block_count = (
        query_token_count + config.block_query_tokens - 1
    ) // config.block_query_tokens
    partials = torch.empty(
        (pair_count, query_block_count),
        dtype=torch.float32,
        device=queries.device,
    )

    partial_grid = (pair_count, query_block_count)
    _maxsim_partial_kernel[partial_grid](
        queries,
        documents,
        query_mask,
        document_mask,
        partials,
        query_batch_size,
        document_batch_size,
        query_token_count,
        document_token_count,
        embedding_dimension,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        documents.stride(0),
        documents.stride(1),
        documents.stride(2),
        query_mask.stride(0),
        query_mask.stride(1),
        document_mask.stride(0),
        document_mask.stride(1),
        partials.stride(0),
        partials.stride(1),
        BLOCK_QUERY_TOKENS=config.block_query_tokens,
        BLOCK_DOCUMENT_TOKENS=config.block_document_tokens,
        BLOCK_EMBEDDING_DIMENSION=(
            config.block_embedding_dimension
        ),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    scores = torch.empty(
        pair_count, dtype=torch.float32, device=queries.device
    )
    reduction_block = _next_power_of_two(query_block_count)
    _maxsim_reduce_kernel[(pair_count,)](
        partials,
        scores,
        query_block_count,
        partials.stride(0),
        partials.stride(1),
        BLOCK_QUERY_BLOCKS=reduction_block,
        num_warps=1,
        num_stages=1,
    )
    return scores.view(query_batch_size, document_batch_size)


def maxsim_triton(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    *,
    query_mask: Optional[Tensor] = None,
    document_mask: Optional[Tensor] = None,
    config: Optional[TritonMaxSimConfig] = None,
) -> Tensor:
    """Compute all-pairs MaxSim with a fused Triton forward kernel.

    The CUDA path tiles query tokens, document tokens, and the embedding
    dimension. It reduces document-token maxima inside each program and stores
    only one float32 partial per query tile and query-document pair, avoiding
    the reference implementation's full ``[Bq, Bd, Nq, Nd]`` tensor.

    FP16 and BF16 inputs use FP32 dot-product accumulation and FP32 score
    accumulation. FP32 inputs also produce FP32 scores and request IEEE input
    precision rather than Triton's default TF32 mode.

    The project specifies a forward Triton kernel; its backward kernel is a
    stretch goal. To preserve the public scorer's autograd contract, calls
    whose embeddings require gradients route through
    :func:`maxsim_vectorized`. Inference tensors execute the Triton kernels.
    """

    if config is None:
        config = DEFAULT_TRITON_MAXSIM_CONFIG
    elif not isinstance(config, TritonMaxSimConfig):
        raise TypeError("config must be a TritonMaxSimConfig or None")

    (
        queries,
        documents,
        prepared_query_mask,
        prepared_document_mask,
        query_was_unbatched,
        document_was_unbatched,
    ) = _prepare_inputs(
        query_embeddings,
        document_embeddings,
        query_mask,
        document_mask,
    )
    _validate_execution(queries, documents, config)
    _require_triton()

    if torch.is_grad_enabled() and (
        queries.requires_grad or documents.requires_grad
    ):
        return maxsim_vectorized(
            query_embeddings,
            document_embeddings,
            query_mask=query_mask,
            document_mask=document_mask,
        )

    scores = _launch_maxsim(
        queries,
        documents,
        prepared_query_mask,
        prepared_document_mask,
        config,
    )
    return _restore_batch_shape(
        scores, query_was_unbatched, document_was_unbatched
    )


__all__ = [
    "DEFAULT_TRITON_MAXSIM_CONFIG",
    "TritonMaxSimConfig",
    "maxsim_triton",
    "triton_is_available",
]
