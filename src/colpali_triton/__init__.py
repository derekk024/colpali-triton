"""Core components for the scoped ColPali reproduction."""

from colpali_triton.baselines import (
    BM25ChunkMaxIndex,
    DenseChunkMaxIndex,
    SentenceTransformerTextEncoder,
    chunk_text,
    tokenize,
)
from colpali_triton.evaluation import EvaluationResult, evaluate_retrieval
from colpali_triton.loss import hardest_in_batch_contrastive_loss
from colpali_triton.maxsim import maxsim, maxsim_nested, maxsim_vectorized
from colpali_triton.metrics import (
    RetrievalMetrics,
    rank_scores,
    retrieval_metrics,
)
from colpali_triton.synthetic import (
    SyntheticOverfitConfig,
    SyntheticOverfitResult,
    run_synthetic_overfit,
)
from colpali_triton.vidore import (
    DatasetValidationError,
    RetrievalDataset,
    TextDocument,
    TextQuery,
    VidoreDatasetSpec,
    VidoreManifest,
    load_manifest,
    load_vidore_text_dataset,
)

__all__ = [
    "BM25ChunkMaxIndex",
    "DatasetValidationError",
    "DenseChunkMaxIndex",
    "EvaluationResult",
    "RetrievalDataset",
    "RetrievalMetrics",
    "SentenceTransformerTextEncoder",
    "SyntheticOverfitConfig",
    "SyntheticOverfitResult",
    "TextDocument",
    "TextQuery",
    "VidoreDatasetSpec",
    "VidoreManifest",
    "chunk_text",
    "evaluate_retrieval",
    "hardest_in_batch_contrastive_loss",
    "load_manifest",
    "load_vidore_text_dataset",
    "maxsim",
    "maxsim_nested",
    "maxsim_vectorized",
    "rank_scores",
    "retrieval_metrics",
    "run_synthetic_overfit",
    "tokenize",
]
__version__ = "0.3.0"
