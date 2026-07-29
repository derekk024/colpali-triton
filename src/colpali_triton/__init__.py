"""Core components for the scoped ColPali reproduction."""

from colpali_triton.adaptation import (
    AdaptationMode,
    LoRASpec,
    TrainabilityReport,
    apply_lora,
    configure_frozen_backbone_ablation,
    freeze_for_inference,
)
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
from colpali_triton.modeling import (
    ColPaliBatch,
    ColPaliEncoder,
    EncodingKind,
    MultiVectorEncoding,
    RetrievalBackbone,
    late_interaction_scores,
)
from colpali_triton.phase7_config import (
    PHASE7_CONFIG,
    Phase7Config,
    load_phase7_config,
)
from colpali_triton.processing import ColPaliProcessor
from colpali_triton.synthetic import (
    SyntheticOverfitConfig,
    SyntheticOverfitResult,
    run_synthetic_overfit,
)
from colpali_triton.triton_maxsim import (
    DEFAULT_TRITON_MAXSIM_CONFIG,
    TritonMaxSimConfig,
    maxsim_triton,
    triton_is_available,
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
    "AdaptationMode",
    "BM25ChunkMaxIndex",
    "ColPaliBatch",
    "ColPaliEncoder",
    "ColPaliProcessor",
    "DatasetValidationError",
    "DenseChunkMaxIndex",
    "DEFAULT_TRITON_MAXSIM_CONFIG",
    "EncodingKind",
    "EvaluationResult",
    "LoRASpec",
    "MultiVectorEncoding",
    "PHASE7_CONFIG",
    "Phase7Config",
    "RetrievalDataset",
    "RetrievalBackbone",
    "RetrievalMetrics",
    "SentenceTransformerTextEncoder",
    "SyntheticOverfitConfig",
    "SyntheticOverfitResult",
    "TextDocument",
    "TextQuery",
    "TrainabilityReport",
    "TritonMaxSimConfig",
    "VidoreDatasetSpec",
    "VidoreManifest",
    "apply_lora",
    "chunk_text",
    "configure_frozen_backbone_ablation",
    "evaluate_retrieval",
    "freeze_for_inference",
    "hardest_in_batch_contrastive_loss",
    "late_interaction_scores",
    "load_manifest",
    "load_phase7_config",
    "load_vidore_text_dataset",
    "maxsim",
    "maxsim_nested",
    "maxsim_triton",
    "maxsim_vectorized",
    "rank_scores",
    "retrieval_metrics",
    "run_synthetic_overfit",
    "tokenize",
    "triton_is_available",
]
__version__ = "1.0.0"
