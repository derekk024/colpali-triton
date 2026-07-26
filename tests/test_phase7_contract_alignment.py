"""Cross-module guards for the committed Phase 7 retrieval contract."""

from colpali_triton.adaptation import (
    LoRASpec,
    PAPER_LORA_TARGET_PATTERN,
    PAPER_LORA_TRAINABLE_PARAMETERS,
)
from colpali_triton.phase7_config import PHASE7_CONFIG
from colpali_triton.processing import (
    DEFAULT_MAX_LENGTH,
    DOCUMENT_PROMPT,
    QUERY_AUGMENTATION_COUNT,
    QUERY_AUGMENTATION_TOKEN,
    QUERY_PREFIX,
)


def test_processing_constants_match_the_pinned_manifest() -> None:
    processing = PHASE7_CONFIG.processing

    assert DOCUMENT_PROMPT == processing.document_prompt
    assert QUERY_PREFIX == processing.query_prefix
    assert QUERY_AUGMENTATION_TOKEN == processing.query_augmentation_token
    assert QUERY_AUGMENTATION_COUNT == processing.query_augmentation_count
    assert DEFAULT_MAX_LENGTH == processing.text_max_length


def test_adaptation_constants_match_the_pinned_manifest() -> None:
    contract = PHASE7_CONFIG.lora
    spec = LoRASpec()

    assert spec.rank == contract.rank
    assert spec.alpha == contract.alpha
    assert spec.dropout == contract.dropout
    assert spec.init == contract.initialization
    assert spec.bias == contract.bias
    assert spec.task_type == contract.task_type
    assert spec.target_pattern == contract.target_modules_pattern
    assert PAPER_LORA_TARGET_PATTERN == contract.target_modules_pattern
    assert (
        PAPER_LORA_TRAINABLE_PARAMETERS
        == PHASE7_CONFIG.paper_adapter.parameter_count
    )


def test_architecture_constants_match_the_modeling_contract() -> None:
    from colpali_triton.modeling import ColPaliEncoder
    from colpali_triton.paligemma import ColPali

    architecture = PHASE7_CONFIG.architecture
    assert ColPaliEncoder.EMBEDDING_DIM == architecture.projection_dimension
    assert (
        ColPali.RETRIEVAL_EMBEDDING_DIM
        == architecture.projection_dimension
    )
