from pathlib import Path

import pytest
import torch
from transformers import PaliGemmaConfig

from colpali_triton.adaptation import apply_lora, resolve_lora_targets
from colpali_triton.modeling import EncodingKind, MultiVectorEncoding
from colpali_triton.paligemma import ColPali


def _tiny_config() -> PaliGemmaConfig:
    return PaliGemmaConfig(
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 4,
            "patch_size": 2,
            "projection_dim": 16,
        },
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "hidden_activation": "gelu_pytorch_tanh",
            "max_position_embeddings": 32,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        image_token_index=63,
        vocab_size=64,
        projection_dim=16,
        hidden_size=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _query_inputs() -> tuple:
    return (
        torch.tensor([[1, 3, 4, 0]], dtype=torch.int64),
        torch.tensor([[True, True, True, False]]),
    )


def _document_inputs() -> tuple:
    return (
        torch.tensor([[63, 63, 63, 63, 1, 3]], dtype=torch.int64),
        torch.ones(1, 6, dtype=torch.bool),
        torch.rand(1, 3, 4, 4),
    )


def test_query_forward_uses_final_state_projection_and_mask() -> None:
    torch.manual_seed(7)
    model = ColPali(_tiny_config()).eval()
    input_ids, attention_mask = _query_inputs()

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert isinstance(output, MultiVectorEncoding)
    assert output.kind is EncodingKind.QUERY
    assert output.embeddings.shape == (1, 4, 128)
    torch.testing.assert_close(
        output.embeddings[output.mask].norm(dim=-1),
        torch.ones(3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        output.embeddings[~output.mask],
        torch.zeros_like(output.embeddings[~output.mask]),
    )


def test_document_forward_keeps_image_and_prompt_positions() -> None:
    model = ColPali(_tiny_config()).eval()
    input_ids, attention_mask, pixel_values = _document_inputs()

    with torch.no_grad():
        output = model(input_ids, attention_mask, pixel_values)

    assert output.kind is EncodingKind.DOCUMENT
    assert output.embeddings.shape == (1, 6, 128)
    assert output.mask.all()
    torch.testing.assert_close(
        output.embeddings.norm(dim=-1),
        torch.ones(1, 6),
        atol=1e-5,
        rtol=1e-5,
    )


def test_state_dict_layout_matches_released_checkpoint_contract() -> None:
    state_keys = set(ColPali(_tiny_config()).state_dict())

    assert "custom_text_proj.weight" in state_keys
    assert "custom_text_proj.bias" in state_keys
    assert any(
        key.startswith("model.language_model.") for key in state_keys
    )
    assert any(key.startswith("model.vision_tower.") for key in state_keys)
    assert not any(key.startswith("backbone.") for key in state_keys)


def test_tiny_model_save_and_from_pretrained_round_trip(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = ColPali(_tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = ColPali.from_pretrained(
        tmp_path,
        local_files_only=True,
    ).eval()

    assert loaded.retrieval_embedding_dim == 128
    assert loaded.hidden_size == 16
    torch.testing.assert_close(
        loaded.custom_text_proj.weight,
        model.custom_text_proj.weight,
    )
    input_ids, attention_mask = _query_inputs()
    with torch.no_grad():
        expected = model(input_ids, attention_mask)
        actual = loaded(input_ids, attention_mask)
    torch.testing.assert_close(actual.embeddings, expected.embeddings)
    assert torch.equal(actual.mask, expected.mask)


def test_low_memory_load_materializes_and_reties_language_head(
    tmp_path: Path,
) -> None:
    model = ColPali(_tiny_config()).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    loaded = ColPali.from_pretrained(
        tmp_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    assert not [
        name
        for name, parameter in loaded.named_parameters()
        if parameter.device.type == "meta"
    ]
    assert (
        loaded.model.get_input_embeddings().weight
        is loaded.model.get_output_embeddings().weight
    )
    loaded.to("cpu")


def test_tiny_model_resolves_only_language_and_projection_lora() -> None:
    model = ColPali(_tiny_config())

    targets = resolve_lora_targets(model)

    assert len(targets) == 8
    assert "custom_text_proj" in targets
    assert all("vision_tower" not in target for target in targets)
    assert all("multi_modal_projector" not in target for target in targets)


def test_peft_wrapped_tiny_model_preserves_encoding_contract() -> None:
    model, report = apply_lora(
        ColPali(_tiny_config()),
        expected_target_count=8,
    )
    input_ids, attention_mask = _query_inputs()

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert isinstance(output, MultiVectorEncoding)
    assert output.kind is EncodingKind.QUERY
    assert output.embeddings.shape == (1, 4, 128)
    assert report.trainable_parameters > 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"normalization_eps": 0.0},
        {"normalization_eps": float("nan")},
        {"normalization_eps": True},
    ),
)
def test_colpali_rejects_invalid_normalization_epsilon(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="normalization_eps"):
        ColPali(_tiny_config(), **kwargs)


def test_colpali_rejects_non_128_retrieval_config() -> None:
    config = _tiny_config()
    config.retrieval_embedding_dim = 64

    with pytest.raises(ValueError, match="exactly 128"):
        ColPali(config)


@pytest.mark.parametrize(
    "protected_name",
    ("output_hidden_states", "return_dict", "use_cache"),
)
def test_colpali_owns_hidden_state_and_cache_controls(
    protected_name: str,
) -> None:
    model = ColPali(_tiny_config())
    input_ids, attention_mask = _query_inputs()

    with pytest.raises(ValueError, match="controlled by ColPali"):
        model(
            input_ids,
            attention_mask,
            **{protected_name: False},
        )
