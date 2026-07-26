import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest

from colpali_triton.phase7_config import (
    PHASE7_CONFIG,
    ColPaliArchitectureSpec,
    ModelArtifactSpec,
    ModelCheckpointSpec,
    huggingface_model_resolve_url,
    load_phase7_config,
    phase7_config_from_dict,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "colpali_phase7.json"
)


def _manifest() -> Dict[str, Any]:
    return PHASE7_CONFIG.to_dict()


def _nested(value: Dict[str, Any], path: Iterable[object]) -> Any:
    current: Any = value
    for part in path:
        current = current[part]
    return current


def test_official_json_matches_the_immutable_runtime_contract() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    loaded = load_phase7_config(CONFIG_PATH)

    assert raw == PHASE7_CONFIG.to_dict()
    assert loaded == PHASE7_CONFIG
    assert loaded.canonical_json == PHASE7_CONFIG.canonical_json
    assert loaded.fingerprint == (
        "e9accebbdcdf821d1bdb64606ee7427d80947237833e4eaf90e199a90b057d88"
    )


def test_checkpoint_identities_sizes_and_hashes_are_exact() -> None:
    source = PHASE7_CONFIG.source_backbone
    base = PHASE7_CONFIG.runtime_base
    adapter = PHASE7_CONFIG.paper_adapter

    assert source.repository == "google/paligemma-3b-mix-448"
    assert source.revision == "ead2d9a35598cb89119af004f5d023b311d1c4a1"
    assert source.dtype == "float32"
    assert source.parameter_count == 2_924_351_216
    assert source.total_artifact_bytes == 11_697_486_320
    assert tuple(item.size_bytes for item in source.artifacts) == (
        4_956_951_424,
        4_999_820_608,
        1_740_714_288,
    )
    assert source.artifacts[0].sha256 == (
        "570dab6f84d3b784a06707cdc4742f97545dfd57d73742bb2fcb3190a09696a4"
    )
    assert source.artifacts[1].sha256 == (
        "334b225c0ec1db8f3952121f5f67a78b37167623e2178c0babe3086fcc8ea4ad"
    )
    assert source.artifacts[2].sha256 == (
        "8c75421941def510a8c2364726d8ab36cf1a0653b355368d2e2a80766b5a4f5f"
    )

    assert base.repository == "vidore/colpaligemma-3b-mix-448-base"
    assert base.revision == "6ff0d944ea09c3ead97d2bc57427e3d4f01d192f"
    assert base.dtype == "bfloat16"
    assert base.parameter_count == 2_924_613_488
    assert base.total_artifact_bytes == 5_849_312_816
    assert tuple(item.sha256 for item in base.artifacts) == (
        "8b439a322fa98964517fca8fbf0a1d17e6e60e7e8812bc1c373ad87d654108d5",
        "ad075b39b4a24b309028f3d003a0e7cad9e45fa2dbbd267867474e77e10df814",
    )

    assert adapter.repository == "vidore/colpali"
    assert adapter.revision == "234ecbefa176542348dc5fae4f95c9736858edc2"
    assert adapter.dtype == "bfloat16"
    assert adapter.parameter_count == 39_292_928
    assert adapter.total_artifact_bytes == 78_625_112
    assert adapter.artifacts == (
        ModelArtifactSpec(
            path="adapter_model.safetensors",
            size_bytes=78_625_112,
            sha256=(
                "961b72c2b2a1bebc3e11e7d98cd173d263437aad5ef36e760e9265c184c88d64"
            ),
        ),
    )

    assert PHASE7_CONFIG.source_terms.access == "manual_gated"
    assert PHASE7_CONFIG.source_terms.license_id == "gemma"


def test_architecture_processing_and_lora_match_the_paper_contract() -> None:
    architecture = PHASE7_CONFIG.architecture
    processing = PHASE7_CONFIG.processing
    lora = PHASE7_CONFIG.lora

    assert (
        architecture.hidden_size,
        architecture.projection_dimension,
        architecture.image_size_pixels,
        architecture.patch_size_pixels,
        architecture.image_token_count,
    ) == (2048, 128, 448, 14, 1024)
    assert architecture.embedding_normalization == "l2_last_dimension"
    assert architecture.masked_embedding_value == 0.0

    assert processing.document_prompt == "Describe the image."
    assert processing.text_max_length == 50
    assert processing.query_augmentation_suffix == "<unused0>" * 5
    assert processing.format_query("Where is the total?") == (
        "Question: Where is the total?"
        "<unused0><unused0><unused0><unused0><unused0>"
    )

    assert (lora.rank, lora.alpha, lora.dropout) == (32, 32, 0.1)
    assert lora.bias == "none"
    assert lora.initialization == "gaussian"
    assert lora.task_type == "FEATURE_EXTRACTION"
    assert lora.peft_type == "LORA"
    assert not lora.inference_mode
    assert not lora.use_dora
    assert not lora.use_rslora
    assert lora.modules_to_save is None
    assert lora.target_modules_pattern == (
        r"(.*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|"
        r"v_proj|o_proj).*$|.*(custom_text_proj).*$)"
    )


def test_checkpoint_artifact_api_builds_only_revision_pinned_urls() -> None:
    base = PHASE7_CONFIG.runtime_base
    path = "model-00001-of-00002.safetensors"

    assert base.artifact(path) is base.artifacts[0]
    assert base.resolve_url(path) == (
        "https://huggingface.co/"
        "vidore/colpaligemma-3b-mix-448-base/resolve/"
        "6ff0d944ea09c3ead97d2bc57427e3d4f01d192f/"
        "model-00001-of-00002.safetensors"
    )
    assert "/main/" not in base.resolve_url(path)
    with pytest.raises(KeyError, match="has no artifact"):
        base.resolve_url("unlisted.safetensors")


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("source_backbone",),
        ("source_backbone", "artifacts", 0),
        ("source_terms",),
        ("architecture",),
        ("processing",),
        ("lora",),
    ],
)
def test_parser_rejects_unknown_keys_at_every_object_level(
    path: Tuple[object, ...],
) -> None:
    manifest = _manifest()
    target = _nested(manifest, path)
    target["unexpected"] = "drift"

    with pytest.raises(ValueError, match="unknown keys"):
        phase7_config_from_dict(manifest)


def test_parser_rejects_missing_keys_and_non_array_artifacts() -> None:
    missing = _manifest()
    del missing["runtime_base"]["revision"]
    with pytest.raises(ValueError, match="missing keys"):
        phase7_config_from_dict(missing)

    wrong_type = _manifest()
    wrong_type["runtime_base"]["artifacts"] = {}
    with pytest.raises(ValueError, match="must be a JSON array"):
        phase7_config_from_dict(wrong_type)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("revision", "main"), "full 40-character"),
        (("revision", "A" * 40), "full 40-character"),
        (("parameter_count", True), "must be an integer"),
        (("expected_artifact_count", 3), "artifact count"),
    ],
)
def test_parser_rejects_invalid_checkpoint_revision_and_counts(
    mutation: Tuple[str, object],
    match: str,
) -> None:
    manifest = _manifest()
    key, value = mutation
    manifest["runtime_base"][key] = value

    with pytest.raises(ValueError, match=match):
        phase7_config_from_dict(manifest)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("path", "../escape.safetensors", "without"),
        ("path", "weights.bin", "safetensors"),
        ("size_bytes", 0, "positive integer"),
        ("size_bytes", True, "must be an integer"),
        ("sha256", "short", "64 lowercase"),
        ("sha256", "A" * 64, "64 lowercase"),
    ],
)
def test_parser_rejects_invalid_artifact_metadata(
    field: str,
    value: object,
    match: str,
) -> None:
    manifest = _manifest()
    manifest["runtime_base"]["artifacts"][0][field] = value

    with pytest.raises(ValueError, match=match):
        phase7_config_from_dict(manifest)


def test_parser_rejects_duplicate_artifacts_and_valid_but_unpinned_hashes() -> None:
    duplicate = _manifest()
    duplicate["runtime_base"]["artifacts"][1]["path"] = (
        duplicate["runtime_base"]["artifacts"][0]["path"]
    )
    with pytest.raises(ValueError, match="paths must be unique"):
        phase7_config_from_dict(duplicate)

    changed_hash = _manifest()
    changed_hash["paper_adapter"]["artifacts"][0]["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="immutable Phase 7 checkpoint pin"):
        phase7_config_from_dict(changed_hash)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("architecture", "hidden_size", 1024, "H=2048"),
        (
            "architecture",
            "image_token_count",
            1000,
            "square patch-grid",
        ),
        ("processing", "document_prompt", "Describe.", "document_prompt"),
        ("processing", "query_augmentation_count", 4, "exactly 5"),
        ("processing", "text_max_length", 49, "exactly 50"),
        ("lora", "rank", 16, "rank must be exactly 32"),
        ("lora", "dropout", 0.0, "dropout must be exactly 0.1"),
        ("lora", "target_modules_pattern", ".*", "does not match"),
        ("lora", "use_dora", True, "use_dora must be false"),
    ],
)
def test_parser_rejects_architecture_processing_and_lora_drift(
    section: str,
    field: str,
    value: object,
    match: str,
) -> None:
    manifest = _manifest()
    manifest[section][field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        phase7_config_from_dict(manifest)


def test_parser_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_phase7_config(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            '"dropout": 0.1',
            '"dropout": NaN',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_phase7_config(nonfinite)


def test_fingerprint_is_canonical_across_mapping_key_order() -> None:
    reversed_root = dict(reversed(tuple(_manifest().items())))
    parsed = phase7_config_from_dict(reversed_root)

    assert parsed.fingerprint == PHASE7_CONFIG.fingerprint
    assert parsed.canonical_json == PHASE7_CONFIG.canonical_json


def test_configuration_values_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        PHASE7_CONFIG.architecture.hidden_size = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        PHASE7_CONFIG.runtime_base.revision = "a" * 40  # type: ignore[misc]

    with pytest.raises(ValueError, match="square patch-grid"):
        replace(PHASE7_CONFIG.architecture, image_token_count=1000)


def test_low_level_checkpoint_and_url_guards_reject_ambiguous_inputs() -> None:
    artifact = ModelArtifactSpec(
        path="weights.safetensors",
        size_bytes=1,
        sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="artifact count"):
        ModelCheckpointSpec(
            repository="owner/model",
            revision="b" * 40,
            dtype="bfloat16",
            parameter_count=1,
            expected_artifact_count=2,
            artifacts=(artifact,),
        )
    with pytest.raises(ValueError, match="full lowercase"):
        huggingface_model_resolve_url(
            "owner/model",
            "main",
            "weights.safetensors",
        )
    with pytest.raises(ValueError, match="may not contain"):
        huggingface_model_resolve_url(
            "owner/model",
            "b" * 40,
            "../weights.safetensors",
        )
    with pytest.raises(ValueError, match="nonempty"):
        PHASE7_CONFIG.processing.format_query("")
