import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from colpali_triton.vidore import (
    DOCVQA_SPEC,
    INFOVQA_SPEC,
    PHASE6_MANIFEST,
    VIDORE_DATASETS,
    DatasetValidationError,
    ParquetFileSpec,
    RetrievalDataset,
    TextDocument,
    TextQuery,
    VidoreDatasetSpec,
    huggingface_resolve_url,
    load_manifest,
    load_vidore_text_dataset,
    manifest_from_dict,
)


def _file_spec(path: str) -> ParquetFileSpec:
    return ParquetFileSpec(
        path=path,
        size_bytes=1,
        sha256="a" * 64,
    )


def _fixture_spec(
    *,
    document_count: int = 3,
    query_count: int = 2,
    qrel_count: int = 3,
    ocr_count: Optional[int] = None,
) -> VidoreDatasetSpec:
    return VidoreDatasetSpec(
        name="fixture",
        split="test",
        beir_repository="local/beir",
        beir_revision="b" * 40,
        corpus_file=_file_spec("corpus/test.parquet"),
        queries_file=_file_spec("queries/test.parquet"),
        qrels_file=_file_spec("qrels/test.parquet"),
        ocr_repository="local/ocr",
        ocr_revision="c" * 40,
        ocr_file=_file_spec("data/test.parquet"),
        expected_document_count=document_count,
        expected_query_count=query_count,
        expected_qrel_count=qrel_count,
        expected_ocr_row_count=(
            qrel_count if ocr_count is None else ocr_count
        ),
    )


def _write_table(path: Path, values: Mapping[str, Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.table(values), path)


def _write_fixture(
    root: Path,
    spec: VidoreDatasetSpec,
    *,
    corpus_ids: Sequence[object] = (10, 11, 12),
    query_rows: Sequence[Tuple[object, object]] = (
        (100, "shared question"),
        (101, "other question"),
    ),
    qrel_rows: Sequence[Tuple[object, object, object]] = (
        (100, 10, 1),
        (100, 11, 1),
        (101, 12, 1),
    ),
    ocr_rows: Sequence[Tuple[object, object]] = (
        ("shared question", "page ten"),
        ("shared question", "page eleven"),
        ("other question", "page twelve"),
    ),
) -> Callable[[str, str, str], str]:
    destinations: Dict[Tuple[str, str, str], Path] = {}

    def destination(repository: str, revision: str, path: str) -> Path:
        result = (
            root
            / repository.replace("/", "--")
            / revision
            / path
        )
        destinations[(repository, revision, path)] = result
        return result

    _write_table(
        destination(
            spec.beir_repository,
            spec.beir_revision,
            spec.corpus_file.path,
        ),
        {
            "corpus-id": corpus_ids,
            "image": [b"image bytes must not be read"] * len(corpus_ids),
        },
    )
    _write_table(
        destination(
            spec.beir_repository,
            spec.beir_revision,
            spec.queries_file.path,
        ),
        {
            "query-id": [row[0] for row in query_rows],
            "query": [row[1] for row in query_rows],
        },
    )
    _write_table(
        destination(
            spec.beir_repository,
            spec.beir_revision,
            spec.qrels_file.path,
        ),
        {
            "query-id": [row[0] for row in qrel_rows],
            "corpus-id": [row[1] for row in qrel_rows],
            "score": [row[2] for row in qrel_rows],
        },
    )
    _write_table(
        destination(
            spec.ocr_repository,
            spec.ocr_revision,
            spec.ocr_file.path,
        ),
        {
            "query": [row[0] for row in ocr_rows],
            "text_description": [row[1] for row in ocr_rows],
            "image": [b"more unread image bytes"] * len(ocr_rows),
        },
    )

    def resolve(repository: str, revision: str, path: str) -> str:
        key = (repository, revision, path)
        assert key in destinations, f"unexpected source request: {key!r}"
        return str(destinations[key])

    return resolve


def test_local_fixture_preserves_multi_positive_query_and_stringifies_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec()
    resolver = _write_fixture(tmp_path, spec)
    selected_columns = []
    original_read_table = parquet.read_table

    def tracking_read_table(
        source: object,
        *,
        columns: Iterable[str],
        use_threads: bool,
    ) -> pa.Table:
        selected_columns.append(tuple(columns))
        return original_read_table(
            source,
            columns=list(columns),
            use_threads=use_threads,
        )

    monkeypatch.setattr(parquet, "read_table", tracking_read_table)

    dataset = load_vidore_text_dataset(spec, url_resolver=resolver)

    assert dataset.name == "fixture"
    assert dataset.documents == (
        TextDocument("10", "page ten"),
        TextDocument("11", "page eleven"),
        TextDocument("12", "page twelve"),
    )
    assert dataset.queries == (
        TextQuery(
            "100",
            "shared question",
            frozenset({"10", "11"}),
        ),
        TextQuery("101", "other question", frozenset({"12"})),
    )
    assert dataset.relevant_document_ids == {
        "100": frozenset({"10", "11"}),
        "101": frozenset({"12"}),
    }
    assert len(dataset.fingerprint) == 64
    assert selected_columns == [
        ("corpus-id",),
        ("query-id", "query"),
        ("query-id", "corpus-id", "score"),
        ("query", "text_description"),
    ]
    assert all("image" not in columns for columns in selected_columns)


def test_content_fingerprint_is_order_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec()
    dataset = load_vidore_text_dataset(
        spec,
        url_resolver=_write_fixture(tmp_path, spec),
    )

    reordered = RetrievalDataset(
        name="a-different-label",
        documents=tuple(reversed(dataset.documents)),
        queries=tuple(reversed(dataset.queries)),
    )
    changed = RetrievalDataset(
        name=dataset.name,
        documents=(
            TextDocument("10", "changed"),
            *dataset.documents[1:],
        ),
        queries=dataset.queries,
    )

    assert reordered.fingerprint == dataset.fingerprint
    assert changed.fingerprint != dataset.fingerprint


def test_loader_rejects_ocr_query_drift(tmp_path: Path) -> None:
    spec = _fixture_spec()
    resolver = _write_fixture(
        tmp_path,
        spec,
        ocr_rows=(
            ("shared question", "page ten"),
            ("drifted question", "page eleven"),
            ("other question", "page twelve"),
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"OCR/BEIR query mismatch at row 1",
    ):
        load_vidore_text_dataset(spec, url_resolver=resolver)


def test_loader_rejects_exact_count_drift(tmp_path: Path) -> None:
    source_spec = _fixture_spec()
    resolver = _write_fixture(tmp_path, source_spec)
    expected_three_queries = replace(
        source_spec,
        expected_query_count=3,
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"BEIR queries has 2 rows; expected exactly 3",
    ):
        load_vidore_text_dataset(
            expected_three_queries,
            url_resolver=resolver,
        )


@pytest.mark.parametrize(
    "score",
    [0, -1, 2, True, "1", float("nan"), float("inf"), float("-inf")],
)
def test_loader_rejects_invalid_binary_qrels(
    tmp_path: Path, score: object
) -> None:
    spec = _fixture_spec()
    valid_score: object = (
        score if isinstance(score, (bool, str)) else 1
    )
    resolver = _write_fixture(
        tmp_path,
        spec,
        qrel_rows=(
            (100, 10, score),
            (100, 11, valid_score),
            (101, 12, valid_score),
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"qrel score at row 0 must be positive",
    ):
        load_vidore_text_dataset(spec, url_resolver=resolver)


def test_loader_rejects_qrel_for_unknown_corpus_id(tmp_path: Path) -> None:
    spec = _fixture_spec()
    resolver = _write_fixture(
        tmp_path,
        spec,
        qrel_rows=(
            (100, 10, 1),
            (100, 999, 1),
            (101, 12, 1),
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"references unknown corpus document '999'",
    ):
        load_vidore_text_dataset(spec, url_resolver=resolver)


def test_loader_rejects_qrel_corpus_row_order_drift(tmp_path: Path) -> None:
    spec = _fixture_spec(document_count=2, qrel_count=2)
    resolver = _write_fixture(
        tmp_path,
        spec,
        corpus_ids=(10, 11),
        qrel_rows=((100, 10, 1), (101, 10, 1)),
        ocr_rows=(
            ("shared question", "page ten"),
            ("other question", "page eleven"),
        ),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"row-order join contract was violated",
    ):
        load_vidore_text_dataset(spec, url_resolver=resolver)


def test_semantic_fingerprint_rejects_within_query_ocr_swap(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec()
    canonical = load_vidore_text_dataset(
        spec,
        url_resolver=_write_fixture(tmp_path / "canonical", spec),
    )
    pinned_spec = replace(
        spec, expected_content_fingerprint=canonical.fingerprint
    )
    resolver = _write_fixture(
        tmp_path / "swapped",
        pinned_spec,
        ocr_rows=(
            ("shared question", "page ten"),
            ("shared question", "page eleven"),
            ("other question", "page twelve"),
        ),
    )
    first_path = Path(
        resolver(
            pinned_spec.ocr_repository,
            pinned_spec.ocr_revision,
            pinned_spec.ocr_file.path,
        )
    )
    _write_table(
        first_path,
        {
            "query": (
                "shared question",
                "shared question",
                "other question",
            ),
            "text_description": (
                "page eleven",
                "page ten",
                "page twelve",
            ),
            "image": (b"a", b"b", b"c"),
        },
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"semantic dataset fingerprint mismatch",
    ):
        load_vidore_text_dataset(pinned_spec, url_resolver=resolver)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"path": "../escape.parquet"},
        {"path": "not-parquet.txt"},
        {"size_bytes": 0},
        {"size_bytes": True},
        {"sha256": "A" * 64},
        {"sha256": "short"},
    ],
)
def test_parquet_manifest_rejects_invalid_metadata(
    kwargs: Mapping[str, object],
) -> None:
    values = {
        "path": "data/test.parquet",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        ParquetFileSpec(**values)  # type: ignore[arg-type]


def test_dataset_manifest_requires_pinned_revisions_and_aligned_rows() -> None:
    spec = _fixture_spec()

    with pytest.raises(ValueError, match="full 40-character"):
        replace(spec, beir_revision="main")
    with pytest.raises(ValueError, match="must match"):
        replace(spec, expected_ocr_row_count=4)
    with pytest.raises(ValueError, match="document_count"):
        replace(
            spec,
            expected_qrel_count=2,
            expected_ocr_row_count=2,
        )


def test_official_manifest_and_json_config_are_identical() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "vidore_phase6.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parsed_manifest = load_manifest(config_path)

    assert config == PHASE6_MANIFEST.to_dict()
    assert parsed_manifest == PHASE6_MANIFEST
    assert parsed_manifest.fingerprint == PHASE6_MANIFEST.fingerprint
    assert config["datasets"] == {
        "docvqa": DOCVQA_SPEC.to_dict(),
        "infovqa": INFOVQA_SPEC.to_dict(),
    }
    assert tuple(VIDORE_DATASETS) == ("docvqa", "infovqa")
    assert DOCVQA_SPEC.expected_query_count == 451
    assert INFOVQA_SPEC.expected_query_count == 494
    assert DOCVQA_SPEC.ocr_file.sha256 == (
        "48428c3217bf89ef4f0b72a34ea903a99ca71d34312335e9b7777ecdbe3358b6"
    )
    assert INFOVQA_SPEC.ocr_file.sha256 == (
        "a280326796ee34d206943233b6916cfdebe4072b10dd00fb93e3d978b547319f"
    )
    assert DOCVQA_SPEC.split == INFOVQA_SPEC.split == "test"


def test_public_manifest_parser_rejects_schema_drift() -> None:
    manifest = PHASE6_MANIFEST.to_dict()
    manifest["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        manifest_from_dict(manifest)

    malformed = PHASE6_MANIFEST.to_dict()
    malformed["datasets"]["docvqa"]["beir"]["queries"]["sha256"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError, match="sha256"):
        manifest_from_dict(malformed)


def test_manifest_file_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text(
        '{"schema_version": 1, "schema_version": 1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_manifest(manifest_path)


def test_benchmark_url_requires_revision_in_blob_path() -> None:
    manifest = PHASE6_MANIFEST.to_dict()
    revision = manifest["benchmark_definition"]["revision"]  # type: ignore[index]
    manifest["benchmark_definition"]["url"] = (  # type: ignore[index]
        f"https://github.com/embeddings-benchmark/mteb/blob/main/file.py?"
        f"claimed={revision}"
    )

    with pytest.raises(ValueError, match="pinned to its revision"):
        manifest_from_dict(manifest)


def test_huggingface_url_is_revision_pinned() -> None:
    url = huggingface_resolve_url(
        DOCVQA_SPEC.beir_repository,
        DOCVQA_SPEC.beir_revision,
        DOCVQA_SPEC.queries_file.path,
    )

    assert url == (
        "https://huggingface.co/datasets/"
        "mteb/docvqa_test_subsampled_beir/resolve/"
        "e78cd130055fcb8d69bb0ff4115c4712a157f3d3/"
        "queries/test-00000-of-00001.parquet"
    )
    assert "/main/" not in url


def test_dataset_dataclasses_are_frozen_and_validate_references() -> None:
    document = TextDocument("d", "")
    query = TextQuery("q", "question", frozenset({"d"}))
    dataset = RetrievalDataset("fixture", (document,), (query,))

    with pytest.raises(FrozenInstanceError):
        document.text = "mutated"  # type: ignore[misc]
    with pytest.raises(ValueError, match="absent from the corpus"):
        RetrievalDataset(
            "bad",
            (document,),
            (TextQuery("q", "question", frozenset({"missing"})),),
        )
    with pytest.raises(TypeError):
        dataset.relevant_document_ids["q"] = frozenset()  # type: ignore[index]
