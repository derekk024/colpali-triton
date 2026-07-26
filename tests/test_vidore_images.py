from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from colpali_triton.phase9_config import (
    Phase9SelectionSpec,
    Phase9TaskSpec,
)
from colpali_triton.vidore import (
    DatasetValidationError,
    ParquetFileSpec,
    VidoreDatasetSpec,
)
from colpali_triton.vidore_images import (
    _selection_payload,
    image_content_fingerprint,
    load_vidore_image_subset,
    select_subset_ids,
    selection_fingerprint,
    verify_image_dataset_artifacts,
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 6), color=color).save(output, format="PNG")
    return output.getvalue()


def _file_spec(path: Path, logical_path: str) -> ParquetFileSpec:
    content = path.read_bytes()
    return ParquetFileSpec(
        path=logical_path,
        size_bytes=len(content),
        sha256=sha256(content).hexdigest(),
    )


def _fixture(tmp_path: Path) -> tuple:
    image_bytes = {
        "10": _png_bytes((255, 0, 0)),
        "11": _png_bytes((0, 255, 0)),
        "12": _png_bytes((0, 0, 255)),
        "13": _png_bytes((128, 128, 128)),
    }
    corpus_path = tmp_path / "corpus.parquet"
    query_path = tmp_path / "queries.parquet"
    qrel_path = tmp_path / "qrels.parquet"
    pq.write_table(
        pa.table(
            {
                "corpus-id": pa.array([10, 11, 12, 13], type=pa.int64()),
                "image": pa.array(
                    [
                        {"bytes": image_bytes[str(index)], "path": None}
                        for index in (10, 11, 12, 13)
                    ],
                    type=pa.struct(
                        [
                            pa.field("bytes", pa.binary()),
                            pa.field("path", pa.string()),
                        ]
                    ),
                ),
            }
        ),
        corpus_path,
    )
    pq.write_table(
        pa.table(
            {
                "query-id": [1, 2, 3, 4],
                "query": ["one", "two", "three", "four"],
            }
        ),
        query_path,
    )
    pq.write_table(
        pa.table(
            {
                "query-id": [1, 2, 3, 4],
                "corpus-id": [10, 11, 12, 13],
                "score": [1, 1, 1, 1],
            }
        ),
        qrel_path,
    )
    dummy_ocr = ParquetFileSpec(
        path="data/test.parquet",
        size_bytes=1,
        sha256="0" * 64,
    )
    spec = VidoreDatasetSpec(
        name="fixture",
        split="test",
        beir_repository="owner/fixture",
        beir_revision="a" * 40,
        corpus_file=_file_spec(corpus_path, "corpus/test.parquet"),
        queries_file=_file_spec(query_path, "queries/test.parquet"),
        qrels_file=_file_spec(qrel_path, "qrels/test.parquet"),
        ocr_repository="owner/fixture-ocr",
        ocr_revision="b" * 40,
        ocr_file=dummy_ocr,
        expected_document_count=4,
        expected_query_count=4,
        expected_qrel_count=4,
        expected_ocr_row_count=4,
        expected_content_fingerprint="1" * 64,
    )
    selection = Phase9SelectionSpec(
        method="sha256_identifier_order",
        salt="fixture-salt",
        queries_per_task=2,
        documents_per_task=3,
        positive_policy="include_all_selected_query_positives",
        distractor_policy="sha256_identifier_order",
    )
    query_text = {"1": "one", "2": "two", "3": "three", "4": "four"}
    relevance = {
        "1": frozenset({"10"}),
        "2": frozenset({"11"}),
        "3": frozenset({"12"}),
        "4": frozenset({"13"}),
    }
    query_ids, document_ids = select_subset_ids(
        ("10", "11", "12", "13"),
        query_text,
        relevance,
        selection,
    )
    payload = _selection_payload(
        task_name="fixture",
        source_spec_fingerprint=spec.fingerprint,
        salt=selection.salt,
        query_ids=query_ids,
        document_ids=document_ids,
        query_text_by_id=query_text,
        relevant_by_query=relevance,
    )
    contract = Phase9TaskSpec(
        source_spec_fingerprint=spec.fingerprint,
        selection_fingerprint=selection_fingerprint(
            task_name="fixture",
            source_spec_fingerprint=spec.fingerprint,
            salt=selection.salt,
            query_ids=query_ids,
            document_ids=document_ids,
            query_text_by_id=query_text,
            relevant_by_query=relevance,
        ),
        image_content_fingerprint=image_content_fingerprint(
            selection_payload=payload,
            image_digests={
                document_id: sha256(image_bytes[document_id]).hexdigest()
                for document_id in document_ids
            },
        ),
    )
    paths = {
        spec.corpus_file.path: corpus_path,
        spec.queries_file.path: query_path,
        spec.qrels_file.path: qrel_path,
    }

    def download(
        repository: str,
        revision: str,
        filename: str,
        cache_dir: Path,
        local_files_only: bool,
    ) -> Path:
        assert repository == spec.beir_repository
        assert revision == spec.beir_revision
        assert cache_dir.is_absolute()
        assert local_files_only
        return paths[filename]

    return spec, selection, contract, download, query_ids, document_ids


def test_verified_fixture_loads_exact_selected_images(tmp_path: Path) -> None:
    spec, selection, contract, download, query_ids, document_ids = _fixture(
        tmp_path
    )

    dataset = load_vidore_image_subset(
        spec,
        selection,
        contract,
        cache_dir=tmp_path / "cache",
        local_files_only=True,
        download_file=download,
    )

    assert tuple(query.query_id for query in dataset.queries) == query_ids
    assert tuple(document.document_id for document in dataset.documents) == (
        document_ids
    )
    for document in dataset.documents:
        with document.open_image() as image:
            assert image.mode == "RGB"
    assert all(document.source_format == "PNG" for document in dataset.documents)
    assert all(document.source_size == (8, 6) for document in dataset.documents)
    assert len(dataset.artifacts) == 3
    assert dataset.selection_fingerprint == contract.selection_fingerprint
    assert (
        dataset.image_content_fingerprint
        == contract.image_content_fingerprint
    )


def test_selection_is_independent_of_input_order() -> None:
    query_text = {"a": "A", "b": "B", "c": "C"}
    relevance = {
        "a": frozenset({"1"}),
        "b": frozenset({"2"}),
        "c": frozenset({"3"}),
    }
    selection = Phase9SelectionSpec(
        method="sha256_identifier_order",
        salt="stable",
        queries_per_task=2,
        documents_per_task=3,
        positive_policy="include_all_selected_query_positives",
        distractor_policy="sha256_identifier_order",
    )

    first = select_subset_ids(
        ("1", "2", "3", "4"), query_text, relevance, selection
    )
    second = select_subset_ids(
        ("4", "3", "2", "1"),
        dict(reversed(tuple(query_text.items()))),
        dict(reversed(tuple(relevance.items()))),
        selection,
    )

    assert first == second


def test_selection_rejects_positive_overflow() -> None:
    selection = Phase9SelectionSpec(
        method="sha256_identifier_order",
        salt="stable",
        queries_per_task=2,
        documents_per_task=2,
        positive_policy="include_all_selected_query_positives",
        distractor_policy="sha256_identifier_order",
    )

    with pytest.raises(ValueError, match="positives exceed"):
        select_subset_ids(
            ("1", "2", "3", "4"),
            {"a": "A", "b": "B"},
            {
                "a": frozenset({"1", "2"}),
                "b": frozenset({"3", "4"}),
            },
            selection,
        )


def test_loader_rejects_selection_and_image_fingerprint_drift(
    tmp_path: Path,
) -> None:
    spec, selection, contract, download, _, _ = _fixture(tmp_path)

    with pytest.raises(
        DatasetValidationError, match="identifier fingerprint"
    ):
        load_vidore_image_subset(
            spec,
            selection,
            replace(contract, selection_fingerprint="f" * 64),
            cache_dir=tmp_path / "cache",
            local_files_only=True,
            download_file=download,
        )
    with pytest.raises(DatasetValidationError, match="image fingerprint"):
        load_vidore_image_subset(
            spec,
            selection,
            replace(contract, image_content_fingerprint="f" * 64),
            cache_dir=tmp_path / "cache",
            local_files_only=True,
            download_file=download,
        )


def test_artifact_verification_rejects_hash_drift(tmp_path: Path) -> None:
    spec, _, _, download, _, _ = _fixture(tmp_path)
    corrupted = replace(
        spec,
        corpus_file=replace(spec.corpus_file, sha256="f" * 64),
    )

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        verify_image_dataset_artifacts(
            corrupted,
            cache_dir=tmp_path / "cache",
            local_files_only=True,
            download_file=download,
        )


def test_source_spec_fingerprint_must_match(tmp_path: Path) -> None:
    spec, selection, contract, download, _, _ = _fixture(tmp_path)

    with pytest.raises(DatasetValidationError, match="spec fingerprint"):
        load_vidore_image_subset(
            spec,
            selection,
            replace(contract, source_spec_fingerprint="f" * 64),
            cache_dir=tmp_path / "cache",
            local_files_only=True,
            download_file=download,
        )
