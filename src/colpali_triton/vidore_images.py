"""Verified image loading for the fixed Phase 9 ViDoRe subset."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from PIL import Image

from colpali_triton.phase9_config import (
    Phase9SelectionSpec,
    Phase9TaskSpec,
)
from colpali_triton.vidore import (
    DatasetValidationError,
    ParquetFileSpec,
    TextQuery,
    VidoreDatasetSpec,
)


DatasetDownloadFile = Callable[
    [str, str, str, Path, bool],
    Union[str, Path],
]


@dataclass(frozen=True)
class VerifiedDatasetArtifact:
    """Observed local file metadata for one pinned dataset artifact."""

    repository: str
    revision: str
    path: str
    local_path: Path
    size_bytes: int
    sha256: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "path": self.path,
            "local_path": str(self.local_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "verified": True,
        }


@dataclass(frozen=True)
class ImageDocument:
    """One selected page with its source-image identity."""

    document_id: str
    encoded_image_bytes: bytes
    encoded_image_sha256: str
    source_format: str
    source_size: Tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a nonempty string")
        if (
            not isinstance(self.encoded_image_bytes, bytes)
            or not self.encoded_image_bytes
        ):
            raise ValueError("encoded_image_bytes must be nonempty bytes")
        if (
            not isinstance(self.encoded_image_sha256, str)
            or len(self.encoded_image_sha256) != 64
        ):
            raise ValueError("encoded_image_sha256 must be a SHA-256 digest")
        if (
            sha256(self.encoded_image_bytes).hexdigest()
            != self.encoded_image_sha256
        ):
            raise ValueError("encoded image bytes do not match their digest")
        if not isinstance(self.source_format, str) or not self.source_format:
            raise ValueError("source_format must be a nonempty string")
        if (
            not isinstance(self.source_size, tuple)
            or len(self.source_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.source_size
            )
        ):
            raise ValueError("source_size must contain two positive integers")

    def open_image(self) -> Image.Image:
        """Decode a fresh source image for one indexing call."""

        try:
            return Image.open(BytesIO(self.encoded_image_bytes))
        except Exception as error:
            raise DatasetValidationError(
                f"document {self.document_id!r} image could not be decoded"
            ) from error

    def metadata(self) -> Dict[str, object]:
        return {
            "document_id": self.document_id,
            "encoded_image_sha256": self.encoded_image_sha256,
            "encoded_size_bytes": len(self.encoded_image_bytes),
            "source_format": self.source_format,
            "source_size": list(self.source_size),
        }


@dataclass(frozen=True)
class MultimodalRetrievalDataset:
    """Immutable selected image corpus and binary query relevance."""

    name: str
    documents: Tuple[ImageDocument, ...]
    queries: Tuple[TextQuery, ...]
    artifacts: Tuple[VerifiedDatasetArtifact, ...]
    selection_fingerprint: str
    image_content_fingerprint: str

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        queries = tuple(self.queries)
        artifacts = tuple(self.artifacts)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        if not documents or any(
            not isinstance(item, ImageDocument) for item in documents
        ):
            raise TypeError("documents must contain ImageDocument values")
        if not queries or any(
            not isinstance(item, TextQuery) for item in queries
        ):
            raise TypeError("queries must contain TextQuery values")
        if not artifacts or any(
            not isinstance(item, VerifiedDatasetArtifact)
            for item in artifacts
        ):
            raise TypeError(
                "artifacts must contain VerifiedDatasetArtifact values"
            )
        document_ids = tuple(item.document_id for item in documents)
        query_ids = tuple(item.query_id for item in queries)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document IDs must be unique")
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query IDs must be unique")
        corpus = frozenset(document_ids)
        for query in queries:
            missing = query.relevant_document_ids - corpus
            if missing:
                raise ValueError(
                    f"query {query.query_id!r} has positives outside "
                    f"the selected corpus: {sorted(missing)!r}"
                )
        for label, value in (
            ("selection_fingerprint", self.selection_fingerprint),
            ("image_content_fingerprint", self.image_content_fingerprint),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{label} must be a SHA-256 digest")
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "artifacts", artifacts)

    @property
    def relevant_document_ids(self) -> Mapping[str, FrozenSet[str]]:
        return {
            query.query_id: query.relevant_document_ids
            for query in self.queries
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _default_download_file(
    repository: str,
    revision: str,
    filename: str,
    cache_dir: Path,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "huggingface-hub is required for image dataset loading"
        ) from error
    return Path(
        hf_hub_download(
            repo_id=repository,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
        )
    )


def _verify_artifact(
    spec: VidoreDatasetSpec,
    artifact: ParquetFileSpec,
    *,
    cache_dir: Path,
    local_files_only: bool,
    download_file: DatasetDownloadFile,
) -> VerifiedDatasetArtifact:
    path = Path(
        download_file(
            spec.beir_repository,
            spec.beir_revision,
            artifact.path,
            cache_dir,
            local_files_only,
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset artifact is not a file: {path}")
    size = path.stat().st_size
    if size != artifact.size_bytes:
        raise DatasetValidationError(
            f"dataset artifact size mismatch for {artifact.path!r}: "
            f"got {size}, expected {artifact.size_bytes}"
        )
    digest = _sha256_file(path)
    if digest != artifact.sha256:
        raise DatasetValidationError(
            f"dataset artifact SHA-256 mismatch for {artifact.path!r}: "
            f"got {digest}, expected {artifact.sha256}"
        )
    return VerifiedDatasetArtifact(
        repository=spec.beir_repository,
        revision=spec.beir_revision,
        path=artifact.path,
        local_path=path,
        size_bytes=size,
        sha256=digest,
    )


def verify_image_dataset_artifacts(
    spec: VidoreDatasetSpec,
    *,
    cache_dir: Union[str, Path],
    local_files_only: bool = False,
    download_file: Optional[DatasetDownloadFile] = None,
) -> Tuple[VerifiedDatasetArtifact, ...]:
    """Resolve and hash the BEIR corpus, query, and qrel files."""

    if not isinstance(spec, VidoreDatasetSpec):
        raise TypeError("spec must be a VidoreDatasetSpec")
    if not isinstance(local_files_only, bool):
        raise TypeError("local_files_only must be a bool")
    downloader = download_file or _default_download_file
    if not callable(downloader):
        raise TypeError("download_file must be callable or None")
    prepared_cache = Path(cache_dir).resolve()
    prepared_cache.mkdir(parents=True, exist_ok=True)
    return tuple(
        _verify_artifact(
            spec,
            artifact,
            cache_dir=prepared_cache,
            local_files_only=local_files_only,
            download_file=downloader,
        )
        for artifact in (
            spec.corpus_file,
            spec.queries_file,
            spec.qrels_file,
        )
    )


def _identifier_rank(salt: str, kind: str, identifier: str) -> Tuple[str, str]:
    digest = sha256(
        f"{salt}\0{kind}\0{identifier}".encode("utf-8")
    ).hexdigest()
    return digest, identifier


def select_subset_ids(
    corpus_ids: Sequence[str],
    query_text_by_id: Mapping[str, str],
    relevant_by_query: Mapping[str, FrozenSet[str]],
    selection: Phase9SelectionSpec,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Select queries, all their positives, and hash-ordered distractors."""

    if not isinstance(selection, Phase9SelectionSpec):
        raise TypeError("selection must be a Phase9SelectionSpec")
    prepared_corpus = tuple(corpus_ids)
    if not prepared_corpus or any(
        not isinstance(item, str) or not item for item in prepared_corpus
    ):
        raise ValueError("corpus_ids must contain nonempty strings")
    if len(set(prepared_corpus)) != len(prepared_corpus):
        raise ValueError("corpus_ids must be unique")
    if set(query_text_by_id) != set(relevant_by_query):
        raise ValueError(
            "query_text_by_id and relevant_by_query IDs must match"
        )
    if len(query_text_by_id) < selection.queries_per_task:
        raise ValueError("not enough queries for the configured subset")

    query_ids = tuple(
        sorted(
            query_text_by_id,
            key=lambda item: _identifier_rank(
                selection.salt, "query", item
            ),
        )[: selection.queries_per_task]
    )
    corpus = frozenset(prepared_corpus)
    required_documents: set = set()
    for query_id in query_ids:
        relevant = frozenset(relevant_by_query[query_id])
        if not relevant:
            raise ValueError(f"query {query_id!r} has no positive documents")
        missing = relevant - corpus
        if missing:
            raise ValueError(
                f"query {query_id!r} references unknown documents"
            )
        required_documents.update(relevant)
    if len(required_documents) > selection.documents_per_task:
        raise ValueError(
            "selected query positives exceed the document subset capacity"
        )

    distractors_needed = (
        selection.documents_per_task - len(required_documents)
    )
    distractors = sorted(
        corpus - required_documents,
        key=lambda item: _identifier_rank(
            selection.salt, "document", item
        ),
    )[:distractors_needed]
    selected_documents = required_documents.union(distractors)
    if len(selected_documents) != selection.documents_per_task:
        raise ValueError("not enough documents for the configured subset")
    document_ids = tuple(
        sorted(
            selected_documents,
            key=lambda item: _identifier_rank(
                selection.salt, "document", item
            ),
        )
    )
    return query_ids, document_ids


def _selection_payload(
    *,
    task_name: str,
    source_spec_fingerprint: str,
    salt: str,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    query_text_by_id: Mapping[str, str],
    relevant_by_query: Mapping[str, FrozenSet[str]],
) -> Dict[str, object]:
    return {
        "format": "colpali-triton-phase9-selection-v1",
        "task": task_name,
        "salt": salt,
        "source_spec_fingerprint": source_spec_fingerprint,
        "documents": list(document_ids),
        "queries": [
            {
                "query_id": query_id,
                "text": query_text_by_id[query_id],
                "relevant_document_ids": sorted(
                    relevant_by_query[query_id]
                ),
            }
            for query_id in query_ids
        ],
    }


def _canonical_fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def selection_fingerprint(
    *,
    task_name: str,
    source_spec_fingerprint: str,
    salt: str,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    query_text_by_id: Mapping[str, str],
    relevant_by_query: Mapping[str, FrozenSet[str]],
) -> str:
    """Fingerprint the selected identifiers, query text, and complete qrels."""

    return _canonical_fingerprint(
        _selection_payload(
            task_name=task_name,
            source_spec_fingerprint=source_spec_fingerprint,
            salt=salt,
            query_ids=query_ids,
            document_ids=document_ids,
            query_text_by_id=query_text_by_id,
            relevant_by_query=relevant_by_query,
        )
    )


def image_content_fingerprint(
    *,
    selection_payload: Mapping[str, object],
    image_digests: Mapping[str, str],
) -> str:
    """Extend the selection fingerprint payload with encoded image hashes."""

    payload = dict(selection_payload)
    document_ids = payload.get("documents")
    if not isinstance(document_ids, list):
        raise TypeError("selection payload documents must be a list")
    payload["documents"] = [
        {
            "document_id": document_id,
            "encoded_image_sha256": image_digests[document_id],
        }
        for document_id in document_ids
    ]
    return _canonical_fingerprint(payload)


def _read_table(path: Path, columns: Sequence[str]) -> object:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "image dataset loading requires the 'evaluation' extra"
        ) from error
    try:
        return parquet.ParquetFile(path).read(
            columns=list(columns), use_threads=False
        )
    except Exception as error:
        raise DatasetValidationError(
            f"failed to read {tuple(columns)!r} from {path}: {error}"
        ) from error


def _stringify_id(value: object, label: str, row: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise DatasetValidationError(
            f"{label} at row {row} must be a string or integer"
        )
    result = str(value)
    if not result:
        raise DatasetValidationError(
            f"{label} at row {row} must not be empty"
        )
    return result


def _path_by_artifact(
    artifacts: Sequence[VerifiedDatasetArtifact],
) -> Mapping[str, Path]:
    return {artifact.path: artifact.local_path for artifact in artifacts}


def load_vidore_image_subset(
    source_spec: VidoreDatasetSpec,
    selection: Phase9SelectionSpec,
    task_contract: Phase9TaskSpec,
    *,
    cache_dir: Union[str, Path],
    local_files_only: bool = False,
    download_file: Optional[DatasetDownloadFile] = None,
) -> MultimodalRetrievalDataset:
    """Load, select, decode, and fingerprint one pinned image subset."""

    if not isinstance(source_spec, VidoreDatasetSpec):
        raise TypeError("source_spec must be a VidoreDatasetSpec")
    if not isinstance(selection, Phase9SelectionSpec):
        raise TypeError("selection must be a Phase9SelectionSpec")
    if not isinstance(task_contract, Phase9TaskSpec):
        raise TypeError("task_contract must be a Phase9TaskSpec")
    if source_spec.fingerprint != task_contract.source_spec_fingerprint:
        raise DatasetValidationError("source dataset spec fingerprint drifted")

    artifacts = verify_image_dataset_artifacts(
        source_spec,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        download_file=download_file,
    )
    paths = _path_by_artifact(artifacts)
    corpus_table = _read_table(
        paths[source_spec.corpus_file.path],
        ("corpus-id", "image"),
    )
    query_table = _read_table(
        paths[source_spec.queries_file.path],
        ("query-id", "query"),
    )
    qrel_table = _read_table(
        paths[source_spec.qrels_file.path],
        ("query-id", "corpus-id", "score"),
    )
    if corpus_table.num_rows != source_spec.expected_document_count:
        raise DatasetValidationError("BEIR corpus row count drifted")
    if query_table.num_rows != source_spec.expected_query_count:
        raise DatasetValidationError("BEIR query row count drifted")
    if qrel_table.num_rows != source_spec.expected_qrel_count:
        raise DatasetValidationError("BEIR qrel row count drifted")

    raw_corpus_ids = corpus_table.column("corpus-id").to_pylist()
    corpus_ids = tuple(
        _stringify_id(value, "corpus-id", index)
        for index, value in enumerate(raw_corpus_ids)
    )
    if len(set(corpus_ids)) != len(corpus_ids):
        raise DatasetValidationError("corpus IDs are not unique")
    corpus = frozenset(corpus_ids)

    query_rows = query_table.to_pydict()
    query_text_by_id: Dict[str, str] = {}
    for index, (raw_id, raw_text) in enumerate(
        zip(query_rows["query-id"], query_rows["query"])
    ):
        query_id = _stringify_id(raw_id, "query-id", index)
        if not isinstance(raw_text, str) or not raw_text:
            raise DatasetValidationError(
                f"query at row {index} must be a nonempty string"
            )
        if query_id in query_text_by_id:
            raise DatasetValidationError("query IDs are not unique")
        query_text_by_id[query_id] = raw_text

    qrel_rows = qrel_table.to_pydict()
    relevant: Dict[str, set] = {
        query_id: set() for query_id in query_text_by_id
    }
    for index, (raw_query_id, raw_document_id, raw_score) in enumerate(
        zip(
            qrel_rows["query-id"],
            qrel_rows["corpus-id"],
            qrel_rows["score"],
        )
    ):
        query_id = _stringify_id(raw_query_id, "qrel query-id", index)
        document_id = _stringify_id(
            raw_document_id, "qrel corpus-id", index
        )
        if query_id not in query_text_by_id:
            raise DatasetValidationError("qrel references an unknown query")
        if document_id not in corpus:
            raise DatasetValidationError("qrel references an unknown document")
        if index >= len(corpus_ids) or document_id != corpus_ids[index]:
            raise DatasetValidationError(
                "qrel-to-corpus row-order contract drifted"
            )
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, Real)
            or not math.isfinite(float(raw_score))
            or float(raw_score) != 1.0
        ):
            raise DatasetValidationError(
                "qrel scores must be finite binary positives"
            )
        relevant[query_id].add(document_id)
    missing_qrels = [
        query_id for query_id, values in relevant.items() if not values
    ]
    if missing_qrels:
        raise DatasetValidationError("some BEIR queries have no qrels")
    frozen_relevance = {
        query_id: frozenset(values)
        for query_id, values in relevant.items()
    }

    query_ids, document_ids = select_subset_ids(
        corpus_ids,
        query_text_by_id,
        frozen_relevance,
        selection,
    )
    payload = _selection_payload(
        task_name=source_spec.name,
        source_spec_fingerprint=source_spec.fingerprint,
        salt=selection.salt,
        query_ids=query_ids,
        document_ids=document_ids,
        query_text_by_id=query_text_by_id,
        relevant_by_query=frozen_relevance,
    )
    observed_selection_fingerprint = _canonical_fingerprint(payload)
    if (
        observed_selection_fingerprint
        != task_contract.selection_fingerprint
    ):
        raise DatasetValidationError("selected identifier fingerprint drifted")

    row_by_id = {
        document_id: index
        for index, document_id in enumerate(corpus_ids)
    }
    documents: List[ImageDocument] = []
    image_digests: Dict[str, str] = {}
    image_column = corpus_table.column("image")
    for document_id in document_ids:
        raw_image = image_column[row_by_id[document_id]].as_py()
        if not isinstance(raw_image, Mapping):
            raise DatasetValidationError("image cell must be a struct")
        image_bytes = raw_image.get("bytes")
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise DatasetValidationError(
                "image struct must contain nonempty encoded bytes"
            )
        digest = sha256(image_bytes).hexdigest()
        try:
            with Image.open(BytesIO(image_bytes)) as source_image:
                source_format = source_image.format
                source_size = tuple(source_image.size)
                source_image.verify()
        except Exception as error:
            raise DatasetValidationError(
                f"document {document_id!r} image could not be decoded"
            ) from error
        if not isinstance(source_format, str) or not source_format:
            raise DatasetValidationError("decoded image format is missing")
        image_digests[document_id] = digest
        documents.append(
            ImageDocument(
                document_id=document_id,
                encoded_image_bytes=image_bytes,
                encoded_image_sha256=digest,
                source_format=source_format,
                source_size=source_size,
            )
        )

    observed_image_fingerprint = image_content_fingerprint(
        selection_payload=payload,
        image_digests=image_digests,
    )
    if (
        observed_image_fingerprint
        != task_contract.image_content_fingerprint
    ):
        raise DatasetValidationError("selected image fingerprint drifted")
    queries = tuple(
        TextQuery(
            query_id=query_id,
            text=query_text_by_id[query_id],
            relevant_document_ids=frozen_relevance[query_id],
        )
        for query_id in query_ids
    )
    return MultimodalRetrievalDataset(
        name=source_spec.name,
        documents=tuple(documents),
        queries=queries,
        artifacts=artifacts,
        selection_fingerprint=observed_selection_fingerprint,
        image_content_fingerprint=observed_image_fingerprint,
    )


__all__ = [
    "ImageDocument",
    "MultimodalRetrievalDataset",
    "VerifiedDatasetArtifact",
    "image_content_fingerprint",
    "load_vidore_image_subset",
    "select_subset_ids",
    "selection_fingerprint",
    "verify_image_dataset_artifacts",
]
