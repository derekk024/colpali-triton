"""Pinned, text-only ViDoRe dataset loading.

The phase-6 baselines use the official BEIR query/qrel files together with
ViDoRe's page-level Tesseract artifacts.  The Parquet reader requests only
identifier and text columns, so neither the BEIR images nor the image column
duplicated in the OCR artifact is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
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
from urllib.parse import quote, urlsplit


_FULL_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


class DatasetValidationError(ValueError):
    """Raised when downloaded dataset content violates the pinned contract."""


@dataclass(frozen=True)
class ParquetFileSpec:
    """Expected LFS metadata for one revision-pinned Parquet artifact."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Parquet path must be a nonempty string")
        parsed_path = PurePosixPath(self.path)
        if (
            parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or self.path.endswith("/")
            or parsed_path.suffix != ".parquet"
        ):
            raise ValueError(
                "Parquet path must be a relative .parquet path without '..'"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ValueError("Parquet size_bytes must be a positive integer")
        if (
            not isinstance(self.sha256, str)
            or _SHA256_PATTERN.fullmatch(self.sha256) is None
        ):
            raise ValueError("Parquet sha256 must be 64 lowercase hex characters")

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class BenchmarkDefinitionSpec:
    """Pinned upstream task definition used to select revisions and split."""

    url: str
    revision: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, str)
            or _FULL_REVISION_PATTERN.fullmatch(self.revision) is None
        ):
            raise ValueError(
                "benchmark definition revision must be a full commit SHA"
            )
        if not isinstance(self.url, str):
            raise ValueError(
                "benchmark definition URL must be an HTTPS GitHub URL "
                "pinned to its revision"
            )
        parsed = urlsplit(self.url)
        path_parts = parsed.path.split("/")
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or len(path_parts) < 6
            or path_parts[3] != "blob"
            or path_parts[4] != self.revision
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "benchmark definition URL must be an HTTPS GitHub URL "
                "pinned to its revision"
            )

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-ready representation."""

        return {"url": self.url, "revision": self.revision}


@dataclass(frozen=True)
class VidoreDatasetSpec:
    """Validated source manifest for one OCR-backed ViDoRe task."""

    name: str
    split: str
    beir_repository: str
    beir_revision: str
    corpus_file: ParquetFileSpec
    queries_file: ParquetFileSpec
    qrels_file: ParquetFileSpec
    ocr_repository: str
    ocr_revision: str
    ocr_file: ParquetFileSpec
    expected_document_count: int
    expected_query_count: int
    expected_qrel_count: int
    expected_ocr_row_count: int
    expected_content_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.name) is None
        ):
            raise ValueError(
                "Dataset name must be a nonempty lowercase identifier"
            )
        if self.split != "test":
            raise ValueError("ViDoRe phase-6 split must be exactly 'test'")
        for label, repository in (
            ("beir_repository", self.beir_repository),
            ("ocr_repository", self.ocr_repository),
        ):
            if (
                not isinstance(repository, str)
                or _REPOSITORY_PATTERN.fullmatch(repository) is None
            ):
                raise ValueError(
                    f"{label} must have the form 'owner/dataset'"
                )
        for label, revision in (
            ("beir_revision", self.beir_revision),
            ("ocr_revision", self.ocr_revision),
        ):
            if (
                not isinstance(revision, str)
                or _FULL_REVISION_PATTERN.fullmatch(revision) is None
            ):
                raise ValueError(
                    f"{label} must be a full 40-character lowercase commit SHA"
                )
        for label, count in (
            ("expected_document_count", self.expected_document_count),
            ("expected_query_count", self.expected_query_count),
            ("expected_qrel_count", self.expected_qrel_count),
            ("expected_ocr_row_count", self.expected_ocr_row_count),
        ):
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        if self.expected_qrel_count != self.expected_ocr_row_count:
            raise ValueError(
                "expected_qrel_count and expected_ocr_row_count must match "
                "because OCR rows are joined to qrels by row order"
            )
        if self.expected_qrel_count != self.expected_document_count:
            raise ValueError(
                "expected_qrel_count and expected_document_count must match "
                "because qrel and corpus rows are joined by row order"
            )
        if (
            self.expected_content_fingerprint is not None
            and (
                not isinstance(self.expected_content_fingerprint, str)
                or _SHA256_PATTERN.fullmatch(
                    self.expected_content_fingerprint
                )
                is None
            )
        ):
            raise ValueError(
                "expected_content_fingerprint must be 64 lowercase hex "
                "characters or None"
            )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the source manifest."""

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "name": self.name,
            "split": self.split,
            "beir": {
                "repository": self.beir_repository,
                "revision": self.beir_revision,
                "corpus": self.corpus_file.to_dict(),
                "queries": self.queries_file.to_dict(),
                "qrels": self.qrels_file.to_dict(),
            },
            "ocr": {
                "repository": self.ocr_repository,
                "revision": self.ocr_revision,
                "data": self.ocr_file.to_dict(),
            },
            "expected_counts": {
                "documents": self.expected_document_count,
                "queries": self.expected_query_count,
                "qrels": self.expected_qrel_count,
                "ocr_rows": self.expected_ocr_row_count,
            },
            "expected_content_fingerprint": (
                self.expected_content_fingerprint
            ),
        }


@dataclass(frozen=True)
class VidoreManifest:
    """Validated top-level phase-6 manifest."""

    schema_version: int
    benchmark_definition: BenchmarkDefinitionSpec
    join_method: str
    join_guard: str
    relevance: str
    datasets: Mapping[str, VidoreDatasetSpec]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("manifest schema_version must be exactly 1")
        if self.join_method != "ocr_rows_to_qrels_by_row_order":
            raise ValueError("manifest join method is not supported")
        if (
            self.join_guard
            != "ocr_query_must_exactly_equal_beir_query_for_qrel"
        ):
            raise ValueError("manifest join guard is not supported")
        if self.relevance != "binary_positive_scores":
            raise ValueError("manifest relevance contract is not supported")
        if not isinstance(self.datasets, Mapping) or not self.datasets:
            raise ValueError("manifest datasets must be a nonempty mapping")

        prepared: Dict[str, VidoreDatasetSpec] = {}
        for name, spec in self.datasets.items():
            if not isinstance(name, str) or not isinstance(
                spec, VidoreDatasetSpec
            ):
                raise TypeError(
                    "manifest datasets must map names to VidoreDatasetSpec"
                )
            if name != spec.name:
                raise ValueError(
                    f"manifest dataset key {name!r} does not match "
                    f"spec name {spec.name!r}"
                )
            prepared[name] = spec
        object.__setattr__(
            self, "datasets", MappingProxyType(prepared)
        )

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the complete manifest."""

        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-ready representation."""

        return {
            "schema_version": self.schema_version,
            "benchmark_definition": self.benchmark_definition.to_dict(),
            "join_contract": {
                "method": self.join_method,
                "guard": self.join_guard,
                "relevance": self.relevance,
            },
            "datasets": {
                name: spec.to_dict()
                for name, spec in self.datasets.items()
            },
        }


@dataclass(frozen=True)
class TextDocument:
    """One document page represented by OCR text."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a nonempty string")
        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")


@dataclass(frozen=True)
class TextQuery:
    """One retrieval query with binary relevant-document judgments."""

    query_id: str
    text: str
    relevant_document_ids: FrozenSet[str]

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a nonempty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("query text must be a nonempty string")
        prepared_relevance = frozenset(self.relevant_document_ids)
        if not prepared_relevance:
            raise ValueError("relevant_document_ids must not be empty")
        if any(
            not isinstance(document_id, str) or not document_id
            for document_id in prepared_relevance
        ):
            raise ValueError(
                "relevant_document_ids must contain only nonempty strings"
            )
        object.__setattr__(
            self, "relevant_document_ids", prepared_relevance
        )


@dataclass(frozen=True)
class RetrievalDataset:
    """An immutable text-retrieval corpus and its binary qrels."""

    name: str
    documents: Tuple[TextDocument, ...]
    queries: Tuple[TextQuery, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("dataset name must be a nonempty string")

        documents = tuple(self.documents)
        queries = tuple(self.queries)
        if not documents:
            raise ValueError("documents must not be empty")
        if not queries:
            raise ValueError("queries must not be empty")
        if any(not isinstance(item, TextDocument) for item in documents):
            raise TypeError("documents must contain only TextDocument values")
        if any(not isinstance(item, TextQuery) for item in queries):
            raise TypeError("queries must contain only TextQuery values")

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
                    f"query {query.query_id!r} references documents absent "
                    f"from the corpus: {sorted(missing)!r}"
                )

        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(
            self,
            "fingerprint",
            retrieval_dataset_fingerprint(documents, queries),
        )

    @property
    def relevant_document_ids(self) -> Mapping[str, FrozenSet[str]]:
        """Return binary qrels keyed by query ID."""

        return MappingProxyType(
            {
                query.query_id: query.relevant_document_ids
                for query in self.queries
            }
        )


def retrieval_dataset_fingerprint(
    documents: Sequence[TextDocument],
    queries: Sequence[TextQuery],
) -> str:
    """Hash semantic dataset content independent of input row ordering."""

    canonical = {
        "format": "colpali-triton-text-retrieval-v1",
        "documents": [
            {"document_id": document.document_id, "text": document.text}
            for document in sorted(
                documents, key=lambda item: item.document_id
            )
        ],
        "queries": [
            {
                "query_id": query.query_id,
                "text": query.text,
                "relevant_document_ids": sorted(
                    query.relevant_document_ids
                ),
            }
            for query in sorted(queries, key=lambda item: item.query_id)
        ],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


DOCVQA_SPEC = VidoreDatasetSpec(
    name="docvqa",
    split="test",
    beir_repository="mteb/docvqa_test_subsampled_beir",
    beir_revision="e78cd130055fcb8d69bb0ff4115c4712a157f3d3",
    corpus_file=ParquetFileSpec(
        path="corpus/test-00000-of-00001.parquet",
        size_bytes=292286618,
        sha256="1d2f0d488fb0ed218b7ab3f63c2b3dedb0a01a90ec3ccf30db8cd4a909c79a7f",
    ),
    queries_file=ParquetFileSpec(
        path="queries/test-00000-of-00001.parquet",
        size_bytes=15679,
        sha256="bcc4a349613a2a89bdc1a7635c81d8d36cabda58b5e6df7e97b17c95fddb5cc0",
    ),
    qrels_file=ParquetFileSpec(
        path="qrels/test-00000-of-00001.parquet",
        size_bytes=6533,
        sha256="f5297c53733a3d2de676b044953d513b0a33972963c42e22f80ee4a50c6d8c7f",
    ),
    ocr_repository="vidore/docvqa_test_subsampled_tesseract",
    ocr_revision="9a183a7237d678eaaf68dc0f640a42f46016802f",
    ocr_file=ParquetFileSpec(
        path="data/test-00000-of-00001.parquet",
        size_bytes=292710972,
        sha256="48428c3217bf89ef4f0b72a34ea903a99ca71d34312335e9b7777ecdbe3358b6",
    ),
    expected_document_count=500,
    expected_query_count=451,
    expected_qrel_count=500,
    expected_ocr_row_count=500,
    expected_content_fingerprint=(
        "3f0df908eb592425c4bc2ab68883736f6ca2f34ac9f67383a09468a600b7e580"
    ),
)


INFOVQA_SPEC = VidoreDatasetSpec(
    name="infovqa",
    split="test",
    beir_repository="mteb/infovqa_test_subsampled_beir",
    beir_revision="74d396242cc281c021eeb38d0ee5f6f30afc1fad",
    corpus_file=ParquetFileSpec(
        path="corpus/test-00000-of-00001.parquet",
        size_bytes=176206329,
        sha256="95981ef151bb4c85e49234e7a19322242ad41a03bc41aead5f5d5240aa0545fd",
    ),
    queries_file=ParquetFileSpec(
        path="queries/test-00000-of-00001.parquet",
        size_bytes=24775,
        sha256="f5e3a1a4992cda32b710b67099e6ea3997bdb1e8089180a418a753f212f5b1f0",
    ),
    qrels_file=ParquetFileSpec(
        path="qrels/test-00000-of-00001.parquet",
        size_bytes=6705,
        sha256="b4093f0963426b1206ea1f23bc2b4d6f6fff76531b55eb5d577250244d7ef325",
    ),
    ocr_repository="vidore/infovqa_test_subsampled_tesseract",
    ocr_revision="7bd56b41a40e767bba7b1467c5520c337246291c",
    ocr_file=ParquetFileSpec(
        path="data/test-00000-of-00001.parquet",
        size_bytes=176824433,
        sha256="a280326796ee34d206943233b6916cfdebe4072b10dd00fb93e3d978b547319f",
    ),
    expected_document_count=500,
    expected_query_count=494,
    expected_qrel_count=500,
    expected_ocr_row_count=500,
    expected_content_fingerprint=(
        "fc39812e6d6350c6fc8894f0f37978739684a240da4fda102fd48551b7ac434d"
    ),
)


VIDORE_DATASETS: Mapping[str, VidoreDatasetSpec] = MappingProxyType(
    {
        DOCVQA_SPEC.name: DOCVQA_SPEC,
        INFOVQA_SPEC.name: INFOVQA_SPEC,
    }
)


MTEB_BENCHMARK_DEFINITION = BenchmarkDefinitionSpec(
    url=(
        "https://github.com/embeddings-benchmark/mteb/blob/"
        "c67d000b43c841df834a53f92ccf6f2e3be77cfb/"
        "mteb/tasks/retrieval/eng/vidore_bench_retrieval.py"
    ),
    revision="c67d000b43c841df834a53f92ccf6f2e3be77cfb",
)


PHASE6_MANIFEST = VidoreManifest(
    schema_version=1,
    benchmark_definition=MTEB_BENCHMARK_DEFINITION,
    join_method="ocr_rows_to_qrels_by_row_order",
    join_guard="ocr_query_must_exactly_equal_beir_query_for_qrel",
    relevance="binary_positive_scores",
    datasets=VIDORE_DATASETS,
)


UrlResolver = Callable[[str, str, str], str]


def huggingface_resolve_url(
    repository: str, revision: str, path: str
) -> str:
    """Build an immutable Hugging Face dataset-file URL."""

    if _REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError("repository must have the form 'owner/dataset'")
    if _FULL_REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("revision must be a full lowercase commit SHA")
    parsed_path = PurePosixPath(path)
    if parsed_path.is_absolute() or ".." in parsed_path.parts:
        raise ValueError("path must be relative and may not contain '..'")
    return (
        "https://huggingface.co/datasets/"
        f"{quote(repository, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(path, safe='/')}"
    )


def get_vidore_dataset_spec(name: str) -> VidoreDatasetSpec:
    """Look up one phase-6 task by its stable short name."""

    try:
        return VIDORE_DATASETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(VIDORE_DATASETS))
        raise KeyError(
            f"unknown ViDoRe dataset {name!r}; expected one of: {choices}"
        ) from error


def _manifest_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _manifest_keys(
    value: Mapping[str, object],
    expected: FrozenSet[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys do not match the schema; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _manifest_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _manifest_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _parse_parquet_file(
    value: object, label: str
) -> ParquetFileSpec:
    file_data = _manifest_mapping(value, label)
    _manifest_keys(
        file_data,
        frozenset({"path", "size_bytes", "sha256"}),
        label,
    )
    return ParquetFileSpec(
        path=_manifest_string(file_data["path"], f"{label}.path"),
        size_bytes=_manifest_integer(
            file_data["size_bytes"], f"{label}.size_bytes"
        ),
        sha256=_manifest_string(
            file_data["sha256"], f"{label}.sha256"
        ),
    )


def manifest_from_dict(value: object) -> VidoreManifest:
    """Parse and fully validate an in-memory phase-6 manifest object."""

    root = _manifest_mapping(value, "manifest")
    _manifest_keys(
        root,
        frozenset(
            {
                "schema_version",
                "benchmark_definition",
                "join_contract",
                "datasets",
            }
        ),
        "manifest",
    )

    benchmark = _manifest_mapping(
        root["benchmark_definition"], "benchmark_definition"
    )
    _manifest_keys(
        benchmark,
        frozenset({"url", "revision"}),
        "benchmark_definition",
    )
    benchmark_spec = BenchmarkDefinitionSpec(
        url=_manifest_string(
            benchmark["url"], "benchmark_definition.url"
        ),
        revision=_manifest_string(
            benchmark["revision"], "benchmark_definition.revision"
        ),
    )

    join = _manifest_mapping(root["join_contract"], "join_contract")
    _manifest_keys(
        join,
        frozenset({"method", "guard", "relevance"}),
        "join_contract",
    )

    raw_datasets = _manifest_mapping(root["datasets"], "datasets")
    datasets: Dict[str, VidoreDatasetSpec] = {}
    for dataset_key, raw_dataset in raw_datasets.items():
        label = f"datasets.{dataset_key}"
        dataset_data = _manifest_mapping(raw_dataset, label)
        _manifest_keys(
            dataset_data,
            frozenset(
                {
                    "name",
                    "split",
                    "beir",
                    "ocr",
                    "expected_counts",
                    "expected_content_fingerprint",
                }
            ),
            label,
        )
        beir = _manifest_mapping(dataset_data["beir"], f"{label}.beir")
        _manifest_keys(
            beir,
            frozenset(
                {
                    "repository",
                    "revision",
                    "corpus",
                    "queries",
                    "qrels",
                }
            ),
            f"{label}.beir",
        )
        ocr = _manifest_mapping(dataset_data["ocr"], f"{label}.ocr")
        _manifest_keys(
            ocr,
            frozenset({"repository", "revision", "data"}),
            f"{label}.ocr",
        )
        counts = _manifest_mapping(
            dataset_data["expected_counts"],
            f"{label}.expected_counts",
        )
        _manifest_keys(
            counts,
            frozenset({"documents", "queries", "qrels", "ocr_rows"}),
            f"{label}.expected_counts",
        )

        spec = VidoreDatasetSpec(
            name=_manifest_string(dataset_data["name"], f"{label}.name"),
            split=_manifest_string(
                dataset_data["split"], f"{label}.split"
            ),
            beir_repository=_manifest_string(
                beir["repository"], f"{label}.beir.repository"
            ),
            beir_revision=_manifest_string(
                beir["revision"], f"{label}.beir.revision"
            ),
            corpus_file=_parse_parquet_file(
                beir["corpus"], f"{label}.beir.corpus"
            ),
            queries_file=_parse_parquet_file(
                beir["queries"], f"{label}.beir.queries"
            ),
            qrels_file=_parse_parquet_file(
                beir["qrels"], f"{label}.beir.qrels"
            ),
            ocr_repository=_manifest_string(
                ocr["repository"], f"{label}.ocr.repository"
            ),
            ocr_revision=_manifest_string(
                ocr["revision"], f"{label}.ocr.revision"
            ),
            ocr_file=_parse_parquet_file(
                ocr["data"], f"{label}.ocr.data"
            ),
            expected_document_count=_manifest_integer(
                counts["documents"],
                f"{label}.expected_counts.documents",
            ),
            expected_query_count=_manifest_integer(
                counts["queries"], f"{label}.expected_counts.queries"
            ),
            expected_qrel_count=_manifest_integer(
                counts["qrels"], f"{label}.expected_counts.qrels"
            ),
            expected_ocr_row_count=_manifest_integer(
                counts["ocr_rows"],
                f"{label}.expected_counts.ocr_rows",
            ),
            expected_content_fingerprint=_manifest_string(
                dataset_data["expected_content_fingerprint"],
                f"{label}.expected_content_fingerprint",
            ),
        )
        if dataset_key != spec.name:
            raise ValueError(
                f"dataset key {dataset_key!r} does not match "
                f"name {spec.name!r}"
            )
        datasets[dataset_key] = spec

    return VidoreManifest(
        schema_version=_manifest_integer(
            root["schema_version"], "schema_version"
        ),
        benchmark_definition=benchmark_spec,
        join_method=_manifest_string(
            join["method"], "join_contract.method"
        ),
        join_guard=_manifest_string(
            join["guard"], "join_contract.guard"
        ),
        relevance=_manifest_string(
            join["relevance"], "join_contract.relevance"
        ),
        datasets=datasets,
    )


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_manifest(path: Union[str, Path]) -> VidoreManifest:
    """Read and validate a phase-6 JSON manifest from disk."""

    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError) as error:
        raise ValueError(f"could not read manifest {path!r}: {error}") from error
    try:
        parsed = json.loads(
            raw_text, object_pairs_hook=_reject_duplicate_json_keys
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"manifest {str(path)!r} is not valid JSON: {error}"
        ) from error
    return manifest_from_dict(parsed)


def load_vidore_manifest(path: Union[str, Path]) -> VidoreManifest:
    """Descriptive alias for :func:`load_manifest`."""

    return load_manifest(path)


def _read_parquet_columns(
    url: str, columns: Sequence[str]
) -> Mapping[str, List[object]]:
    """Read selected Parquet columns through fsspec.

    Imports are deliberately local so the package's core MaxSim path does not
    require evaluation-only dependencies.
    """

    try:
        import fsspec
    except ImportError as error:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "ViDoRe loading requires the 'evaluation' extra (fsspec)"
        ) from error
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "ViDoRe loading requires the 'evaluation' extra (pyarrow)"
        ) from error

    try:
        with fsspec.open(url, mode="rb") as stream:
            table = parquet.read_table(
                stream,
                columns=list(columns),
                use_threads=False,
            )
    except Exception as error:
        raise DatasetValidationError(
            f"failed to read columns {tuple(columns)!r} from {url!r}: {error}"
        ) from error

    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise DatasetValidationError(
            f"{url!r} is missing required columns: {missing!r}"
        )
    return {
        column: table.column(column).to_pylist()
        for column in columns
    }


def _validate_row_count(
    rows: Mapping[str, Sequence[object]],
    expected: int,
    label: str,
) -> None:
    lengths = {len(values) for values in rows.values()}
    if len(lengths) != 1:
        raise DatasetValidationError(
            f"{label} columns have inconsistent row counts: {sorted(lengths)!r}"
        )
    actual = next(iter(lengths), 0)
    if actual != expected:
        raise DatasetValidationError(
            f"{label} has {actual} rows; expected exactly {expected}"
        )


def _stringify_id(value: object, label: str, row_index: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise DatasetValidationError(
            f"{label} at row {row_index} must be a string or integer"
        )
    result = str(value)
    if not result:
        raise DatasetValidationError(
            f"{label} at row {row_index} must not be empty"
        )
    return result


def _required_string(
    value: object,
    label: str,
    row_index: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(
            f"{label} at row {row_index} must be a string"
        )
    if not allow_empty and not value:
        raise DatasetValidationError(
            f"{label} at row {row_index} must not be empty"
        )
    return value


def load_vidore_text_dataset(
    dataset: Union[str, VidoreDatasetSpec],
    *,
    url_resolver: Optional[UrlResolver] = None,
) -> RetrievalDataset:
    """Load a pinned ViDoRe task without materializing image columns.

    OCR rows do not carry BEIR IDs.  The official artifact preserves the
    source row order, so each OCR row is paired with the qrel at the same
    offset.  Before accepting that pair, this loader checks the OCR query
    against the canonical query text addressed by that qrel.
    """

    spec = (
        get_vidore_dataset_spec(dataset)
        if isinstance(dataset, str)
        else dataset
    )
    if not isinstance(spec, VidoreDatasetSpec):
        raise TypeError("dataset must be a name or VidoreDatasetSpec")
    resolve = (
        huggingface_resolve_url
        if url_resolver is None
        else url_resolver
    )

    def source_url(
        repository: str, revision: str, file_spec: ParquetFileSpec
    ) -> str:
        url = resolve(repository, revision, file_spec.path)
        if not isinstance(url, str) or not url:
            raise TypeError("url_resolver must return a nonempty string")
        return url

    corpus_rows = _read_parquet_columns(
        source_url(
            spec.beir_repository,
            spec.beir_revision,
            spec.corpus_file,
        ),
        ("corpus-id",),
    )
    query_rows = _read_parquet_columns(
        source_url(
            spec.beir_repository,
            spec.beir_revision,
            spec.queries_file,
        ),
        ("query-id", "query"),
    )
    qrel_rows = _read_parquet_columns(
        source_url(
            spec.beir_repository,
            spec.beir_revision,
            spec.qrels_file,
        ),
        ("query-id", "corpus-id", "score"),
    )
    ocr_rows = _read_parquet_columns(
        source_url(
            spec.ocr_repository,
            spec.ocr_revision,
            spec.ocr_file,
        ),
        ("query", "text_description"),
    )

    _validate_row_count(
        corpus_rows, spec.expected_document_count, "BEIR corpus"
    )
    _validate_row_count(
        query_rows, spec.expected_query_count, "BEIR queries"
    )
    _validate_row_count(qrel_rows, spec.expected_qrel_count, "BEIR qrels")
    _validate_row_count(
        ocr_rows, spec.expected_ocr_row_count, "OCR artifact"
    )

    corpus_ids = tuple(
        _stringify_id(value, "corpus-id", index)
        for index, value in enumerate(corpus_rows["corpus-id"])
    )
    if len(set(corpus_ids)) != len(corpus_ids):
        raise DatasetValidationError(
            "BEIR corpus IDs are not unique after stringification"
        )
    corpus_id_set = frozenset(corpus_ids)

    query_order: List[str] = []
    query_text_by_id: Dict[str, str] = {}
    for index, (raw_query_id, raw_query_text) in enumerate(
        zip(query_rows["query-id"], query_rows["query"])
    ):
        query_id = _stringify_id(raw_query_id, "query-id", index)
        query_text = _required_string(
            raw_query_text, "query", index
        )
        if query_id in query_text_by_id:
            raise DatasetValidationError(
                f"duplicate BEIR query-id after stringification: {query_id!r}"
            )
        query_order.append(query_id)
        query_text_by_id[query_id] = query_text

    prepared_qrels: List[Tuple[str, str]] = []
    relevant_by_query: Dict[str, set] = {
        query_id: set() for query_id in query_order
    }
    for index, (raw_query_id, raw_corpus_id, raw_score) in enumerate(
        zip(
            qrel_rows["query-id"],
            qrel_rows["corpus-id"],
            qrel_rows["score"],
        )
    ):
        query_id = _stringify_id(raw_query_id, "qrel query-id", index)
        corpus_id = _stringify_id(raw_corpus_id, "qrel corpus-id", index)
        if query_id not in query_text_by_id:
            raise DatasetValidationError(
                f"qrel row {index} references unknown query {query_id!r}"
            )
        if corpus_id not in corpus_id_set:
            raise DatasetValidationError(
                f"qrel row {index} references unknown corpus document "
                f"{corpus_id!r}"
            )
        if corpus_id != corpus_ids[index]:
            raise DatasetValidationError(
                f"qrel corpus-id at row {index} is {corpus_id!r}, "
                f"but corpus row {index} is {corpus_ids[index]!r}; "
                "the pinned row-order join contract was violated"
            )
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, Real)
            or not math.isfinite(float(raw_score))
            or float(raw_score) != 1.0
        ):
            raise DatasetValidationError(
                f"qrel score at row {index} must be positive, finite, "
                "and equal to 1 for binary relevance"
            )
        prepared_qrels.append((query_id, corpus_id))
        relevant_by_query[query_id].add(corpus_id)

    missing_query_qrels = [
        query_id
        for query_id, relevant in relevant_by_query.items()
        if not relevant
    ]
    if missing_query_qrels:
        raise DatasetValidationError(
            "BEIR queries without positive qrels: "
            f"{missing_query_qrels!r}"
        )

    ocr_text_by_document: Dict[str, str] = {}
    for index, (raw_ocr_query, raw_ocr_text, qrel) in enumerate(
        zip(
            ocr_rows["query"],
            ocr_rows["text_description"],
            prepared_qrels,
        )
    ):
        query_id, corpus_id = qrel
        ocr_query = _required_string(raw_ocr_query, "OCR query", index)
        expected_query = query_text_by_id[query_id]
        if ocr_query != expected_query:
            raise DatasetValidationError(
                f"OCR/BEIR query mismatch at row {index}: qrel query "
                f"{query_id!r} expects {expected_query!r}, got "
                f"{ocr_query!r}"
            )
        ocr_text = _required_string(
            raw_ocr_text,
            "OCR text_description",
            index,
            allow_empty=True,
        )
        if (
            corpus_id in ocr_text_by_document
            and ocr_text_by_document[corpus_id] != ocr_text
        ):
            raise DatasetValidationError(
                f"conflicting OCR text mapped to corpus document "
                f"{corpus_id!r}"
            )
        ocr_text_by_document[corpus_id] = ocr_text

    missing_ocr = corpus_id_set - ocr_text_by_document.keys()
    extra_ocr = ocr_text_by_document.keys() - corpus_id_set
    if missing_ocr or extra_ocr:
        raise DatasetValidationError(
            "OCR-to-corpus mapping must cover every corpus ID exactly; "
            f"missing={sorted(missing_ocr)!r}, extra={sorted(extra_ocr)!r}"
        )

    documents = tuple(
        TextDocument(
            document_id=corpus_id,
            text=ocr_text_by_document[corpus_id],
        )
        for corpus_id in corpus_ids
    )
    queries = tuple(
        TextQuery(
            query_id=query_id,
            text=query_text_by_id[query_id],
            relevant_document_ids=frozenset(
                relevant_by_query[query_id]
            ),
        )
        for query_id in query_order
    )
    result = RetrievalDataset(
        name=spec.name,
        documents=documents,
        queries=queries,
    )
    if (
        spec.expected_content_fingerprint is not None
        and result.fingerprint != spec.expected_content_fingerprint
    ):
        raise DatasetValidationError(
            "semantic dataset fingerprint mismatch: "
            f"got {result.fingerprint}, expected "
            f"{spec.expected_content_fingerprint}"
        )
    return result


__all__ = [
    "DOCVQA_SPEC",
    "INFOVQA_SPEC",
    "MTEB_BENCHMARK_DEFINITION",
    "PHASE6_MANIFEST",
    "VIDORE_DATASETS",
    "BenchmarkDefinitionSpec",
    "DatasetValidationError",
    "ParquetFileSpec",
    "RetrievalDataset",
    "TextDocument",
    "TextQuery",
    "VidoreDatasetSpec",
    "VidoreManifest",
    "get_vidore_dataset_spec",
    "huggingface_resolve_url",
    "load_manifest",
    "load_vidore_manifest",
    "load_vidore_text_dataset",
    "manifest_from_dict",
    "retrieval_dataset_fingerprint",
]
