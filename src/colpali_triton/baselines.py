"""Deterministic OCR-text retrieval baselines.

The indexes in this module score complete pages while indexing fixed-size text
chunks.  BM25 uses the same term weighting as ``rank-bm25==0.2.2``'s
``BM25Okapi`` implementation; dense retrieval uses maximum query-to-chunk
cosine similarity for each page.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TextDocument(Protocol):
    """The structural document interface consumed by the text indexes."""

    document_id: str
    text: str


@runtime_checkable
class TextEncoder(Protocol):
    """Minimal, fakeable interface for a batched sentence encoder."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        """Return one finite embedding row per input string."""


def tokenize(text: str) -> Tuple[str, ...]:
    """Tokenize text into deterministic Unicode-alphanumeric terms.

    Compatibility normalization happens before caseless comparison.  Any
    character for which :meth:`str.isalnum` is false acts as a separator.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    current = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def chunk_text(
    text: str,
    *,
    max_words: int = 256,
    overlap: int = 32,
) -> Tuple[str, ...]:
    """Split text into fixed, overlapping whitespace-delimited chunks.

    Every document produces at least one chunk.  An empty or whitespace-only
    document is represented by one empty chunk so it keeps its place in the
    corpus and receives a stable zero score.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_chunk_parameters(max_words=max_words, overlap=overlap)

    words = text.split()
    if not words:
        return ("",)

    step = max_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return tuple(chunks)


def _validate_chunk_parameters(*, max_words: int, overlap: int) -> None:
    if isinstance(max_words, bool) or not isinstance(max_words, int):
        raise TypeError("max_words must be an integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap < 0:
        raise ValueError("overlap must be nonnegative")
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")


def _prepare_documents(
    documents: Sequence[TextDocument],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence")
    if not documents:
        raise ValueError("documents must not be empty")

    document_ids = []
    texts = []
    for document in documents:
        try:
            document_id = document.document_id
            text = document.text
        except AttributeError as error:
            raise TypeError(
                "each document must expose document_id and text attributes"
            ) from error
        if not isinstance(document_id, str):
            raise TypeError("document_id must be a string")
        if not document_id:
            raise ValueError("document_id must not be empty")
        if not isinstance(text, str):
            raise TypeError("document text must be a string")
        document_ids.append(document_id)
        texts.append(text)

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document IDs must be unique")
    return tuple(document_ids), tuple(texts)


def _flatten_chunks(
    texts: Sequence[str],
    *,
    max_words: int,
    overlap: int,
) -> Tuple[Tuple[str, ...], NDArray[np.int64]]:
    chunks = []
    document_indices = []
    for document_index, text in enumerate(texts):
        document_chunks = chunk_text(
            text, max_words=max_words, overlap=overlap
        )
        chunks.extend(document_chunks)
        document_indices.extend([document_index] * len(document_chunks))
    return tuple(chunks), np.asarray(document_indices, dtype=np.int64)


class BM25ChunkMaxIndex:
    """BM25Okapi over text chunks with maximum chunk score per document."""

    def __init__(
        self,
        documents: Sequence[TextDocument],
        *,
        max_words: int = 256,
        overlap: int = 32,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        _validate_chunk_parameters(max_words=max_words, overlap=overlap)
        self._document_ids, texts = _prepare_documents(documents)
        self._chunks, self._chunk_document_indices = _flatten_chunks(
            texts, max_words=max_words, overlap=overlap
        )

        self.k1 = _validate_finite_parameter(k1, "k1", minimum=0.0)
        if self.k1 == 0.0:
            raise ValueError("k1 must be positive")
        self.b = _validate_finite_parameter(b, "b", minimum=0.0)
        if self.b > 1.0:
            raise ValueError("b must not exceed 1")
        self.epsilon = _validate_finite_parameter(
            epsilon, "epsilon", minimum=0.0
        )

        tokenized_chunks = tuple(tokenize(chunk) for chunk in self._chunks)
        self._chunk_lengths = np.asarray(
            [len(tokens) for tokens in tokenized_chunks], dtype=np.float64
        )
        self._average_chunk_length = float(self._chunk_lengths.mean())

        term_frequencies = []
        document_frequencies = {}
        for tokens in tokenized_chunks:
            frequencies = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            term_frequencies.append(frequencies)
            for token in frequencies:
                document_frequencies[token] = (
                    document_frequencies.get(token, 0) + 1
                )
        self._term_frequencies = tuple(term_frequencies)
        self._inverse_document_frequencies = self._calculate_idf(
            document_frequencies
        )

    @property
    def document_ids(self) -> Tuple[str, ...]:
        """Document IDs in the same order as score vectors."""

        return self._document_ids

    @property
    def chunks(self) -> Tuple[str, ...]:
        """The flattened chunks, exposed for reproducibility reports."""

        return self._chunks

    @property
    def index_size_bytes(self) -> int:
        """Return a deterministic logical size for the searchable payload.

        The estimate counts each unique UTF-8 term once, one float64 IDF per
        term, two int64 values (chunk row and frequency) per posting, plus the
        float64 chunk lengths and int64 chunk-to-page mapping.  It deliberately
        excludes Python object overhead so runs are comparable across Python
        builds.
        """

        term_bytes = sum(
            len(term.encode("utf-8"))
            for term in self._inverse_document_frequencies
        )
        idf_bytes = len(self._inverse_document_frequencies) * 8
        posting_count = sum(
            len(frequencies) for frequencies in self._term_frequencies
        )
        posting_bytes = posting_count * 2 * 8
        return int(
            term_bytes
            + idf_bytes
            + posting_bytes
            + self._chunk_lengths.nbytes
            + self._chunk_document_indices.nbytes
        )

    @property
    def logical_index_bytes(self) -> int:
        """Alias for :attr:`index_size_bytes`."""

        return self.index_size_bytes

    @property
    def index_nbytes(self) -> int:
        """Alias for :attr:`index_size_bytes`."""

        return self.index_size_bytes

    def _calculate_idf(self, document_frequencies: dict) -> dict:
        # This intentionally follows rank-bm25 0.2.2's BM25Okapi exactly.
        corpus_size = len(self._chunks)
        idf = {}
        negative_terms = []
        idf_sum = 0.0
        for term, frequency in document_frequencies.items():
            value = math.log(corpus_size - frequency + 0.5) - math.log(
                frequency + 0.5
            )
            idf[term] = value
            idf_sum += value
            if value < 0.0:
                negative_terms.append(term)

        average_idf = idf_sum / len(idf) if idf else 0.0
        floor = self.epsilon * average_idf
        for term in negative_terms:
            idf[term] = floor
        return idf

    def score(self, query: str) -> NDArray[np.float64]:
        """Score a query and return one finite value per input document."""

        query_tokens = tokenize(query)
        chunk_scores = np.zeros(len(self._chunks), dtype=np.float64)
        if query_tokens and self._average_chunk_length > 0.0:
            length_normalizer = self.k1 * (
                1.0
                - self.b
                + self.b
                * self._chunk_lengths
                / self._average_chunk_length
            )
            for term in query_tokens:
                term_frequency = np.fromiter(
                    (
                        frequencies.get(term, 0)
                        for frequencies in self._term_frequencies
                    ),
                    dtype=np.float64,
                    count=len(self._chunks),
                )
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + length_normalizer
                contribution = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator != 0.0,
                )
                chunk_scores += self._inverse_document_frequencies.get(
                    term, 0.0
                ) * contribution

        document_scores = np.full(
            len(self._document_ids), -np.inf, dtype=np.float64
        )
        np.maximum.at(
            document_scores, self._chunk_document_indices, chunk_scores
        )
        # Each page always has a chunk; this is a defensive guarantee that the
        # public contract never leaks a non-finite value.
        return np.nan_to_num(
            document_scores,
            copy=False,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=0.0,
        )


def _validate_finite_parameter(
    value: float, name: str, *, minimum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    prepared = float(value)
    if not math.isfinite(prepared):
        raise ValueError(f"{name} must be finite")
    if prepared < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return prepared


class SentenceTransformerTextEncoder:
    """Lazily loaded, revision-pinned Sentence Transformers encoder."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: Optional[int] = None,
        cache_folder: Optional[str] = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty string")
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be a nonempty string")
        if device is not None and not isinstance(device, str):
            raise TypeError("device must be a string or None")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_seq_length is not None:
            if isinstance(max_seq_length, bool) or not isinstance(
                max_seq_length, int
            ):
                raise TypeError("max_seq_length must be an integer or None")
            if max_seq_length <= 0:
                raise ValueError("max_seq_length must be positive")
        if cache_folder is not None:
            if not isinstance(cache_folder, str):
                raise TypeError("cache_folder must be a string or None")
            if not cache_folder:
                raise ValueError("cache_folder must not be empty")

        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_folder = cache_folder
        self._model = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for dense retrieval"
                ) from error
            self._model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                device=self.device,
                cache_folder=self.cache_folder,
            )
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def load(self) -> "SentenceTransformerTextEncoder":
        """Load the pinned model now while retaining lazy construction."""

        model = self._load_model()
        if self._dimension is None:
            dimension = model.get_sentence_embedding_dimension()
            if dimension is None or int(dimension) <= 0:
                raise ValueError("encoder returned an invalid embedding dimension")
            self._dimension = int(dimension)
        return self

    @property
    def embedding_dimension(self) -> int:
        """The output width, loading the model first when necessary."""

        self.load()
        assert self._dimension is not None
        return self._dimension

    @property
    def effective_max_seq_length(self) -> int:
        """The model's effective token limit after any configured override."""

        model = self._load_model()
        value = int(model.max_seq_length)
        if value <= 0:
            raise ValueError("encoder returned an invalid max sequence length")
        return value

    @property
    def effective_model_dtype(self) -> str:
        """Return the dtype of the loaded transformer's parameters."""

        model = self._load_model()
        first_module = model._first_module()
        auto_model = getattr(first_module, "auto_model", None)
        if auto_model is None:
            raise ValueError("encoder does not expose its transformer model")
        try:
            parameter = next(auto_model.parameters())
        except StopIteration as error:
            raise ValueError("encoder transformer has no parameters") from error
        return str(parameter.dtype).removeprefix("torch.")

    @property
    def effective_pooling_mode(self) -> str:
        """Return the configured Sentence Transformers pooling mode."""

        model = self._load_model()
        for module in model._modules.values():
            if module.__class__.__name__ != "Pooling":
                continue
            enabled = [
                name
                for name, attribute in (
                    ("cls", "pooling_mode_cls_token"),
                    ("mean", "pooling_mode_mean_tokens"),
                    ("max", "pooling_mode_max_tokens"),
                    (
                        "mean_sqrt_len_tokens",
                        "pooling_mode_mean_sqrt_len_tokens",
                    ),
                    (
                        "weightedmean",
                        "pooling_mode_weightedmean_tokens",
                    ),
                    ("lasttoken", "pooling_mode_lasttoken"),
                )
                if bool(getattr(module, attribute, False))
            ]
            if not enabled:
                raise ValueError("encoder pooling module has no enabled mode")
            return "+".join(enabled)
        raise ValueError("encoder does not expose a pooling module")

    @property
    def has_normalization_module(self) -> bool:
        """Whether the loaded pipeline ends with explicit L2 normalization."""

        model = self._load_model()
        modules = tuple(model._modules.values())
        return bool(
            modules and modules[-1].__class__.__name__ == "Normalize"
        )

    def token_lengths(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Tokenize without truncation and return one length per text."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        prepared = tuple(texts)
        if any(not isinstance(text, str) for text in prepared):
            raise TypeError("texts must contain only strings")
        if not prepared:
            return np.empty(0, dtype=np.int64)

        model = self._load_model()
        tokenizer = getattr(model._first_module(), "tokenizer", None)
        if tokenizer is None:
            raise ValueError("encoder does not expose its tokenizer")
        encoded = tokenizer(
            list(prepared),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )
        lengths = np.asarray(encoded["length"], dtype=np.int64)
        if lengths.shape != (len(prepared),) or np.any(lengths <= 0):
            raise ValueError("encoder tokenizer returned invalid lengths")
        return lengths

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        prepared = tuple(texts)
        if any(not isinstance(text, str) for text in prepared):
            raise TypeError("texts must contain only strings")

        model = self._load_model()
        if not prepared:
            if self._dimension is None:
                dimension = model.get_sentence_embedding_dimension()
                if dimension is None or int(dimension) <= 0:
                    raise ValueError(
                        "encoder returned an invalid embedding dimension"
                    )
                self._dimension = int(dimension)
            return np.empty((0, self._dimension), dtype=np.float32)

        encoded = model.encode(
            list(prepared),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = _validate_embeddings(
            encoded, expected_rows=len(prepared), name="encoder output"
        )
        self._dimension = embeddings.shape[1]
        return _normalize_rows(embeddings)


def _validate_embeddings(
    embeddings: object,
    *,
    expected_rows: int,
    name: str,
    expected_dimension: Optional[int] = None,
) -> NDArray[np.float32]:
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError(f"{name} must have two dimensions")
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"{name} must have {expected_rows} rows; got {array.shape[0]}"
        )
    if array.shape[1] <= 0:
        raise ValueError(f"{name} must have a positive embedding dimension")
    if expected_dimension is not None and array.shape[1] != expected_dimension:
        raise ValueError(
            f"{name} must have dimension {expected_dimension}; "
            f"got {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    prepared = np.asarray(array, dtype=np.float32)
    if not np.isfinite(prepared).all():
        raise ValueError(f"{name} must contain only finite values")
    return prepared


def _normalize_rows(
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings, dtype=np.float32),
        where=norms > 0.0,
    )


class DenseChunkMaxIndex:
    """Normalized dense chunk index with MaxSim aggregation by page."""

    def __init__(
        self,
        documents: Sequence[TextDocument],
        encoder: TextEncoder,
        *,
        max_words: int = 256,
        overlap: int = 32,
    ) -> None:
        _validate_chunk_parameters(max_words=max_words, overlap=overlap)
        if not isinstance(encoder, TextEncoder):
            raise TypeError("encoder must provide an encode method")
        self._encoder = encoder
        self._document_ids, texts = _prepare_documents(documents)
        self._chunks, self._chunk_document_indices = _flatten_chunks(
            texts, max_words=max_words, overlap=overlap
        )

        nonblank_indices = [
            index for index, chunk in enumerate(self._chunks) if chunk.strip()
        ]
        if nonblank_indices:
            encoded = encoder.encode(
                [self._chunks[index] for index in nonblank_indices]
            )
            nonblank_embeddings = _validate_embeddings(
                encoded,
                expected_rows=len(nonblank_indices),
                name="document encoder output",
            )
            dimension = nonblank_embeddings.shape[1]
        else:
            encoded = encoder.encode(())
            nonblank_embeddings = _validate_embeddings(
                encoded,
                expected_rows=0,
                name="document encoder output",
            )
            dimension = nonblank_embeddings.shape[1]

        embeddings = np.zeros(
            (len(self._chunks), dimension), dtype=np.float32
        )
        if nonblank_indices:
            embeddings[nonblank_indices] = _normalize_rows(
                nonblank_embeddings
            )
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    @property
    def document_ids(self) -> Tuple[str, ...]:
        """Document IDs in the same order as score vectors."""

        return self._document_ids

    @property
    def chunks(self) -> Tuple[str, ...]:
        """Flattened text chunks in embedding row order."""

        return self._chunks

    @property
    def embeddings(self) -> NDArray[np.float32]:
        """Read-only view of the normalized flattened chunk embeddings."""

        view = self._embeddings.view()
        view.flags.writeable = False
        return view

    @property
    def chunk_document_indices(self) -> NDArray[np.int64]:
        """Read-only chunk-row to document-row mapping."""

        view = self._chunk_document_indices.view()
        view.flags.writeable = False
        return view

    @property
    def logical_index_bytes(self) -> int:
        """Bytes occupied by the dense vectors and integer page mapping."""

        return int(
            self._embeddings.nbytes + self._chunk_document_indices.nbytes
        )

    @property
    def index_size_bytes(self) -> int:
        """Bytes occupied by the dense vectors and integer page mapping."""

        return self.logical_index_bytes

    @property
    def index_nbytes(self) -> int:
        """Alias for :attr:`logical_index_bytes`."""

        return self.logical_index_bytes

    def score(self, query: str) -> NDArray[np.float32]:
        """Score a query by maximum normalized dot product per document."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return np.zeros(len(self._document_ids), dtype=np.float32)

        encoded = self._encoder.encode((query,))
        query_embedding = _validate_embeddings(
            encoded,
            expected_rows=1,
            expected_dimension=self._embeddings.shape[1],
            name="query encoder output",
        )
        query_embedding = _normalize_rows(query_embedding)[0]
        chunk_scores = self._embeddings @ query_embedding

        document_scores = np.full(
            len(self._document_ids), -np.inf, dtype=np.float32
        )
        np.maximum.at(
            document_scores, self._chunk_document_indices, chunk_scores
        )
        return np.nan_to_num(
            document_scores,
            copy=False,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=0.0,
        )


__all__ = [
    "BM25ChunkMaxIndex",
    "DenseChunkMaxIndex",
    "SentenceTransformerTextEncoder",
    "TextDocument",
    "TextEncoder",
    "chunk_text",
    "tokenize",
]
