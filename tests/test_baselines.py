from dataclasses import dataclass
import sys
from types import SimpleNamespace
from typing import Dict, Sequence

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from colpali_triton.baselines import (
    BM25ChunkMaxIndex,
    DenseChunkMaxIndex,
    SentenceTransformerTextEncoder,
    chunk_text,
    tokenize,
)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


class FakeEncoder:
    def __init__(self, vectors: Dict[str, Sequence[float]]) -> None:
        self.vectors = {
            text: np.asarray(vector, dtype=np.float64)
            for text, vector in vectors.items()
        }
        self.calls = []
        self.dimension = len(next(iter(self.vectors.values())))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(tuple(texts))
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float64)
        return np.stack([self.vectors[text] for text in texts])


def test_unicode_tokenizer_normalizes_case_width_and_separators() -> None:
    assert tokenize(" ＦＯＯ Café—Straße_42 中文! ") == (
        "foo",
        "café",
        "strasse",
        "42",
        "中文",
    )


def test_chunk_text_uses_fixed_whitespace_windows() -> None:
    assert chunk_text(" a\n b  c\td e ", max_words=3, overlap=1) == (
        "a b c",
        "c d e",
    )
    assert chunk_text("   ", max_words=3, overlap=1) == ("",)


@pytest.mark.parametrize(
    ("max_words", "overlap", "error_type"),
    [
        (0, 0, ValueError),
        (-1, 0, ValueError),
        (2, -1, ValueError),
        (2, 2, ValueError),
        (2, 3, ValueError),
        (True, 0, TypeError),
        (2, False, TypeError),
        (2.0, 0, TypeError),
    ],
)
def test_chunk_text_rejects_invalid_parameters(
    max_words: object, overlap: object, error_type: type
) -> None:
    with pytest.raises(error_type):
        chunk_text(
            "text",
            max_words=max_words,  # type: ignore[arg-type]
            overlap=overlap,  # type: ignore[arg-type]
        )


def test_bm25_matches_hand_computed_okapi_score() -> None:
    documents = (
        Document("d0", "apple apple"),
        Document("d1", "banana"),
        Document("d2", "banana"),
    )
    index = BM25ChunkMaxIndex(documents, max_words=10, overlap=0)

    scores = index.score("apple")

    idf = np.log(2.5) - np.log(1.5)
    average_length = 4.0 / 3.0
    denominator = 2.0 + 1.5 * (
        1.0 - 0.75 + 0.75 * 2.0 / average_length
    )
    expected = idf * (2.0 * 2.5) / denominator
    assert scores == pytest.approx([expected, 0.0, 0.0])
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()


def test_bm25_empty_and_oov_queries_are_stable() -> None:
    index = BM25ChunkMaxIndex(
        (Document("blank", " \n"), Document("text", "known term")),
        max_words=2,
        overlap=0,
    )

    assert index.score("") == pytest.approx([0.0, 0.0])
    assert index.score("out-of-vocabulary") == pytest.approx([0.0, 0.0])
    assert index.document_ids == ("blank", "text")
    assert index.index_size_bytes > 0
    assert index.logical_index_bytes == index.index_size_bytes
    assert index.index_nbytes == index.index_size_bytes


def test_bm25_aggregates_the_best_chunk_per_page() -> None:
    index = BM25ChunkMaxIndex(
        (
            Document("d0", "needle filler filler filler"),
            Document("d1", "needle needle"),
            Document("d2", "other words more tokens"),
        ),
        max_words=2,
        overlap=0,
    )

    scores = index.score("needle")

    assert index.chunks == (
        "needle filler",
        "filler filler",
        "needle needle",
        "other words",
        "more tokens",
    )
    assert scores[1] > scores[0] > scores[2]


@pytest.mark.parametrize(
    "query",
    [
        "common",
        "common common alpha",
        "not-in-the-vocabulary",
        "beta gamma delta theta",
        "",
    ],
)
def test_bm25_page_scores_match_rank_bm25_0_2_2(query: str) -> None:
    documents = (
        Document("blank", ""),
        Document("short", "common alpha"),
        Document("medium", "common beta common gamma"),
        Document(
            "long",
            "common delta common zeta common theta",
        ),
        Document("rare", "iota"),
    )
    max_words = 3
    overlap = 1
    k1 = 1.2
    b = 0.6
    epsilon = 0.4
    index = BM25ChunkMaxIndex(
        documents,
        max_words=max_words,
        overlap=overlap,
        k1=k1,
        b=b,
        epsilon=epsilon,
    )

    reference_tokens = [list(tokenize(chunk)) for chunk in index.chunks]
    reference = BM25Okapi(
        reference_tokens,
        k1=k1,
        b=b,
        epsilon=epsilon,
    )
    common_document_frequency = sum(
        "common" in tokens for tokens in reference_tokens
    )
    assert common_document_frequency > len(reference_tokens) / 2
    assert reference_tokens[0] == []
    assert len({len(tokens) for tokens in reference_tokens}) > 1

    reference_chunk_scores = reference.get_scores(list(tokenize(query)))
    chunk_document_indices = np.asarray(
        [
            document_index
            for document_index, document in enumerate(documents)
            for _ in chunk_text(
                document.text,
                max_words=max_words,
                overlap=overlap,
            )
        ],
        dtype=np.int64,
    )
    expected_page_scores = np.full(len(documents), -np.inf)
    np.maximum.at(
        expected_page_scores,
        chunk_document_indices,
        reference_chunk_scores,
    )
    expected_page_scores = np.nan_to_num(
        expected_page_scores,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )

    np.testing.assert_allclose(
        index.score(query),
        expected_page_scores,
        rtol=1e-13,
        atol=1e-15,
    )


def test_dense_normalizes_vectors_and_uses_chunk_max() -> None:
    encoder = FakeEncoder(
        {
            "wrong": (0.0, 2.0),
            "right": (3.0, 0.0),
            "middle": (1.0, 1.0),
            "query": (10.0, 0.0),
        }
    )
    index = DenseChunkMaxIndex(
        (
            Document("best", "wrong right"),
            Document("middle", "middle"),
            Document("blank", " "),
        ),
        encoder,
        max_words=1,
        overlap=0,
    )

    scores = index.score("query")

    assert scores.dtype == np.float32
    assert scores == pytest.approx([1.0, 2**-0.5, 0.0])
    assert index.document_ids == ("best", "middle", "blank")
    assert index.chunk_document_indices.tolist() == [0, 0, 1, 2]
    assert index.embeddings[-1].tolist() == [0.0, 0.0]
    assert np.linalg.norm(index.embeddings[0]) == pytest.approx(1.0)
    assert index.logical_index_bytes == (
        index.embeddings.nbytes + index.chunk_document_indices.nbytes
    )
    assert index.index_size_bytes == index.logical_index_bytes
    assert encoder.calls == [
        ("wrong", "right", "middle"),
        ("query",),
    ]


def test_dense_blank_query_does_not_call_encoder() -> None:
    encoder = FakeEncoder({"document": (1.0, 0.0)})
    index = DenseChunkMaxIndex(
        (Document("d0", "document"),),
        encoder,
        max_words=10,
        overlap=0,
    )

    assert index.score("  ").tolist() == [0.0]
    assert encoder.calls == [("document",)]


def test_dense_all_blank_documents_get_zero_vectors() -> None:
    encoder = FakeEncoder({"unused": (1.0, 0.0, 0.0), "q": (1.0, 0.0, 0.0)})
    index = DenseChunkMaxIndex(
        (Document("d0", ""), Document("d1", "\n")),
        encoder,
        max_words=10,
        overlap=0,
    )

    assert index.embeddings.shape == (2, 3)
    assert not index.embeddings.any()
    assert index.score("q").tolist() == [0.0, 0.0]
    assert encoder.calls == [(), ("q",)]


@pytest.mark.parametrize(
    "vectors",
    [
        {"doc": (float("nan"), 0.0)},
        {"doc": (float("inf"), 0.0)},
    ],
)
def test_dense_rejects_nonfinite_encoder_output(vectors: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        DenseChunkMaxIndex(
            (Document("d", "doc"),),
            FakeEncoder(vectors),  # type: ignore[arg-type]
            max_words=2,
            overlap=0,
        )


def test_dense_rejects_query_dimension_change() -> None:
    class ChangingEncoder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            dimension = 2 if texts == ["doc"] else 3
            return np.ones((len(texts), dimension))

    index = DenseChunkMaxIndex(
        (Document("d", "doc"),),
        ChangingEncoder(),
        max_words=2,
        overlap=0,
    )

    with pytest.raises(ValueError, match="dimension 2"):
        index.score("query")


@pytest.mark.parametrize(
    "documents",
    [
        (),
        (Document("", "text"),),
        (Document("same", "a"), Document("same", "b")),
    ],
)
def test_indexes_reject_invalid_document_collections(documents: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25ChunkMaxIndex(
            documents,  # type: ignore[arg-type]
            max_words=2,
            overlap=0,
        )


def test_sentence_transformer_wrapper_is_lazy_and_validates_pin() -> None:
    encoder = SentenceTransformerTextEncoder(
        "BAAI/bge-small-en-v1.5",
        revision="0123456789abcdef",
        device="cpu",
        batch_size=16,
        max_seq_length=512,
        cache_folder="/tmp/model-cache",
    )

    assert encoder._model is None
    assert encoder.model_id == "BAAI/bge-small-en-v1.5"
    assert encoder.revision == "0123456789abcdef"

    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerTextEncoder("model", revision="")
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerTextEncoder("model", revision="pin", batch_size=0)
    with pytest.raises(ValueError, match="max_seq_length"):
        SentenceTransformerTextEncoder(
            "model", revision="pin", max_seq_length=0
        )
    with pytest.raises(ValueError, match="cache_folder"):
        SentenceTransformerTextEncoder(
            "model", revision="pin", cache_folder=""
        )


def test_sentence_transformer_can_be_loaded_and_inspected_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls = []

    class Transformer:
        def __init__(self) -> None:
            parameter = SimpleNamespace(dtype="torch.float32")
            self.auto_model = SimpleNamespace(
                parameters=lambda: iter((parameter,))
            )
            self.tokenizer = self._tokenize

        @staticmethod
        def _tokenize(texts: Sequence[str], **kwargs: object) -> object:
            assert kwargs == {
                "add_special_tokens": True,
                "padding": False,
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
                "return_length": True,
            }
            return {"length": [len(text.split()) + 2 for text in texts]}

    class Pooling:
        pooling_mode_cls_token = True
        pooling_mode_mean_tokens = False
        pooling_mode_max_tokens = False
        pooling_mode_mean_sqrt_len_tokens = False
        pooling_mode_weightedmean_tokens = False
        pooling_mode_lasttoken = False

    class Normalize:
        pass

    class FakeSentenceTransformer:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            constructor_calls.append((model_id, kwargs))
            self.max_seq_length = 8192
            self._modules = {
                "0": Transformer(),
                "1": Pooling(),
                "2": Normalize(),
            }

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def _first_module(self) -> object:
            return self._modules["0"]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = SentenceTransformerTextEncoder(
        "BAAI/bge-m3",
        revision="full-revision",
        device="cpu",
        max_seq_length=1024,
        cache_folder="/tmp/model-cache",
    )

    assert constructor_calls == []
    assert encoder.load() is encoder
    assert encoder.embedding_dimension == 3
    assert encoder.effective_max_seq_length == 1024
    assert encoder.effective_model_dtype == "float32"
    assert encoder.effective_pooling_mode == "cls"
    assert encoder.has_normalization_module is True
    assert encoder.token_lengths(("two words", "one")) == pytest.approx(
        [4, 3]
    )
    assert encoder.token_lengths(()).shape == (0,)
    assert constructor_calls == [
        (
            "BAAI/bge-m3",
            {
                "revision": "full-revision",
                "device": "cpu",
                "cache_folder": "/tmp/model-cache",
            },
        )
    ]


def test_sentence_transformer_reports_pooling_and_final_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transformer:
        auto_model = SimpleNamespace(
            parameters=lambda: iter((SimpleNamespace(dtype="torch.float16"),))
        )

    class Pooling:
        pooling_mode_cls_token = False
        pooling_mode_mean_tokens = False
        pooling_mode_max_tokens = False
        pooling_mode_mean_sqrt_len_tokens = True
        pooling_mode_weightedmean_tokens = True
        pooling_mode_lasttoken = True

    class Normalize:
        pass

    class OutputModule:
        pass

    class FakeSentenceTransformer:
        max_seq_length = 512

        def __init__(self, *_: object, **__: object) -> None:
            self._modules = {
                "0": Transformer(),
                "1": Normalize(),
                "2": Pooling(),
                "3": OutputModule(),
            }

        def _first_module(self) -> object:
            return self._modules["0"]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    encoder = SentenceTransformerTextEncoder("model", revision="revision")

    assert encoder.effective_model_dtype == "float16"
    assert encoder.effective_pooling_mode == (
        "mean_sqrt_len_tokens+weightedmean+lasttoken"
    )
    assert encoder.has_normalization_module is False
