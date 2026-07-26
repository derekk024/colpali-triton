from dataclasses import FrozenInstanceError
import json
from math import log2
from typing import Dict, Sequence

import numpy as np
import pytest

import colpali_triton.evaluation as evaluation_module
from colpali_triton.evaluation import (
    EvaluationResult,
    RetrievalIndex,
    evaluate_retrieval,
)
from colpali_triton.vidore import TextQuery


class FakeIndex:
    def __init__(
        self,
        document_ids: Sequence[str],
        scores_by_text: Dict[str, object],
        *,
        index_size_bytes: object = 128,
    ) -> None:
        self.document_ids = tuple(document_ids)
        self.scores_by_text = scores_by_text
        self.index_size_bytes = index_size_bytes
        self.calls = []

    def score(self, query_text: str) -> object:
        self.calls.append(query_text)
        return self.scores_by_text[query_text]


def test_evaluate_retrieval_measures_full_rankings_and_multirelevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = FakeIndex(
        ("d0", "d1", "d2", "d3"),
        {
            "first": np.asarray([4.0, 3.0, 2.0, 1.0]),
            "second": np.asarray([2.0, 1.0, 3.0, 4.0]),
        },
        index_size_bytes=512,
    )
    queries = (
        TextQuery("q0", "first", frozenset({"d0", "d2"})),
        TextQuery("q1", "second", frozenset({"d1", "d3"})),
    )
    timestamps = iter((0.0, 0.1, 0.3, 1.0, 1.4, 2.0))
    monkeypatch.setattr(
        evaluation_module, "perf_counter", lambda: next(timestamps)
    )

    result = evaluate_retrieval(index, queries, k=3)

    ideal_dcg = 1.0 + 1.0 / log2(3.0)
    expected_ndcg = ((1.0 + 1.0 / log2(4.0)) / ideal_dcg)
    expected_ndcg += 1.0 / ideal_dcg
    expected_ndcg /= 2.0
    assert result.ndcg_at_k == pytest.approx(expected_ndcg)
    assert result.recall_at_k == pytest.approx(0.75)
    assert result.mrr_at_k == pytest.approx(1.0)
    assert result.query_scoring_seconds == pytest.approx(0.5)
    assert result.ranking_seconds == pytest.approx(0.8)
    assert result.total_scoring_seconds == pytest.approx(1.3)
    assert result.latency_p50_ms == pytest.approx(650.0)
    assert result.latency_p95_ms == pytest.approx(965.0)
    assert result.scoring_query_document_pairs_per_second == pytest.approx(
        8.0 / 0.5
    )
    assert (
        result.end_to_end_query_document_pairs_per_second
        == pytest.approx(8.0 / 1.3)
    )
    assert result.documents_per_second == pytest.approx(8.0 / 1.3)
    assert result.index_size_bytes == 512
    assert result.num_queries == 2
    assert result.num_documents == 4
    assert result.to_dict() == {
        "k": 3,
        "num_queries": 2,
        "num_documents": 4,
        "warmup_count": 0,
        "ndcg@3": pytest.approx(expected_ndcg),
        "recall@3": pytest.approx(0.75),
        "mrr@3": pytest.approx(1.0),
        "latency_p50_ms": pytest.approx(650.0),
        "latency_p95_ms": pytest.approx(965.0),
        "query_scoring_seconds": pytest.approx(0.5),
        "postprocessing_and_ranking_seconds": pytest.approx(0.8),
        "total_scoring_seconds": pytest.approx(1.3),
        "scoring_query_document_pairs_per_second": pytest.approx(8.0 / 0.5),
        "end_to_end_query_document_pairs_per_second": pytest.approx(8.0 / 1.3),
        "logical_index_payload_bytes": 512,
    }
    assert index.calls == ["first", "second"]


def test_evaluate_retrieval_uses_canonical_trec_tie_order() -> None:
    index = FakeIndex(
        ("a", "z", "m"),
        {"tied": np.zeros(3, dtype=np.float32)},
    )
    query = TextQuery("q", "tied", frozenset({"z"}))

    result = evaluate_retrieval(index, (query,), k=3)

    assert result.ndcg_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.mrr_at_k == 1.0


def test_warmups_cycle_queries_and_are_excluded_from_query_count() -> None:
    index = FakeIndex(
        ("d0", "d1"),
        {
            "first": np.asarray([1.0, 0.0]),
            "second": np.asarray([0.0, 1.0]),
        },
    )
    queries = (
        TextQuery("q0", "first", frozenset({"d0"})),
        TextQuery("q1", "second", frozenset({"d1"})),
    )

    result = evaluate_retrieval(
        index, queries, k=1, warmup_count=3
    )

    assert index.calls == [
        "first",
        "second",
        "first",
        "first",
        "second",
    ]
    assert result.warmup_count == 3
    assert result.num_queries == 2


def test_result_is_frozen_finite_and_json_ready() -> None:
    index = FakeIndex(
        ("d",), {"query": np.asarray([1.0], dtype=np.float32)}
    )
    result = evaluate_retrieval(
        index,
        (TextQuery("q", "query", frozenset({"d"})),),
        k=1,
    )

    assert isinstance(result, EvaluationResult)
    assert result.latency_p50_ms >= 0.0
    assert result.latency_p95_ms >= 0.0
    assert result.total_scoring_seconds >= 0.0
    assert result.documents_per_second >= 0.0
    assert np.isfinite(
        [
            result.latency_p50_ms,
            result.latency_p95_ms,
            result.total_scoring_seconds,
            result.scoring_query_document_pairs_per_second,
            result.end_to_end_query_document_pairs_per_second,
            result.documents_per_second,
        ]
    ).all()
    assert json.loads(json.dumps(result.to_dict()))["ndcg@1"] == 1.0
    with pytest.raises(FrozenInstanceError):
        result.k = 2  # type: ignore[misc]


def test_fake_index_satisfies_runtime_protocol() -> None:
    index = FakeIndex(("d",), {"query": np.asarray([1.0])})

    assert isinstance(index, RetrievalIndex)


@pytest.mark.parametrize(
    ("document_ids", "error_type", "match"),
    [
        ((), ValueError, "must not be empty"),
        (("d", "d"), ValueError, "duplicates"),
        (("",), ValueError, "empty strings"),
        ((1,), TypeError, "only strings"),
    ],
)
def test_rejects_invalid_index_document_ids(
    document_ids: object, error_type: type, match: str
) -> None:
    index = FakeIndex(
        document_ids,  # type: ignore[arg-type]
        {"query": np.asarray([1.0])},
    )
    query = TextQuery("q", "query", frozenset({"d"}))

    with pytest.raises(error_type, match=match):
        evaluate_retrieval(index, (query,))


def test_rejects_duplicate_query_ids() -> None:
    index = FakeIndex(
        ("d0", "d1"), {"text": np.asarray([1.0, 0.0])}
    )
    queries = (
        TextQuery("same", "text", frozenset({"d0"})),
        TextQuery("same", "text", frozenset({"d1"})),
    )

    with pytest.raises(ValueError, match="query IDs must be unique"):
        evaluate_retrieval(index, queries)


def test_rejects_qrels_outside_index_corpus() -> None:
    index = FakeIndex(("d0",), {"text": np.asarray([1.0])})
    query = TextQuery("q", "text", frozenset({"missing"}))

    with pytest.raises(ValueError, match="absent from the index"):
        evaluate_retrieval(index, (query,))


@pytest.mark.parametrize(
    ("scores", "error_type", "match"),
    [
        ([1.0, 0.0], TypeError, "NumPy array"),
        (np.asarray([[1.0, 0.0]]), ValueError, "one-dimensional"),
        (np.asarray([1.0]), ValueError, "shape"),
        (np.asarray([1, 0]), TypeError, "floating dtype"),
        (np.asarray([np.nan, 0.0]), ValueError, "finite"),
        (np.asarray([np.inf, 0.0]), ValueError, "finite"),
    ],
)
def test_rejects_invalid_score_vectors(
    scores: object, error_type: type, match: str
) -> None:
    index = FakeIndex(("d0", "d1"), {"text": scores})
    query = TextQuery("q", "text", frozenset({"d0"}))

    with pytest.raises(error_type, match=match):
        evaluate_retrieval(index, (query,))


@pytest.mark.parametrize(
    ("index_size_bytes", "error_type"),
    [
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_index_size(
    index_size_bytes: object, error_type: type
) -> None:
    index = FakeIndex(
        ("d",),
        {"text": np.asarray([1.0])},
        index_size_bytes=index_size_bytes,
    )
    query = TextQuery("q", "text", frozenset({"d"}))

    with pytest.raises(error_type, match="index_size_bytes"):
        evaluate_retrieval(index, (query,))


@pytest.mark.parametrize(
    ("queries", "error_type"),
    [
        ((), ValueError),
        ((object(),), TypeError),
        ((query for query in ()), TypeError),
    ],
)
def test_rejects_invalid_query_collections(
    queries: object, error_type: type
) -> None:
    index = FakeIndex(("d",), {"text": np.asarray([1.0])})

    with pytest.raises(error_type):
        evaluate_retrieval(index, queries)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("k", "warmup_count", "error_type"),
    [
        (0, 0, ValueError),
        (True, 0, TypeError),
        (1, -1, ValueError),
        (1, True, TypeError),
    ],
)
def test_rejects_invalid_evaluation_parameters(
    k: object, warmup_count: object, error_type: type
) -> None:
    index = FakeIndex(("d",), {"text": np.asarray([1.0])})
    query = TextQuery("q", "text", frozenset({"d"}))

    with pytest.raises(error_type):
        evaluate_retrieval(
            index,
            (query,),
            k=k,  # type: ignore[arg-type]
            warmup_count=warmup_count,  # type: ignore[arg-type]
        )
