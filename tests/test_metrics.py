import pytest
import torch

from colpali_triton.metrics import rank_scores, retrieval_metrics


def test_rank_scores_matches_trec_tie_order() -> None:
    query_ids = ("q0",)
    document_ids = ("1", "2", "10")
    scores = torch.zeros(1, 3)

    rankings = rank_scores(query_ids, document_ids, scores)

    assert rankings == {"q0": ("2", "10", "1")}


def test_tie_order_does_not_depend_on_corpus_column_order() -> None:
    first = rank_scores(
        ("q",), ("1", "2", "10"), torch.tensor([[0.5, 0.5, 0.5]])
    )
    second = rank_scores(
        ("q",), ("10", "1", "2"), torch.tensor([[0.5, 0.5, 0.5]])
    )

    assert first == second


def test_rank_scores_handles_negative_and_zero_scores() -> None:
    rankings = rank_scores(
        ("q",),
        ("negative", "zero", "positive"),
        torch.tensor([[-2.0, 0.0, 1.5]]),
    )

    assert rankings["q"] == ("positive", "zero", "negative")


def test_hand_computed_macro_metrics() -> None:
    ranking = ("d0", "d1", "d2", "d3", "d4", "d5")
    rankings = {f"q{index}": ranking for index in range(4)}
    qrels = {
        "q0": frozenset({"d0"}),
        "q1": frozenset({"d2"}),
        "q2": frozenset({"d1", "d3", "d5"}),
        "q3": frozenset({"d5"}),
    }

    metrics = retrieval_metrics(
        rankings, qrels, k=5, include_mrr=True
    )

    assert metrics.num_queries == 4
    assert metrics.ndcg_at_k == pytest.approx(0.4995473143666032)
    assert metrics.recall_at_k == pytest.approx(2.0 / 3.0)
    assert metrics.mrr_at_k == pytest.approx(11.0 / 24.0)


def test_cutoff_limits_ideal_dcg_but_recall_uses_all_relevant_documents() -> None:
    ranking = tuple(f"d{index}" for index in range(6))

    metrics = retrieval_metrics(
        {"q": ranking},
        {"q": frozenset(ranking)},
        k=5,
        include_mrr=True,
    )

    assert metrics.ndcg_at_k == 1.0
    assert metrics.recall_at_k == pytest.approx(5.0 / 6.0)
    assert metrics.mrr_at_k == 1.0


def test_recall_is_macro_averaged() -> None:
    rankings = {
        "q0": ("d0", "d1", "d2", "d3", "d4"),
        "q1": ("d0", "d1", "d2", "d3", "d4"),
    }
    qrels = {
        "q0": frozenset({"d0"}),
        "q1": frozenset({"d0", "d2", "d3", "d4"}),
    }

    metrics = retrieval_metrics(rankings, qrels, k=1)

    assert metrics.recall_at_k == pytest.approx(0.625)
    assert metrics.mrr_at_k is None
    assert metrics.to_dict() == {
        "k": 1,
        "num_queries": 2,
        "ndcg@1": 1.0,
        "recall@1": pytest.approx(0.625),
    }


@pytest.mark.parametrize(
    ("scores", "error_type"),
    [
        ([[1.0]], TypeError),
        (torch.tensor([1.0]), ValueError),
        (torch.ones(1, 1, 1), ValueError),
        (torch.ones(1, 1, dtype=torch.int64), TypeError),
        (torch.tensor([[float("nan")]]), ValueError),
        (torch.tensor([[float("inf")]]), ValueError),
        (torch.tensor([[float("-inf")]]), ValueError),
    ],
)
def test_rank_scores_rejects_invalid_scores(scores: object, error_type: type) -> None:
    with pytest.raises(error_type):
        rank_scores(("q",), ("d",), scores)  # type: ignore[arg-type]


def test_rank_scores_rejects_non_strided_scores() -> None:
    scores = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (1, 1)
    )

    with pytest.raises(TypeError, match="strided"):
        rank_scores(("q",), ("d",), scores)


@pytest.mark.parametrize(
    "document_ids",
    [
        {"d0", "d1"},
        frozenset({"d0", "d1"}),
        {"d0": 0, "d1": 1},
        (document_id for document_id in ("d0", "d1")),
    ],
)
def test_rank_scores_rejects_unordered_or_one_shot_ids(
    document_ids: object,
) -> None:
    with pytest.raises(TypeError, match="sequence"):
        rank_scores(
            ("q",),
            document_ids,  # type: ignore[arg-type]
            torch.ones(1, 2),
        )


@pytest.mark.parametrize(
    ("query_ids", "document_ids", "scores"),
    [
        ((), ("d",), torch.empty(0, 1)),
        (("q", "q"), ("d",), torch.ones(2, 1)),
        (("q",), ("d", "d"), torch.ones(1, 2)),
        (("q",), ("d",), torch.ones(2, 1)),
        (("",), ("d",), torch.ones(1, 1)),
    ],
)
def test_rank_scores_rejects_invalid_ids_or_shape(
    query_ids: object, document_ids: object, scores: torch.Tensor
) -> None:
    with pytest.raises((TypeError, ValueError)):
        rank_scores(
            query_ids,  # type: ignore[arg-type]
            document_ids,  # type: ignore[arg-type]
            scores,
        )


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_metrics_rejects_invalid_cutoff(k: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        retrieval_metrics(
            {"q": ("d",)},
            {"q": frozenset({"d"})},
            k=k,  # type: ignore[arg-type]
        )


def test_metrics_rejects_mismatched_query_sets() -> None:
    with pytest.raises(ValueError, match="identical query IDs"):
        retrieval_metrics(
            {"q0": ("d",)}, {"q1": frozenset({"d"})}
        )


@pytest.mark.parametrize(
    ("rankings", "qrels", "match"),
    [
        ({"q": ("d", "d")}, {"q": frozenset({"d"})}, "duplicates"),
        (
            {"q0": ("d0", "d1"), "q1": ("d0", "d2")},
            {"q0": frozenset({"d0"}), "q1": frozenset({"d0"})},
            "complete permutation",
        ),
        ({"q": ("d",)}, {"q": frozenset()}, "must not be empty"),
        (
            {"q": ("d0",)},
            {"q": frozenset({"missing"})},
            "absent from the corpus",
        ),
    ],
)
def test_metrics_rejects_invalid_rankings_or_qrels(
    rankings: object, qrels: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        retrieval_metrics(
            rankings,  # type: ignore[arg-type]
            qrels,  # type: ignore[arg-type]
        )


def test_metrics_rejects_non_boolean_mrr_flag() -> None:
    with pytest.raises(TypeError, match="bool"):
        retrieval_metrics(
            {"q": ("d",)},
            {"q": frozenset({"d"})},
            include_mrr=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "qrels",
    [
        {"q": ["d"]},
        {"q": ("d",)},
        {"q": {"d": 1}},
    ],
)
def test_metrics_rejects_non_set_relevance_values(qrels: object) -> None:
    with pytest.raises(TypeError, match="set of strings"):
        retrieval_metrics(
            {"q": ("d",)},
            qrels,  # type: ignore[arg-type]
        )
