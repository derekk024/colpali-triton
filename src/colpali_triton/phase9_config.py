"""Strict configuration for the scoped Phase 9 ColPali evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Dict, FrozenSet, Mapping, Sequence, Tuple, Union


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FORMAT = "colpali-triton-phase9-subset-v1"
_TASK_NAMES = frozenset({"docvqa", "infovqa"})
_SCORERS = (
    "late_interaction_maxsim",
    "l2_normalized_masked_mean",
)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class Phase9SelectionSpec:
    """Content-independent subset selection rule."""

    method: str
    salt: str
    queries_per_task: int
    documents_per_task: int
    positive_policy: str
    distractor_policy: str

    def __post_init__(self) -> None:
        if self.method != "sha256_identifier_order":
            raise ValueError("Phase 9 selection method is not supported")
        if not isinstance(self.salt, str) or not self.salt:
            raise ValueError("selection salt must be a nonempty string")
        _positive_integer(self.queries_per_task, "queries_per_task")
        _positive_integer(self.documents_per_task, "documents_per_task")
        if self.queries_per_task > self.documents_per_task:
            raise ValueError(
                "queries_per_task may not exceed documents_per_task"
            )
        if (
            self.positive_policy
            != "include_all_selected_query_positives"
        ):
            raise ValueError("Phase 9 positive policy is not supported")
        if self.distractor_policy != "sha256_identifier_order":
            raise ValueError("Phase 9 distractor policy is not supported")

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "salt": self.salt,
            "queries_per_task": self.queries_per_task,
            "documents_per_task": self.documents_per_task,
            "positive_policy": self.positive_policy,
            "distractor_policy": self.distractor_policy,
        }


@dataclass(frozen=True)
class Phase9EvaluationSpec:
    """Exact local evaluation settings."""

    k: int
    warmup_queries: int
    document_batch_size: int
    score_document_batch_size: int
    dtype: str
    device: str
    attention_implementation: str
    scorers: Tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.k, "evaluation k")
        if (
            isinstance(self.warmup_queries, bool)
            or not isinstance(self.warmup_queries, int)
            or self.warmup_queries < 0
        ):
            raise ValueError("warmup_queries must be a nonnegative integer")
        _positive_integer(
            self.document_batch_size, "document_batch_size"
        )
        _positive_integer(
            self.score_document_batch_size,
            "score_document_batch_size",
        )
        if self.document_batch_size != 1:
            raise ValueError(
                "the memory-audited local document batch size must be 1"
            )
        if self.dtype != "bfloat16":
            raise ValueError("Phase 9 dtype must be bfloat16")
        if self.device != "mps":
            raise ValueError("Phase 9 device must be mps")
        if self.attention_implementation != "eager":
            raise ValueError("Phase 9 attention implementation must be eager")
        prepared_scorers = tuple(self.scorers)
        if prepared_scorers != _SCORERS:
            raise ValueError(
                "Phase 9 scorers must contain the exact ordered pair"
            )
        object.__setattr__(self, "scorers", prepared_scorers)

    def to_dict(self) -> Dict[str, object]:
        return {
            "k": self.k,
            "warmup_queries": self.warmup_queries,
            "document_batch_size": self.document_batch_size,
            "score_document_batch_size": (
                self.score_document_batch_size
            ),
            "dtype": self.dtype,
            "device": self.device,
            "attention_implementation": self.attention_implementation,
            "scorers": list(self.scorers),
        }


@dataclass(frozen=True)
class Phase9TaskSpec:
    """Expected source and selected-image fingerprints for one task."""

    source_spec_fingerprint: str
    selection_fingerprint: str
    image_content_fingerprint: str

    def __post_init__(self) -> None:
        _digest(
            self.source_spec_fingerprint, "source_spec_fingerprint"
        )
        _digest(self.selection_fingerprint, "selection_fingerprint")
        _digest(
            self.image_content_fingerprint,
            "image_content_fingerprint",
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "source_spec_fingerprint": self.source_spec_fingerprint,
            "selection_fingerprint": self.selection_fingerprint,
            "image_content_fingerprint": self.image_content_fingerprint,
        }


@dataclass(frozen=True)
class Phase9Config:
    """Complete immutable Phase 9 experiment contract."""

    schema_version: int
    format: str
    seed: int
    model_manifest_fingerprint: str
    dataset_manifest_fingerprint: str
    selection: Phase9SelectionSpec
    evaluation: Phase9EvaluationSpec
    tasks: Mapping[str, Phase9TaskSpec]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be exactly 1")
        if self.format != _FORMAT:
            raise ValueError(f"format must be exactly {_FORMAT!r}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        _digest(
            self.model_manifest_fingerprint,
            "model_manifest_fingerprint",
        )
        _digest(
            self.dataset_manifest_fingerprint,
            "dataset_manifest_fingerprint",
        )
        if not isinstance(self.selection, Phase9SelectionSpec):
            raise TypeError("selection must be a Phase9SelectionSpec")
        if not isinstance(self.evaluation, Phase9EvaluationSpec):
            raise TypeError("evaluation must be a Phase9EvaluationSpec")
        if not isinstance(self.tasks, Mapping):
            raise TypeError("tasks must be a mapping")
        if frozenset(self.tasks) != _TASK_NAMES:
            raise ValueError("tasks must define exactly docvqa and infovqa")
        prepared: Dict[str, Phase9TaskSpec] = {}
        for name, task in self.tasks.items():
            if not isinstance(task, Phase9TaskSpec):
                raise TypeError("tasks must contain Phase9TaskSpec values")
            prepared[name] = task
        object.__setattr__(self, "tasks", MappingProxyType(prepared))

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "seed": self.seed,
            "model_manifest_fingerprint": (
                self.model_manifest_fingerprint
            ),
            "dataset_manifest_fingerprint": (
                self.dataset_manifest_fingerprint
            ),
            "selection": self.selection.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "tasks": {
                name: task.to_dict() for name, task in self.tasks.items()
            },
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def fingerprint(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _keys(
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


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def phase9_config_from_dict(value: object) -> Phase9Config:
    """Strictly parse an in-memory Phase 9 configuration."""

    root = _mapping(value, "config")
    _keys(
        root,
        frozenset(
            {
                "schema_version",
                "format",
                "seed",
                "model_manifest_fingerprint",
                "dataset_manifest_fingerprint",
                "selection",
                "evaluation",
                "tasks",
            }
        ),
        "config",
    )
    selection = _mapping(root["selection"], "selection")
    _keys(
        selection,
        frozenset(
            {
                "method",
                "salt",
                "queries_per_task",
                "documents_per_task",
                "positive_policy",
                "distractor_policy",
            }
        ),
        "selection",
    )
    evaluation = _mapping(root["evaluation"], "evaluation")
    _keys(
        evaluation,
        frozenset(
            {
                "k",
                "warmup_queries",
                "document_batch_size",
                "score_document_batch_size",
                "dtype",
                "device",
                "attention_implementation",
                "scorers",
            }
        ),
        "evaluation",
    )
    raw_scorers = evaluation["scorers"]
    if (
        isinstance(raw_scorers, (str, bytes))
        or not isinstance(raw_scorers, Sequence)
        or any(not isinstance(item, str) for item in raw_scorers)
    ):
        raise ValueError("evaluation.scorers must be a sequence of strings")

    raw_tasks = _mapping(root["tasks"], "tasks")
    tasks: Dict[str, Phase9TaskSpec] = {}
    for name, raw_task in raw_tasks.items():
        task = _mapping(raw_task, f"tasks.{name}")
        _keys(
            task,
            frozenset(
                {
                    "source_spec_fingerprint",
                    "selection_fingerprint",
                    "image_content_fingerprint",
                }
            ),
            f"tasks.{name}",
        )
        tasks[name] = Phase9TaskSpec(
            source_spec_fingerprint=_digest(
                task["source_spec_fingerprint"],
                f"tasks.{name}.source_spec_fingerprint",
            ),
            selection_fingerprint=_digest(
                task["selection_fingerprint"],
                f"tasks.{name}.selection_fingerprint",
            ),
            image_content_fingerprint=_digest(
                task["image_content_fingerprint"],
                f"tasks.{name}.image_content_fingerprint",
            ),
        )

    return Phase9Config(
        schema_version=_positive_integer(
            root["schema_version"], "schema_version"
        ),
        format=_string(root["format"], "format"),
        seed=(
            root["seed"]
            if isinstance(root["seed"], int)
            and not isinstance(root["seed"], bool)
            else -1
        ),
        model_manifest_fingerprint=_digest(
            root["model_manifest_fingerprint"],
            "model_manifest_fingerprint",
        ),
        dataset_manifest_fingerprint=_digest(
            root["dataset_manifest_fingerprint"],
            "dataset_manifest_fingerprint",
        ),
        selection=Phase9SelectionSpec(
            method=_string(selection["method"], "selection.method"),
            salt=_string(selection["salt"], "selection.salt"),
            queries_per_task=_positive_integer(
                selection["queries_per_task"],
                "selection.queries_per_task",
            ),
            documents_per_task=_positive_integer(
                selection["documents_per_task"],
                "selection.documents_per_task",
            ),
            positive_policy=_string(
                selection["positive_policy"],
                "selection.positive_policy",
            ),
            distractor_policy=_string(
                selection["distractor_policy"],
                "selection.distractor_policy",
            ),
        ),
        evaluation=Phase9EvaluationSpec(
            k=_positive_integer(evaluation["k"], "evaluation.k"),
            warmup_queries=(
                evaluation["warmup_queries"]
                if isinstance(evaluation["warmup_queries"], int)
                and not isinstance(evaluation["warmup_queries"], bool)
                else -1
            ),
            document_batch_size=_positive_integer(
                evaluation["document_batch_size"],
                "evaluation.document_batch_size",
            ),
            score_document_batch_size=_positive_integer(
                evaluation["score_document_batch_size"],
                "evaluation.score_document_batch_size",
            ),
            dtype=_string(evaluation["dtype"], "evaluation.dtype"),
            device=_string(evaluation["device"], "evaluation.device"),
            attention_implementation=_string(
                evaluation["attention_implementation"],
                "evaluation.attention_implementation",
            ),
            scorers=tuple(raw_scorers),
        ),
        tasks=tasks,
    )


def _reject_duplicate_json_keys(
    pairs: Sequence[Tuple[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = item
    return result


def load_phase9_config(path: Union[str, Path]) -> Phase9Config:
    """Read and strictly validate a Phase 9 JSON configuration."""

    try:
        raw_text = Path(path).read_text(encoding="utf-8")
    except (OSError, TypeError) as error:
        raise ValueError(
            f"could not read Phase 9 config {path!r}: {error}"
        ) from error
    try:
        parsed = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is not allowed: {item}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"Phase 9 config {str(path)!r} is not valid JSON: {error}"
        ) from error
    return phase9_config_from_dict(parsed)


PHASE9_CONFIG = Phase9Config(
    schema_version=1,
    format=_FORMAT,
    seed=17,
    model_manifest_fingerprint=(
        "e9accebbdcdf821d1bdb64606ee7427d80947237833e4eaf90e199a90b057d88"
    ),
    dataset_manifest_fingerprint=(
        "9f431875d89549c3ae9f58904cbb8f579db095405743200bd249988b58917ebe"
    ),
    selection=Phase9SelectionSpec(
        method="sha256_identifier_order",
        salt="colpali-triton-phase9-v1",
        queries_per_task=50,
        documents_per_task=100,
        positive_policy="include_all_selected_query_positives",
        distractor_policy="sha256_identifier_order",
    ),
    evaluation=Phase9EvaluationSpec(
        k=5,
        warmup_queries=1,
        document_batch_size=1,
        score_document_batch_size=16,
        dtype="bfloat16",
        device="mps",
        attention_implementation="eager",
        scorers=_SCORERS,
    ),
    tasks={
        "docvqa": Phase9TaskSpec(
            source_spec_fingerprint=(
                "98e29159168e03bbabcd4a50a86c8ed414335efb5f236f6ed432896006e3b2ac"
            ),
            selection_fingerprint=(
                "2003c978f51a8591b49f2129e52d9d33c12acd4f9ecef7e87c49fee1d9d1cbaa"
            ),
            image_content_fingerprint=(
                "d199fe34628534923868254578395d628d7508fd07f26dc41da3fd433e769054"
            ),
        ),
        "infovqa": Phase9TaskSpec(
            source_spec_fingerprint=(
                "6b73523943d2187c78bdba80abf19712c72942cf28d15c92ddeebc1a4f27a816"
            ),
            selection_fingerprint=(
                "9542196b7b4305749caa7137b374c878bd105f43e96bd099205d5977cfc8af37"
            ),
            image_content_fingerprint=(
                "3903702a5543afc2534d8c7999812ab19119ca9c3ce02b75f710fd846a1eb2d5"
            ),
        ),
    },
)


__all__ = [
    "PHASE9_CONFIG",
    "Phase9Config",
    "Phase9EvaluationSpec",
    "Phase9SelectionSpec",
    "Phase9TaskSpec",
    "load_phase9_config",
    "phase9_config_from_dict",
]
