from typing import Callable

import pytest
import torch
from torch import Tensor

import colpali_triton.triton_maxsim as subject
from colpali_triton.maxsim import maxsim_vectorized
from colpali_triton.triton_maxsim import (
    DEFAULT_TRITON_MAXSIM_CONFIG,
    TritonMaxSimConfig,
    maxsim_triton,
    triton_is_available,
)


def _enable_mock_execution(
    monkeypatch: pytest.MonkeyPatch,
    launcher: Callable[[Tensor, Tensor, Tensor, Tensor, object], Tensor],
) -> None:
    monkeypatch.setattr(
        subject,
        "_validate_execution",
        lambda queries, documents, config: None,
    )
    monkeypatch.setattr(subject, "_require_triton", lambda: None)
    monkeypatch.setattr(subject, "_launch_maxsim", launcher)


def _reference_launcher(
    queries: Tensor,
    documents: Tensor,
    query_mask: Tensor,
    document_mask: Tensor,
    config: object,
) -> Tensor:
    del config
    return maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )


def test_default_config_is_explicit_and_reproducible() -> None:
    assert DEFAULT_TRITON_MAXSIM_CONFIG == TritonMaxSimConfig(
        block_query_tokens=16,
        block_document_tokens=64,
        block_embedding_dimension=128,
        num_warps=4,
        num_stages=2,
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("block_query_tokens", 8, ValueError, "one of"),
        ("block_document_tokens", 48, ValueError, "one of"),
        ("block_embedding_dimension", 256, ValueError, "one of"),
        ("block_query_tokens", True, TypeError, "integer"),
        ("num_warps", 3, ValueError, "one of"),
        ("num_warps", False, TypeError, "integer"),
        ("num_stages", 0, ValueError, "between"),
        ("num_stages", 2.5, TypeError, "integer"),
    ],
)
def test_invalid_config_is_rejected(
    field: str, value: object, error_type: type, message: str
) -> None:
    arguments = {
        "block_query_tokens": 16,
        "block_document_tokens": 64,
        "block_embedding_dimension": 128,
        "num_warps": 4,
        "num_stages": 2,
    }
    arguments[field] = value

    with pytest.raises(error_type, match=message):
        TritonMaxSimConfig(**arguments)  # type: ignore[arg-type]


def test_availability_probe_always_returns_bool() -> None:
    assert isinstance(triton_is_available(), bool)


def test_missing_optional_dependency_has_focused_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_error = ModuleNotFoundError("No module named 'triton'")
    monkeypatch.setattr(subject, "triton", None)
    monkeypatch.setattr(subject, "tl", None)
    monkeypatch.setattr(subject, "_TRITON_IMPORT_ERROR", import_error)

    with pytest.raises(
        RuntimeError, match="optional Triton dependency"
    ) as error:
        subject._require_triton()

    assert error.value.__cause__ is import_error


def test_cpu_execution_is_rejected_before_compilation() -> None:
    with pytest.raises(RuntimeError, match="CUDA device"):
        maxsim_triton(torch.randn(2, 8), torch.randn(3, 8))


def test_shared_shape_validation_precedes_backend_validation() -> None:
    with pytest.raises(ValueError, match="tokens, dim"):
        maxsim_triton(torch.randn(8), torch.randn(3, 8))


def test_config_type_is_validated() -> None:
    with pytest.raises(TypeError, match="TritonMaxSimConfig"):
        maxsim_triton(
            torch.randn(2, 8),
            torch.randn(3, 8),
            config="fast",  # type: ignore[arg-type]
        )


def test_embedding_tile_must_cover_logical_dimension() -> None:
    with pytest.raises(ValueError, match="at least"):
        subject._validate_execution(
            torch.empty(1, 2, 32, device="meta"),
            torch.empty(1, 3, 32, device="meta"),
            TritonMaxSimConfig(block_embedding_dimension=16),
        )


@pytest.mark.parametrize(
    ("query_shape", "document_shape", "expected_shape"),
    [
        ((5, 8), (7, 8), torch.Size([])),
        ((5, 8), (3, 7, 8), torch.Size([3])),
        ((2, 5, 8), (7, 8), torch.Size([2])),
        ((2, 5, 8), (3, 7, 8), torch.Size([2, 3])),
    ],
)
def test_mocked_kernel_restores_public_batch_shapes(
    monkeypatch: pytest.MonkeyPatch,
    query_shape: tuple,
    document_shape: tuple,
    expected_shape: torch.Size,
) -> None:
    _enable_mock_execution(monkeypatch, _reference_launcher)
    generator = torch.Generator().manual_seed(201)
    queries = torch.randn(query_shape, generator=generator)
    documents = torch.randn(document_shape, generator=generator)

    actual = maxsim_triton(queries, documents)
    expected = maxsim_vectorized(queries, documents)

    assert actual.shape == expected_shape
    torch.testing.assert_close(actual, expected)


def test_mocked_kernel_receives_prepared_masks_and_tuning_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def launcher(
        queries: Tensor,
        documents: Tensor,
        query_mask: Tensor,
        document_mask: Tensor,
        config: TritonMaxSimConfig,
    ) -> Tensor:
        captured.update(
            queries=queries,
            documents=documents,
            query_mask=query_mask,
            document_mask=document_mask,
            config=config,
        )
        return _reference_launcher(
            queries, documents, query_mask, document_mask, config
        )

    _enable_mock_execution(monkeypatch, launcher)
    queries = torch.tensor(
        [[1.0, 0.0], [50.0, 50.0], [0.0, 1.0]]
    )
    documents = torch.tensor(
        [[2.0, 0.0], [100.0, 100.0], [0.0, 3.0]]
    )
    query_mask = torch.tensor([True, False, True])
    document_mask = torch.tensor([True, False, True])
    config = TritonMaxSimConfig(
        block_query_tokens=32,
        block_document_tokens=32,
        block_embedding_dimension=16,
        num_warps=2,
        num_stages=3,
    )

    actual = maxsim_triton(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
        config=config,
    )

    torch.testing.assert_close(actual, torch.tensor(5.0))
    assert captured["queries"].shape == (1, 3, 2)
    assert captured["documents"].shape == (1, 3, 2)
    assert captured["query_mask"].shape == (1, 3)
    assert captured["document_mask"].shape == (1, 3)
    assert captured["config"] is config


def test_autograd_uses_reference_contract_without_forward_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_launcher(*args: object, **kwargs: object) -> Tensor:
        raise AssertionError("forward launcher must not run under autograd")

    _enable_mock_execution(monkeypatch, unexpected_launcher)
    queries = torch.tensor(
        [[1.0, 0.0], [float("nan"), float("nan")]],
        requires_grad=True,
    )
    documents = torch.tensor(
        [[-2.0, 0.0], [float("nan"), float("nan")]],
        requires_grad=True,
    )
    query_mask = torch.tensor([True, False])
    document_mask = torch.tensor([True, False])

    score = maxsim_triton(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )
    score.backward()

    torch.testing.assert_close(score, torch.tensor(-2.0))
    assert queries.grad is not None
    assert documents.grad is not None
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(documents.grad).all()
    torch.testing.assert_close(queries.grad[1], torch.zeros(2))
    torch.testing.assert_close(documents.grad[1], torch.zeros(2))


def test_no_grad_mode_executes_forward_launcher_for_grad_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def launcher(
        queries: Tensor,
        documents: Tensor,
        query_mask: Tensor,
        document_mask: Tensor,
        config: object,
    ) -> Tensor:
        calls.append(True)
        return _reference_launcher(
            queries, documents, query_mask, document_mask, config
        )

    _enable_mock_execution(monkeypatch, launcher)
    queries = torch.randn(2, 4, requires_grad=True)
    documents = torch.randn(3, 4, requires_grad=True)

    with torch.no_grad():
        result = maxsim_triton(queries, documents)

    assert calls == [True]
    assert not result.requires_grad


def _cuda_triton_available() -> bool:
    return torch.cuda.is_available() and triton_is_available()


@pytest.mark.skipif(
    not _cuda_triton_available(),
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float32, 2e-4, 2e-4),
    ],
)
def test_cuda_kernel_matches_vectorized_reference(
    dtype: torch.dtype, rtol: float, atol: float
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    generator = torch.Generator(device="cuda").manual_seed(211)
    queries = torch.randn(
        2, 19, 128, dtype=dtype, device="cuda", generator=generator
    )
    documents = torch.randn(
        3, 73, 128, dtype=dtype, device="cuda", generator=generator
    )
    query_mask = (
        torch.rand((2, 19), device="cuda", generator=generator) > 0.25
    )
    document_mask = (
        torch.rand((3, 73), device="cuda", generator=generator) > 0.25
    )
    query_mask[:, 0] = True
    document_mask[:, 0] = True
    config = TritonMaxSimConfig(
        block_query_tokens=16,
        block_document_tokens=32,
        block_embedding_dimension=128,
        num_warps=4,
        num_stages=2,
    )

    actual = maxsim_triton(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
        config=config,
    )
    expected = maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not _cuda_triton_available(),
    reason="CUDA and Triton are required",
)
def test_cuda_kernel_handles_noncontiguous_inputs_and_empty_masks() -> None:
    generator = torch.Generator(device="cuda").manual_seed(223)
    queries = torch.randn(
        2, 11, 256, device="cuda", dtype=torch.float16, generator=generator
    )[:, :, ::2]
    documents = torch.randn(
        3, 37, 256, device="cuda", dtype=torch.float16, generator=generator
    )[:, :, ::2]
    query_mask = torch.tensor(
        [
            [False] * 11,
            [True, True, False, True, True, False, True, True, True, False, True],
        ],
        device="cuda",
    )
    document_mask = torch.tensor(
        [
            [True] + [False] * 36,
            [False] * 37,
            [True] * 31 + [False] * 6,
        ],
        device="cuda",
    )

    actual = maxsim_triton(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )
    expected = maxsim_vectorized(
        queries,
        documents,
        query_mask=query_mask,
        document_mask=document_mask,
    )

    assert not queries.is_contiguous()
    assert not documents.is_contiguous()
    torch.testing.assert_close(actual[0], torch.zeros(3))
    assert torch.isneginf(actual[1, 1])
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
