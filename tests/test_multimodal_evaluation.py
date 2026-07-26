from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import nn

from colpali_triton.modeling import (
    ColPaliBatch,
    EncodingKind,
    MultiVectorEncoding,
)
from colpali_triton.multimodal_evaluation import (
    encode_document_index,
    evaluate_document_index,
    masked_mean_pool,
)
from colpali_triton.vidore import TextQuery
from colpali_triton.vidore_images import (
    ImageDocument,
    MultimodalRetrievalDataset,
    VerifiedDatasetArtifact,
)


def _image_bytes(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color=(value, 0, 0)).save(
        output, format="PNG"
    )
    return output.getvalue()


class FakeProcessor:
    def process_documents(self, images: list) -> ColPaliBatch:
        image = images[0].convert("RGB")
        identifier = int(image.getpixel((0, 0))[0])
        return ColPaliBatch(
            input_ids=torch.tensor(
                [[identifier, identifier, 0]], dtype=torch.int64
            ),
            attention_mask=torch.tensor([[True, True, False]]),
            kind=EncodingKind.DOCUMENT,
            pixel_values=torch.zeros((1, 3, 4, 4)),
        )

    def process_queries(self, queries: list) -> ColPaliBatch:
        identifier = int(queries[0].split("-")[-1])
        return ColPaliBatch(
            input_ids=torch.tensor(
                [[identifier, identifier, 0]], dtype=torch.int64
            ),
            attention_mask=torch.tensor([[True, True, False]]),
            kind=EncodingKind.QUERY,
        )


class RoleAwareFakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> MultiVectorEncoding:
        kind = (
            EncodingKind.DOCUMENT
            if pixel_values is not None
            else EncodingKind.QUERY
        )
        values = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        for token in range(input_ids.shape[1]):
            if attention_mask[0, token]:
                values[0, token, int(input_ids[0, token].item())] = 1.0
        return MultiVectorEncoding(
            embeddings=values,
            mask=attention_mask,
            kind=kind,
        )


def _dataset(tmp_path: Path) -> MultimodalRetrievalDataset:
    documents = []
    for identifier in (1, 2, 3):
        content = _image_bytes(identifier)
        documents.append(
            ImageDocument(
                document_id=str(identifier),
                encoded_image_bytes=content,
                encoded_image_sha256=sha256(content).hexdigest(),
                source_format="PNG",
                source_size=(4, 4),
            )
        )
    artifact_path = tmp_path / "fixture.parquet"
    artifact_path.write_bytes(b"fixture")
    artifact = VerifiedDatasetArtifact(
        repository="owner/fixture",
        revision="a" * 40,
        path="corpus/test.parquet",
        local_path=artifact_path,
        size_bytes=7,
        sha256=sha256(b"fixture").hexdigest(),
    )
    return MultimodalRetrievalDataset(
        name="fixture",
        documents=tuple(documents),
        queries=(
            TextQuery("q1", "query-1", frozenset({"1"})),
            TextQuery("q2", "query-2", frozenset({"2"})),
        ),
        artifacts=(artifact,),
        selection_fingerprint="1" * 64,
        image_content_fingerprint="2" * 64,
    )


def test_masked_mean_pool_uses_only_active_vectors() -> None:
    embeddings = torch.zeros((1, 3, 128))
    embeddings[0, 0, 0] = 1.0
    embeddings[0, 1, 1] = 1.0
    embeddings[0, 2, 2] = 100.0
    encoding = MultiVectorEncoding(
        embeddings=embeddings.masked_fill(
            ~torch.tensor([[True, True, False]]).unsqueeze(-1), 0.0
        ),
        mask=torch.tensor([[True, True, False]]),
        kind=EncodingKind.QUERY,
    )

    pooled = masked_mean_pool(encoding)

    expected = torch.zeros((1, 128))
    expected[0, 0] = 2**-0.5
    expected[0, 1] = 2**-0.5
    torch.testing.assert_close(pooled, expected)


def test_masked_mean_pool_rejects_zero_mean() -> None:
    embeddings = torch.zeros((1, 2, 128))
    embeddings[0, 0, 0] = 1.0
    embeddings[0, 1, 0] = -1.0
    encoding = MultiVectorEncoding(
        embeddings=embeddings,
        mask=torch.ones((1, 2), dtype=torch.bool),
        kind=EncodingKind.QUERY,
    )

    with pytest.raises(RuntimeError, match="nonzero norms"):
        masked_mean_pool(encoding)


def test_fake_end_to_end_evaluation_scores_both_methods(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    model = RoleAwareFakeModel()
    processor = FakeProcessor()
    index = encode_document_index(
        model,
        processor,
        dataset,
        device=torch.device("cpu"),
        pixel_dtype=torch.float32,
    )

    result = evaluate_document_index(
        model,
        processor,
        dataset,
        index,
        k=2,
        warmup_count=1,
        score_document_batch_size=2,
    )

    assert index.embeddings.shape == (3, 3, 128)
    assert index.multi_vector_payload_bytes == 3 * 3 * 128 * 4 + 3 * 3
    assert index.mean_vector_payload_bytes == 3 * 128 * 4
    assert result.late_interaction.ndcg_at_k == 1.0
    assert result.late_interaction.recall_at_k == 1.0
    assert result.mean_pooling.ndcg_at_k == 1.0
    assert result.mean_pooling.recall_at_k == 1.0
    assert result.late_interaction.top_k_rankings["q1"][0] == "1"
    assert result.mean_pooling.top_k_rankings["q2"][0] == "2"
    assert len(result.late_interaction.score_matrix_sha256) == 64
    assert result.indexing["pages_per_second"] > 0.0


def test_index_rejects_non_unit_document_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 1"):
        encode_document_index(
            RoleAwareFakeModel(),
            FakeProcessor(),
            _dataset(tmp_path),
            device=torch.device("cpu"),
            pixel_dtype=torch.float32,
            document_batch_size=2,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"k": 0}, "k must"),
        ({"warmup_count": -1}, "warmup_count"),
        ({"score_document_batch_size": 0}, "score_document_batch_size"),
    ),
)
def test_evaluation_rejects_invalid_settings(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    dataset = _dataset(tmp_path)
    model = RoleAwareFakeModel()
    processor = FakeProcessor()
    index = encode_document_index(
        model,
        processor,
        dataset,
        device=torch.device("cpu"),
        pixel_dtype=torch.float32,
    )
    arguments = {
        "k": 2,
        "warmup_count": 0,
        "score_document_batch_size": 2,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        evaluate_document_index(
            model,
            processor,
            dataset,
            index,
            **arguments,
        )
