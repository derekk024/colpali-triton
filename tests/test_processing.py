from __future__ import annotations

from typing import Dict, List, Optional

import pytest
import torch

from colpali_triton.modeling import EncodingKind
from colpali_triton.processing import (
    DOCUMENT_PROMPT,
    QUERY_AUGMENTATION_COUNT,
    QUERY_AUGMENTATION_TOKEN,
    QUERY_PREFIX,
    ColPaliProcessor,
)


IMAGE_ID = 99
AUGMENTATION_ID = 7
PAD_ID = 0


class FakeTokenizer:
    def __init__(self) -> None:
        self.padding_side = "left"

    def convert_tokens_to_ids(self, token: str) -> int:
        return {
            "<image>": IMAGE_ID,
            QUERY_AUGMENTATION_TOKEN: AUGMENTATION_ID,
        }[token]


class ImmutablePaddingTokenizer(FakeTokenizer):
    @property
    def padding_side(self) -> str:
        return "left"

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        pass


class FakeImage:
    def __init__(self, name: str) -> None:
        self.name = name
        self.convert_calls: List[str] = []

    def convert(self, mode: str) -> "FakeImage":
        self.convert_calls.append(mode)
        return FakeImage(f"{self.name}:{mode}")


class FakePaliGemmaProcessor:
    image_seq_length = 4
    image_token_id = IMAGE_ID

    def __init__(
        self,
        *,
        tokenizer: Optional[FakeTokenizer] = None,
        fault: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer or FakeTokenizer()
        self.fault = fault
        self.calls: List[Dict[str, object]] = []

    def __call__(
        self,
        *,
        text: List[str],
        images: List[object],
        return_tensors: str,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        self.calls.append(
            {
                "text": text,
                "images": images,
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        rows = [
            [IMAGE_ID] * self.image_seq_length + self._encode(value)
            for value in text
        ]

        if self.fault == "residual_image":
            rows[0][self.image_seq_length + 1] = IMAGE_ID
        if self.fault == "four_augmentations":
            augmentation_index = rows[0].index(AUGMENTATION_ID)
            rows[0][augmentation_index] = 42

        width = max(len(row) for row in rows)
        padded_rows: List[List[int]] = []
        masks: List[List[int]] = []
        for row in rows:
            missing = width - len(row)
            if self.fault == "left_padding" or (
                self.tokenizer.padding_side == "left"
            ):
                padded_rows.append([PAD_ID] * missing + row)
                masks.append([0] * missing + [1] * len(row))
            else:
                padded_rows.append(row + [PAD_ID] * missing)
                masks.append([1] * len(row) + [0] * missing)

        input_ids = torch.tensor(padded_rows, dtype=torch.int64)
        attention_mask = torch.tensor(masks, dtype=torch.int64)
        if self.fault == "holey_mask":
            attention_mask[0, self.image_seq_length + 1] = 0
        if self.fault == "nonbinary_mask":
            attention_mask[0, 0] = 2
        if self.fault == "wrong_ids_dtype":
            input_ids = input_ids.to(torch.int32)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": torch.zeros(
                len(text), 3, 2, 2, dtype=torch.float32
            ),
        }
        if self.fault == "missing_pixels":
            del output["pixel_values"]
        if self.fault == "missing_ids":
            del output["input_ids"]
        return output

    @staticmethod
    def _encode(text: str) -> List[int]:
        if text == DOCUMENT_PROMPT:
            # Pinned PaliGemma tokenization:
            # <bos> Describe ▁the ▁image . \n
            return [2, 20, 21, 22, 23, 108]

        assert text.startswith(QUERY_PREFIX)
        expected_suffix = (
            QUERY_AUGMENTATION_TOKEN * QUERY_AUGMENTATION_COUNT
        )
        assert text.endswith(expected_suffix)
        query = text[
            len(QUERY_PREFIX) : -len(expected_suffix)
        ]
        query_word_ids = [
            30 + index for index, _ in enumerate(query.split())
        ]
        return (
            [2, 24, 25]
            + query_word_ids
            + [AUGMENTATION_ID] * QUERY_AUGMENTATION_COUNT
            + [108]
        )


class DummyFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.image = FakeImage("dummy")

    def __call__(self) -> FakeImage:
        self.calls += 1
        return self.image


def make_subject(
    *,
    fault: Optional[str] = None,
    tokenizer: Optional[FakeTokenizer] = None,
) -> tuple[ColPaliProcessor, FakePaliGemmaProcessor, DummyFactory]:
    base = FakePaliGemmaProcessor(tokenizer=tokenizer, fault=fault)
    factory = DummyFactory()
    return ColPaliProcessor(base, factory), base, factory


def test_documents_use_exact_prompt_rgb_images_and_direct_max_length() -> None:
    subject, base, factory = make_subject()
    images = [FakeImage("one"), FakeImage("two")]

    batch = subject.process_documents(images, max_length=17)

    assert batch.kind is EncodingKind.DOCUMENT
    assert batch.pixel_values is not None
    assert batch.pixel_values.shape == (2, 3, 2, 2)
    assert batch.attention_mask.dtype is torch.bool
    assert batch.input_ids.shape == (2, base.image_seq_length + 6)
    assert base.calls[0]["text"] == [DOCUMENT_PROMPT, DOCUMENT_PROMPT]
    assert base.calls[0]["max_length"] == 17
    assert base.calls[0]["truncation"] is False
    assert base.calls[0]["padding"] == "longest"
    assert all(
        isinstance(image, FakeImage) and image.name.endswith(":RGB")
        for image in base.calls[0]["images"]
    )
    assert [image.convert_calls for image in images] == [["RGB"], ["RGB"]]
    assert factory.calls == 0


def test_queries_use_exact_augmentation_and_remove_only_image_prefix() -> None:
    subject, base, factory = make_subject()

    batch = subject.process_queries(
        ["short", "a query with several words"], max_length=20
    )

    suffix = QUERY_AUGMENTATION_TOKEN * QUERY_AUGMENTATION_COUNT
    assert base.calls[0]["text"] == [
        f"{QUERY_PREFIX}short{suffix}",
        f"{QUERY_PREFIX}a query with several words{suffix}",
    ]
    assert base.calls[0]["max_length"] == 20
    assert base.calls[0]["truncation"] is False
    assert factory.calls == 1
    assert factory.image.convert_calls == ["RGB"]
    assert len({id(image) for image in base.calls[0]["images"]}) == 1

    assert batch.kind is EncodingKind.QUERY
    assert batch.pixel_values is None
    assert batch.attention_mask.dtype is torch.bool
    assert not bool(
        (
            (batch.input_ids == IMAGE_ID)
            & batch.attention_mask
        ).any()
    )
    augmentation_counts = (
        (batch.input_ids == AUGMENTATION_ID) & batch.attention_mask
    ).sum(dim=1)
    assert augmentation_counts.tolist() == [5, 5]
    assert batch.input_ids[0, 0].item() == 2
    assert batch.attention_mask[0].tolist()[-4:] == [
        False,
        False,
        False,
        False,
    ]


def test_processor_is_forced_back_to_right_padding_before_each_call() -> None:
    subject, base, _ = make_subject()
    base.tokenizer.padding_side = "left"

    batch = subject.process_queries(["short", "two words"])

    assert base.tokenizer.padding_side == "right"
    assert batch.attention_mask[0, -1].item() is False
    assert batch.attention_mask[1, -1].item() is True


@pytest.mark.parametrize("bad_query", ["", " ", "\n\t"])
def test_blank_queries_are_rejected_before_processor_call(
    bad_query: str,
) -> None:
    subject, base, factory = make_subject()

    with pytest.raises(ValueError, match="must not be blank"):
        subject.process_queries([bad_query])

    assert base.calls == []
    assert factory.calls == 0


@pytest.mark.parametrize(
    "bad_query",
    [f"contains {QUERY_AUGMENTATION_TOKEN}", "contains <image>"],
)
def test_reserved_tokens_in_queries_are_rejected(bad_query: str) -> None:
    subject, base, _ = make_subject()

    with pytest.raises(ValueError, match="reserved token"):
        subject.process_queries([bad_query])

    assert base.calls == []


def test_nonstring_query_and_empty_batches_are_rejected() -> None:
    subject, _, _ = make_subject()

    with pytest.raises(TypeError, match=r"queries\[0\]"):
        subject.process_queries([123])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="queries must not be empty"):
        subject.process_queries([])
    with pytest.raises(ValueError, match="images must not be empty"):
        subject.process_documents([])


def test_overlong_query_is_rejected_without_silent_truncation() -> None:
    subject, base, _ = make_subject()

    with pytest.raises(ValueError, match="exceeds max_length"):
        subject.process_queries(["one two three four"], max_length=10)

    assert base.calls[0]["max_length"] == 10
    assert base.calls[0]["truncation"] is False


@pytest.mark.parametrize("max_length", [0, -1])
def test_nonpositive_max_length_is_rejected(max_length: int) -> None:
    subject, base, _ = make_subject()

    with pytest.raises(ValueError, match="positive"):
        subject.process_queries(["valid"], max_length=max_length)

    assert base.calls == []


def test_noninteger_max_length_is_rejected() -> None:
    subject, _, _ = make_subject()

    with pytest.raises(TypeError, match="integer"):
        subject.process_queries(["valid"], max_length=True)


def test_tokenizer_that_cannot_use_right_padding_is_rejected() -> None:
    base = FakePaliGemmaProcessor(tokenizer=ImmutablePaddingTokenizer())

    with pytest.raises(ValueError, match="right padding"):
        ColPaliProcessor(base, DummyFactory())


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("left_padding", "image prefix"),
        ("residual_image", "active image token IDs remain"),
        ("four_augmentations", "exactly 5 active"),
        ("holey_mask", "contiguous right padding"),
        ("nonbinary_mask", "only 0 and 1"),
        ("missing_pixels", "dummy pixel_values"),
        ("missing_ids", "missing input_ids"),
        ("wrong_ids_dtype", "dtype torch.int64"),
    ],
)
def test_malformed_query_processor_outputs_are_rejected(
    fault: str, message: str
) -> None:
    subject, _, _ = make_subject(fault=fault)

    with pytest.raises((TypeError, ValueError), match=message):
        subject.process_queries(["short", "two words"])


def test_document_prompt_tokenization_is_validated() -> None:
    subject, base, _ = make_subject()
    original_encode = base._encode

    def malformed_document_encoding(text: str) -> List[int]:
        encoded = original_encode(text)
        if text == DOCUMENT_PROMPT:
            return encoded[:-1]
        return encoded

    base._encode = malformed_document_encoding  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="exactly 6 active prompt tokens"):
        subject.process_documents([FakeImage("page")])
