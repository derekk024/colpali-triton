"""Strict ColPali text-and-image preprocessing contracts.

The paper-era ColPali recipe uses a PaliGemma processor for both documents and
queries.  Queries are passed through that multimodal processor with a dummy
image, then the image-token prefix and pixel tensor are removed.  This module
keeps that slightly unusual boundary explicit and validates the assumptions on
which it depends.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from colpali_triton.modeling import ColPaliBatch, EncodingKind


DOCUMENT_PROMPT = "Describe the image."
QUERY_PREFIX = "Question: "
QUERY_AUGMENTATION_TOKEN = "<unused0>"
QUERY_AUGMENTATION_COUNT = 5
IMAGE_TOKEN = "<image>"
DEFAULT_MAX_LENGTH = 50
EXPECTED_DOCUMENT_TEXT_TOKENS = 6


def _require_max_length(max_length: int) -> int:
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return max_length


def _require_batch(
    values: Sequence[Any], name: str
) -> List[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    batch = list(values)
    if not batch:
        raise ValueError(f"{name} must not be empty")
    return batch


def _token_id(tokenizer: object, token: str, name: str) -> int:
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(converter):
        raise TypeError(
            "processor.tokenizer must define convert_tokens_to_ids"
        )
    token_id = converter(token)
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{name} token ID must be an integer")
    if token_id < 0:
        raise ValueError(f"{name} token ID must be nonnegative")
    return token_id


def _as_boolean_mask(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError("processor attention_mask must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError("processor attention_mask must use strided layout")
    if value.ndim != 2:
        raise ValueError(
            "processor attention_mask must have shape [batch, tokens]"
        )
    if value.dtype == torch.bool:
        return value.contiguous()
    if value.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(
            "processor attention_mask must have a Boolean or integer dtype"
        )
    if not bool(((value == 0) | (value == 1)).all().item()):
        raise ValueError("processor attention_mask must contain only 0 and 1")
    return value.to(dtype=torch.bool).contiguous()


class ColPaliProcessor:
    """Adapt an injected PaliGemma-compatible processor for retrieval.

    No model or tokenizer is loaded here.  ``processor`` supplies the
    PaliGemma-compatible call interface and ``mock_image_factory`` creates the
    dummy image required by its query path.
    """

    def __init__(
        self,
        processor: object,
        mock_image_factory: Callable[[], object],
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not callable(mock_image_factory):
            raise TypeError("mock_image_factory must be callable")

        image_seq_length = getattr(processor, "image_seq_length", None)
        if (
            isinstance(image_seq_length, bool)
            or not isinstance(image_seq_length, int)
        ):
            raise TypeError("processor.image_seq_length must be an integer")
        if image_seq_length <= 0:
            raise ValueError("processor.image_seq_length must be positive")

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("processor must expose a tokenizer")
        try:
            tokenizer.padding_side = "right"
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "processor.tokenizer must support right padding"
            ) from error
        if getattr(tokenizer, "padding_side", None) != "right":
            raise ValueError("processor.tokenizer must use right padding")

        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = _token_id(
                tokenizer, IMAGE_TOKEN, "image"
            )
        elif isinstance(image_token_id, bool) or not isinstance(
            image_token_id, int
        ):
            raise TypeError("processor.image_token_id must be an integer")
        elif image_token_id < 0:
            raise ValueError("processor.image_token_id must be nonnegative")

        augmentation_token_id = _token_id(
            tokenizer, QUERY_AUGMENTATION_TOKEN, "query augmentation"
        )
        if image_token_id == augmentation_token_id:
            raise ValueError(
                "image and query augmentation tokens must have distinct IDs"
            )

        self._processor = processor
        self._tokenizer = tokenizer
        self._mock_image_factory = mock_image_factory
        self._image_seq_length = image_seq_length
        self._image_token_id = image_token_id
        self._augmentation_token_id = augmentation_token_id

    @property
    def image_seq_length(self) -> int:
        """Number of leading image IDs emitted by the base processor."""

        return self._image_seq_length

    def process_documents(
        self,
        images: Sequence[object],
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> ColPaliBatch:
        """Process RGB document pages with the exact ColPali prompt."""

        max_length = _require_max_length(max_length)
        image_batch = _require_batch(images, "images")
        rgb_images = [
            self._to_rgb(image, f"images[{index}]")
            for index, image in enumerate(image_batch)
        ]
        texts = [DOCUMENT_PROMPT] * len(rgb_images)
        output = self._call_processor(
            texts=texts,
            images=rgb_images,
            max_length=max_length,
        )
        input_ids, attention_mask, pixel_values = self._extract_output(
            output, len(rgb_images)
        )
        self._validate_image_prefix(input_ids, attention_mask)
        self._validate_right_padding(attention_mask)

        text_counts = (
            attention_mask.sum(dim=1) - self._image_seq_length
        )
        if not bool(
            (text_counts == EXPECTED_DOCUMENT_TEXT_TOKENS).all().item()
        ):
            raise ValueError(
                "document processor output must contain exactly "
                f"{EXPECTED_DOCUMENT_TEXT_TOKENS} active prompt tokens after "
                "the image prefix"
            )
        self._validate_no_active_image_ids(
            input_ids[:, self._image_seq_length :],
            attention_mask[:, self._image_seq_length :],
        )
        if pixel_values is None:
            raise ValueError(
                "document processor output must contain pixel_values"
            )

        return ColPaliBatch(
            input_ids=input_ids.contiguous(),
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            kind=EncodingKind.DOCUMENT,
        )

    def process_queries(
        self,
        queries: Sequence[str],
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> ColPaliBatch:
        """Process augmented text queries and remove dummy-image features."""

        max_length = _require_max_length(max_length)
        query_batch = _require_batch(queries, "queries")
        texts = [
            self._query_text(query, index)
            for index, query in enumerate(query_batch)
        ]

        dummy_image = self._to_rgb(
            self._mock_image_factory(), "mock image"
        )
        output = self._call_processor(
            texts=texts,
            images=[dummy_image] * len(texts),
            max_length=max_length,
        )
        input_ids, attention_mask, pixel_values = self._extract_output(
            output, len(texts)
        )
        if pixel_values is None:
            raise ValueError(
                "query processor output must contain dummy pixel_values"
            )
        self._validate_image_prefix(input_ids, attention_mask)
        self._validate_right_padding(attention_mask)

        query_ids = input_ids[:, self._image_seq_length :].contiguous()
        query_mask = attention_mask[
            :, self._image_seq_length :
        ].contiguous()
        if query_ids.shape[1] == 0:
            raise ValueError(
                "query processor output has no tokens after the image prefix"
            )
        if not bool(query_mask.any(dim=1).all().item()):
            raise ValueError(
                "each query must retain an active token after prefix removal"
            )
        self._validate_no_active_image_ids(query_ids, query_mask)
        self._validate_augmentation(query_ids, query_mask)
        self._validate_right_padding(query_mask)

        active_lengths = query_mask.sum(dim=1)
        if bool((active_lengths > max_length).any().item()):
            longest = int(active_lengths.max().item())
            raise ValueError(
                "query token length exceeds max_length after removing the "
                f"image prefix: {longest} > {max_length}"
            )

        return ColPaliBatch(
            input_ids=query_ids,
            attention_mask=query_mask,
            pixel_values=None,
            kind=EncodingKind.QUERY,
        )

    def _query_text(self, query: object, index: int) -> str:
        if not isinstance(query, str):
            raise TypeError(f"queries[{index}] must be a string")
        if not query.strip():
            raise ValueError(f"queries[{index}] must not be blank")
        for reserved_token in (
            QUERY_AUGMENTATION_TOKEN,
            IMAGE_TOKEN,
        ):
            if reserved_token in query:
                raise ValueError(
                    f"queries[{index}] must not contain reserved token "
                    f"{reserved_token}"
                )
        return (
            f"{QUERY_PREFIX}{query}"
            f"{QUERY_AUGMENTATION_TOKEN * QUERY_AUGMENTATION_COUNT}"
        )

    def _to_rgb(self, image: object, name: str) -> object:
        converter = getattr(image, "convert", None)
        if not callable(converter):
            raise TypeError(f"{name} must provide convert('RGB')")
        converted = converter("RGB")
        if converted is None:
            raise ValueError(f"{name}.convert('RGB') returned None")
        return converted

    def _call_processor(
        self,
        *,
        texts: List[str],
        images: List[object],
        max_length: int,
    ) -> Mapping[str, object]:
        self._force_right_padding()
        output = self._processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            # PaliGemmaProcessor adds image_seq_length internally.  Passing the
            # text-side limit directly avoids counting that prefix twice.
            max_length=max_length,
        )
        self._force_right_padding()
        if not isinstance(output, Mapping):
            raise TypeError("processor output must be a mapping")
        return output

    def _force_right_padding(self) -> None:
        try:
            self._tokenizer.padding_side = "right"
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "processor.tokenizer must support right padding"
            ) from error
        if getattr(self._tokenizer, "padding_side", None) != "right":
            raise ValueError("processor.tokenizer must use right padding")

    def _extract_output(
        self,
        output: Mapping[str, object],
        expected_batch_size: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if "input_ids" not in output:
            raise ValueError("processor output is missing input_ids")
        if "attention_mask" not in output:
            raise ValueError("processor output is missing attention_mask")

        input_ids = output["input_ids"]
        if not isinstance(input_ids, Tensor):
            raise TypeError("processor input_ids must be a torch.Tensor")
        if input_ids.layout != torch.strided:
            raise TypeError("processor input_ids must use strided layout")
        if input_ids.ndim != 2:
            raise ValueError(
                "processor input_ids must have shape [batch, tokens]"
            )
        if input_ids.dtype != torch.int64:
            raise TypeError("processor input_ids must have dtype torch.int64")
        attention_mask = _as_boolean_mask(output["attention_mask"])
        if tuple(input_ids.shape) != tuple(attention_mask.shape):
            raise ValueError(
                "processor input_ids and attention_mask must have equal shapes"
            )
        if input_ids.shape[0] != expected_batch_size:
            raise ValueError(
                "processor output batch size does not match its inputs"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "processor output token dimension must be nonzero"
            )
        if input_ids.device != attention_mask.device:
            raise ValueError(
                "processor input_ids and attention_mask must share a device"
            )

        pixel_values = output.get("pixel_values")
        if pixel_values is not None and not isinstance(pixel_values, Tensor):
            raise TypeError("processor pixel_values must be a torch.Tensor")
        return input_ids, attention_mask, pixel_values

    def _validate_image_prefix(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> None:
        if input_ids.shape[1] <= self._image_seq_length:
            raise ValueError(
                "processor output must contain tokens after the image prefix"
            )
        prefix_ids = input_ids[:, : self._image_seq_length]
        prefix_mask = attention_mask[:, : self._image_seq_length]
        if not bool(prefix_mask.all().item()):
            raise ValueError(
                "the image prefix must be active and precede any padding"
            )
        if not bool((prefix_ids == self._image_token_id).all().item()):
            raise ValueError(
                "processor output does not begin with exactly "
                "image_seq_length image token IDs"
            )

    def _validate_no_active_image_ids(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> None:
        active_image_ids = (
            (input_ids == self._image_token_id) & attention_mask
        )
        if bool(active_image_ids.any().item()):
            raise ValueError(
                "active image token IDs remain after the expected prefix"
            )

    def _validate_augmentation(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> None:
        counts = (
            (input_ids == self._augmentation_token_id) & attention_mask
        ).sum(dim=1)
        if not bool((counts == QUERY_AUGMENTATION_COUNT).all().item()):
            raise ValueError(
                "each query must retain exactly "
                f"{QUERY_AUGMENTATION_COUNT} active "
                f"{QUERY_AUGMENTATION_TOKEN} tokens"
            )

    @staticmethod
    def _validate_right_padding(attention_mask: Tensor) -> None:
        if attention_mask.shape[1] < 2:
            return
        false_then_true = (
            ~attention_mask[:, :-1] & attention_mask[:, 1:]
        )
        if bool(false_then_true.any().item()):
            raise ValueError(
                "processor attention_mask must use contiguous right padding"
            )
