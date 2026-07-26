# Phase 7 ColPali Model Integration

## Outcome

Phase 7 integrates the paper-era ColPali retrieval model contract without
reimplementing PaliGemma. The package now provides:

- a PaliGemma-backed retrieval model with the released checkpoint's state-key
  layout;
- a shared biased `2048 → 128` projection and masked per-token L2
  normalization;
- strict document and query preprocessing, including query augmentation;
- the exact paper LoRA target regex and trainability audits;
- frozen-backbone and inference modes;
- revision-pinned, byte-verifying base and adapter loading; and
- mask-bound multi-vector encodings that feed the existing tested MaxSim
  scorer.

The large released base is deliberately left for the Phase 8 end-to-end MPS
smoke run. Phase 7 validates the implementation with a real PaliGemma processor,
a real released adapter, miniature PaliGemma models, and network-free unit
tests.

## Pinned upstream artifacts

| Role | Repository | Revision | Verified weight payload |
| --- | --- | --- | ---: |
| Public BF16 runtime base | `vidore/colpaligemma-3b-mix-448-base` | `6ff0d944ea09c3ead97d2bc57427e3d4f01d192f` | 5,849,312,816 bytes |
| Paper adapter, initial upload | `vidore/colpali` | `234ecbefa176542348dc5fae4f95c9736858edc2` | 78,625,112 bytes |
| Gated source reference | `google/paligemma-3b-mix-448` | `ead2d9a35598cb89119af004f5d023b311d1c4a1` | 11,697,486,320 bytes |

Every listed weight has an expected SHA-256 digest in
`configs/colpali_phase7.json`. The loader verifies both size and digest before
loading the public runtime path. The current gated Google revision is recorded
as a source reference only; it is not asserted to be the paper's original
training revision.

The manifest's canonical SHA-256 fingerprint is:

```text
e9accebbdcdf821d1bdb64606ee7427d80947237833e4eaf90e199a90b057d88
```

## Architecture and processing contract

- PaliGemma image size: 448 pixels.
- Vision patch size: 14 pixels.
- Image positions per page: 1,024.
- Language hidden size: 2,048.
- Retrieval dimension: 128.
- Projection bias: enabled.
- Normalization: L2 over the last dimension.
- Padding representation: exactly zero.
- Document prompt: `Describe the image.`
- Query format: `Question: {query}` followed by five adjacent `<unused0>`
  tokens.
- Text-side maximum length: 50.
- Padding side: forced to the right.

Queries follow the paper-era dummy-image route through
`PaliGemmaProcessor`; the 1,024 image IDs and dummy pixels are then removed.
Right padding is enforced explicitly because left-padded query batches were a
known paper-code failure mode.

Transformers 4.42.4 adds `image_seq_length` to its tokenizer-side
`max_length`. The wrapper therefore passes the text-side limit directly and
disables silent truncation. It rejects a query if the active post-prefix length
exceeds the requested limit.

## LoRA contract

The PEFT configuration is:

```text
r=32
lora_alpha=32
lora_dropout=0.1
init_lora_weights=gaussian
bias=none
task_type=FEATURE_EXTRACTION
```

The exact target expression selects `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` in the language model, plus
`custom_text_proj`. It never selects the vision tower or multimodal projector.
For the released 18-layer architecture this resolves to 127 linear modules.

The base projection weight and bias remain frozen in the paper LoRA regime;
the projection is adapted through its LoRA pair. Fully training only the
projection is exposed separately as the frozen-backbone ablation.

## Real integration evidence

The revision-pinned public processor produced:

```text
image_seq_length       1024
document tensor shape  [1, 1030]
document active tokens 1030
query tensor shape     [1, 14]
query active tokens    14
padding side           right
image token ID         257152
<unused0> token ID     7
```

The pinned adapter weight was downloaded and hashed:

```text
size                   78,625,112 bytes
SHA-256                961b72c2b2a1bebc3e11e7d98cd173d263437aad5ef36e760e9265c184c88d64
tensor count           254
LoRA A tensors         127
LoRA B tensors         127
adapter parameters     39,292,928
```

The public base index contains both `custom_text_proj.weight` and
`custom_text_proj.bias`, 605 total weight entries, a 2,048-wide text model,
448/14 vision geometry, and the state prefix expected by the implementation.
These checks inspect revision-pinned metadata; the two large base shards are
byte-verified during Phase 8.

## Deviations and limits

- This phase implements and audits LoRA but does not repeat the paper's full
  eight-GPU training run.
- The public BF16 ViDoRe base is used as the reproducible runtime base because
  the direct Google checkpoint is manually gated and its current revision is
  not proven to be the paper-training revision.
- The released adapter is used for local inference. New LoRA training and the
  frozen-backbone comparison remain later evaluation work.
- Miniature PaliGemma tests prove state layout, forward behavior, PEFT
  insertion, masking, normalization, serialization, and gradients. A real
  released-base forward pass is the separate Phase 8 acceptance gate.

## Environment

- MacBook Pro `Mac15,6`
- Apple M3 Pro
- 18 GiB unified memory
- macOS 15.6.1, arm64
- Python 3.9.6
- PyTorch 2.6.0
- Transformers 4.42.4
- PEFT 0.11.1
- project 0.4.0
- MPS built and available; CUDA unavailable
- clean dependency check: passed

The exact clean package graph is recorded in
`configs/phase7_macos_arm64_constraints.txt`.
