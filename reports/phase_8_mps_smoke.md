# Phase 8 Released-Checkpoint MPS Smoke Test

## Outcome

Phase 8 passed on July 26, 2026. The pinned public BF16 ColPali base and the
paper adapter loaded on Apple MPS, encoded one deterministic document and one
augmented query, and produced a finite PyTorch MaxSim score.

This is a functional smoke test. It is not a retrieval-quality evaluation and
its single cold timings are not benchmark claims.

## Reproduction

From the project root in the clean Phase 7 environment:

```text
.venv-phase7/bin/python scripts/smoke_colpali_mps.py
```

After the checkpoint is cached, the exact offline command used for the final
record is:

```text
.venv-phase7/bin/python scripts/smoke_colpali_mps.py --local-files-only
```

The runner refuses to replace an existing output unless `--overwrite` is
explicitly passed. It writes the ignored raw record to:

```text
artifacts/phase8/colpali_mps_smoke.json
```

The final raw record is 1:1 identified by SHA-256:

```text
b7be6b3a4913cfad00fb7517ea359361f0f43a5559baae7e205d4c481758563c
```

Its recorded Git revision was
`ecb4eace5b362ee95a004513676c589624090253`, with a clean worktree.

## Verified checkpoints

All three weight files were resolved at full immutable revisions, checked for
exact size, and hashed before model loading.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Runtime base shard 1 | 4,986,817,288 | `8b439a322fa98964517fca8fbf0a1d17e6e60e7e8812bc1c373ad87d654108d5` |
| Runtime base shard 2 | 862,495,528 | `ad075b39b4a24b309028f3d003a0e7cad9e45fa2dbbd267867474e77e10df814` |
| Paper adapter | 78,625,112 | `961b72c2b2a1bebc3e11e7d98cd173d263437aad5ef36e760e9265c184c88d64` |

The runtime base revision was
`vidore/colpaligemma-3b-mix-448-base@6ff0d944ea09c3ead97d2bc57427e3d4f01d192f`.
The adapter revision was
`vidore/colpali@234ecbefa176542348dc5fae4f95c9736858edc2`.

The loaded `PeftModelForFeatureExtraction` contained 2,963,906,416 unique
parameters, all BF16, all on `mps:0`, all frozen, and in evaluation mode. The
count equals the 2,924,613,488-parameter runtime base plus 39,292,928 adapter
parameters. The adapter resolved exactly 127 LoRA targets.

## Input and encoding checks

The generated RGB invoice page was 448 × 448 pixels. Its raw pixel bytes had
SHA-256:

```text
1d066f3c5b86964f78179b3b9403ff52cb4b30defbd75a348f99adab48d9296c
```

The page visibly contains an invoice total of `$42.00`. The query was:

```text
What is the invoice total?
```

Measured tensors:

| Value | Document | Query |
| --- | ---: | ---: |
| Active input positions | 1,030 | 15 |
| Encoding shape | `[1, 1030, 128]` | `[1, 15, 128]` |
| Encoding dtype | BF16 | BF16 |
| Logical embedding payload | 263,680 bytes | 3,840 bytes |
| Minimum active-vector norm | 0.996174 | 0.997864 |
| Maximum active-vector norm | 1.003846 | 1.003069 |
| Maximum absolute norm error | 0.003846 | 0.003069 |
| Nonzero masked values | 0 | 0 |

The vectorized PyTorch MaxSim result was:

```text
shape  [1, 1]
dtype  float32
score  13.28125
finite true
```

The score only proves that the complete retrieval path executed and returned a
valid value. A single synthetic page cannot establish retrieval quality.

## Single-shot timing

Every device timing boundary was synchronized.

| Stage | Seconds |
| --- | ---: |
| Cached artifact hashing and model load | 8.865432 |
| Processing and device transfer | 0.185788 |
| Document encoding | 3.066168 |
| Query encoding | 0.961857 |
| MaxSim scoring | 0.224920 |

These measurements include cold compilation and first-use effects and have no
warmup or repeated-trial distribution. Phase 11 owns rigorous GPU benchmarking.

## Memory

| Snapshot | MPS current bytes | MPS driver bytes |
| --- | ---: | ---: |
| Before load | 0 | 475,136 |
| After load | 6,005,667,584 | 6,491,160,576 |
| After document | 6,007,409,408 | 8,765,931,520 |
| After query | 6,007,417,600 | 8,765,964,288 |
| After score | 6,007,417,856 | 8,765,964,288 |

PyTorch reported a 12,884,918,272-byte MPS recommended maximum. The highest
synchronized snapshot was approximately 46.6% of that limit for current
allocator bytes and 68.0% for driver bytes. Peak process RSS was 666,730,496
bytes after the weights had moved to unified GPU allocation.

These are synchronized snapshots, not a continuous profiler trace, so they
must not be described as an exact transient peak.

## Defect found and fixed

The first real run correctly failed before MPS transfer because
`model.language_model.lm_head.weight` remained on Transformers' low-memory
`meta` device. The outer `ColPali` wrapper did not expose the nested
PaliGemma input/output embedding tie, and the released checkpoint omits the
duplicated tied language-head weight.

The fix delegates input and output embedding access through the outer wrapper,
propagates the nested tied-weight key, and audits every loaded parameter and
buffer for unresolved meta tensors before device transfer. A focused
low-memory save/load regression now proves that the language head is
materialized, tied to the input embedding, and movable. The fix is commit
`ecb4eac`.

## Verification state

Before the final run:

```text
400 passed in 3.83s
python -m pip check: no broken requirements
```

This includes the original MaxSim, loss, synthetic-overfit, dataset,
evaluation, baseline, Phase 7 model/processor/LoRA, Phase 8 runner, and
low-memory tied-weight regression tests.

## Environment

- MacBook Pro `Mac15,6`
- Apple M3 Pro
- 19,327,352,832 bytes physical unified memory
- macOS 15.6.1, arm64
- Python 3.9.6
- PyTorch 2.6.0
- Transformers 4.42.4
- PEFT 0.11.1
- Accelerate 0.31.0
- Safetensors 0.7.0
- Pillow 10.4.0
- eager attention implementation
- MPS built and available; CUDA unavailable

The exact constraints file SHA-256 was:

```text
bec1b743b017fad51802a686533cda28d9b2f5611c14cb260d6e3319c6962b42
```

## Remaining project boundary

Phase 8 establishes a working local multimodal pipeline. It does not satisfy
the remaining completion criteria:

- Phase 9 must evaluate the ColPali late-interaction model and mean-pooled
  ablation on documented real ViDoRe subsets.
- Phase 10 must add a one-command AWS/CUDA environment and benchmark harness.
- Phase 11 must run and validate a real Triton kernel on NVIDIA hardware.
- Phase 12 must consolidate quality, ablation, GPU, limitation, and discrepancy
  evidence into the final report and README.
