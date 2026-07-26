# ColPali Reproduction with Triton MaxSim

This repository is a scoped reproduction of
[ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449).
The completed local milestones cover phases 1–7: paper analysis, project
scaffolding, nested-loop and vectorized PyTorch MaxSim, correctness and gradient
tests, the paper's hardest-in-batch loss, and a controlled synthetic overfit
proof, followed by pinned ViDoRe text loaders, retrieval metrics, and measured
OCR-text baselines, plus a paper-era PaliGemma/ColPali model, processor, and
LoRA integration.

Phase 6 uses a revision-pinned BGE-M3 checkpoint and remote, column-selective
reads of two ViDoRe tasks. Phase 7 pins the public BF16 ViDoRe base and the
original released adapter by full revision and weight digest. AWS resources
and Triton kernels are not used yet.

## Current implementation

ColPali represents a query and a document page as sets of token or patch
embeddings. Their late-interaction score is:

```text
score(q, d) = sum over query tokens i of max over document tokens j
              dot(q[i], d[j])
```

`colpali_triton.maxsim` computes every query-document pairing. It accepts either
unbatched `[tokens, dimension]` embeddings or batched
`[batch, tokens, dimension]` embeddings. Optional Boolean masks exclude padded
query or document positions.

```python
import torch

from colpali_triton import maxsim

queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
documents = torch.tensor(
    [
        [[1.0, 0.0], [0.0, 2.0]],
        [[-1.0, 0.0], [0.0, -2.0]],
    ]
)

scores = maxsim(queries, documents)
assert scores.shape == (1, 2)
```

The scorer deliberately does not L2-normalize its inputs. The model encoder
normalizes projected embeddings once, before indexing or scoring.
The final query-token reduction accumulates in float32 for float16, bfloat16,
and float32 inputs, and in float64 for float64 inputs.

Mask semantics are explicit:

- `True` means a real token or patch; `False` means padding.
- Masked query positions contribute zero.
- Masked document positions cannot win the maximum, including when all valid
  similarities are negative.
- A query with no valid tokens scores `0`.
- A document with no valid tokens scores `-inf` against a nonempty query.

## Training objective

For a square in-batch score matrix, the positive for query `k` is
`scores[k, k]` and its negative is the largest off-diagonal value in row `k`.
The paper's one-way pairwise objective is:

```text
loss = mean_k softplus(hardest_negative[k] - positive[k])
```

It is available as:

```python
from colpali_triton import hardest_in_batch_contrastive_loss

loss = hardest_in_batch_contrastive_loss(scores)
```

This is intentionally not full in-batch InfoNCE, and it is not symmetrized in
the document-to-query direction.

## Synthetic overfit proof

The phase-5 experiment trains independent, randomly initialized query and
document multi-vector tables through L2 normalization, MaxSim, and the real
hardest-negative loss. It uses variable-length masks and no external data.

```bash
.venv/bin/python scripts/overfit_synthetic.py
```

With the committed seed-17 configuration, the recorded run achieved:

- loss `0.9450667500 → 0.0888411775` (90.60% reduction);
- paired top-1 accuracy `2/6 → 6/6`;
- minimum positive-versus-hardest-negative margin `2.037635`;
- nonzero valid query and document gradients;
- exactly zero padding gradients and padding-parameter changes.

The full configuration, measured output, limitations, and environment are in
[the phase-5 report](reports/phase_5_synthetic_overfit.md).

## Phase 6 text baselines

The evaluation subset contains the official 500-page test subsamples of
DocVQA and InfoVQA. The loader combines revision-pinned BEIR queries/qrels with
revision-pinned page-level Tesseract OCR, validates exact row/count contracts,
and checks a committed semantic fingerprint before evaluation.

Two complete-corpus baselines are implemented:

- BM25Okapi formula parity with `rank-bm25==0.2.2`, scored over deterministic
  overlapping word windows and aggregated by maximum chunk score per page.
- Revision-pinned `BAAI/bge-m3` dense embeddings with normalized dot product
  over the same windows and maximum chunk score per page.

The four measured runs produced:

| Task | Baseline | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | BM25 | 0.3246 | 0.3653 | 0.3127 |
| DocVQA | BGE-M3 | 0.3112 | 0.3693 | 0.2929 |
| InfoVQA | BM25 | 0.5743 | 0.6518 | 0.5489 |
| InfoVQA | BGE-M3 | 0.6802 | 0.7504 | 0.6576 |

These are local OCR-text measurements, not ColPali or paper-result
reproductions. Fixed Tesseract word windows also differ from the paper's
Unstructured plus Claude-captioned text pipeline. Full quality, latency,
throughput, memory, index-size, provenance, and truncation details are in
[the Phase 6 baseline report](reports/phase_6_text_baselines.md).

## Phase 7 multimodal integration

The retrieval model uses the final PaliGemma hidden state, a shared biased
`2048 → 128` projection, per-position L2 normalization, and exact-zero masked
positions. Documents use the prompt `Describe the image.` and retain 1,024
image positions plus six prompt positions. Queries use the `Question: ` prefix,
five adjacent `<unused0>` augmentation tokens, and no image positions after
preprocessing.

The paper LoRA topology is reproduced with rank 32, alpha 32, dropout 0.1,
Gaussian initialization, and adapters on all language-model attention/MLP
linear layers plus the retrieval projection. The vision tower and multimodal
projector remain frozen. The released adapter was independently verified to
contain 127 LoRA A/B pairs and 39,292,928 adapter parameters.

```python
import torch
from PIL import Image

from colpali_triton import late_interaction_scores, load_phase7_config
from colpali_triton.checkpoints import load_released_colpali

config = load_phase7_config("configs/colpali_phase7.json")
released = load_released_colpali(
    config,
    cache_dir="artifacts/huggingface_phase7",
    device="mps",
)

page = Image.open("page.png")
document_batch = released.processor.process_documents([page]).to(
    released.device,
    pixel_dtype=released.dtype,
)
query_batch = released.processor.process_queries(
    ["Which total is shown?"]
).to(released.device)

with torch.inference_mode():
    documents = released.model(**document_batch.as_model_inputs())
    queries = released.model(**query_batch.as_model_inputs())
    scores = late_interaction_scores(queries, documents)
```

Loading verifies approximately 5.93 GB of pinned public base and adapter
weights before inference. The gated Google checkpoint in the manifest is a
source reference, not a claim that its current revision is the paper's original
training revision. See [the Phase 7 integration report](reports/phase_7_model_integration.md)
for the exact revisions, contracts, checks, and remaining Phase 8 boundary.

## Local setup

Python 3.9 or newer is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c configs/phase6_macos_arm64_constraints.txt \
  -e ".[test,evaluation]"
python -m pytest
```

The constraints file records the exact Apple-arm64 Phase 6 dependency graph.
Use a clean virtual environment for benchmark measurements. A
`--system-site-packages` environment may be convenient for development, but
its unrelated packages can make `pip check` fail and are recorded as an
environment limitation in the local report.

For the Phase 7 multimodal code and released checkpoint, create a clean
environment with the newer pinned constraints:

```bash
python3 -m venv .venv-phase7
source .venv-phase7/bin/activate
python -m pip install --upgrade pip
python -m pip install -c configs/phase7_macos_arm64_constraints.txt \
  -e ".[test,evaluation,multimodal]"
python -m pip check
python -m pytest
```

Run one measured baseline from the project root:

```bash
.venv/bin/python scripts/evaluate_text_baseline.py \
  --task docvqa \
  --baseline bm25 \
  --output artifacts/phase6/docvqa_bm25.json
```

For BGE-M3 on Apple Silicon, use `--baseline dense --device mps`. The first
dense run downloads the pinned 2.27 GB checkpoint into the ignored
`artifacts/huggingface` cache and verifies its size and SHA-256 before model
loading. The runner never replaces an existing output unless `--overwrite` is
explicitly supplied.

## Repository layout

```text
src/colpali_triton/       scorers, models, processing, loaders, and metrics
tests/                    unit, gradient, integration-contract, and overfit tests
reports/                  paper notes, implementation plan, and recorded results
configs/                  committed experiment settings and thresholds
scripts/                  reproducible experiment entry points
benchmarks/               future CUDA/Triton benchmark harnesses
```

See [the phase 1 implementation plan](reports/phase_1_implementation_plan.md)
for paper-derived requirements, assumptions, and the boundary between reported
paper results and this reproduction's future measurements. The historical
phases 1–4 run remains in [the original local test report](reports/local_test_results.md);
the phase-5 experiment is in [its test report](reports/phase_5_test_results.md);
the Phase 6 baseline suite is in
[its test report](reports/phase_6_test_results.md); and the latest complete
suite is recorded in [the Phase 7 test report](reports/phase_7_test_results.md).
