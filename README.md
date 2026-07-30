# ColPali Reproduction with Triton MaxSim

[![CI](https://github.com/derekk024/colpali-triton/actions/workflows/ci.yml/badge.svg)](https://github.com/derekk024/colpali-triton/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)
[![GPU: NVIDIA L4](https://img.shields.io/badge/GPU-NVIDIA%20L4-76B900.svg)](reports/phase_12_final_report.md)

This repository is an independent, scoped reproduction of
[ColPali](https://arxiv.org/abs/2407.01449) with an optimized Triton
implementation of the MaxSim late-interaction scorer. It contains:

- PyTorch reference implementations of MaxSim and the paper's hardest
  in-batch negative loss;
- loaders and evaluation code for fixed subsets of DocVQA and InfoVQA;
- integration with the released ColPali adapter and its PaliGemma backbone;
- a fused Triton MaxSim forward kernel; and
- correctness, tuning, benchmarking, and profiling tools for NVIDIA GPUs.

The work does not reproduce the paper's full training run or full ViDoRe
evaluation. The model experiments use released weights, and the retrieval
results below use explicitly defined subsets. See
[Limitations](#limitations) and the
[final report](reports/phase_12_final_report.md) for the exact scope.

## Results

The Triton kernel was measured against the vectorized PyTorch implementation
on one NVIDIA L4 (`g6.xlarge`). All GPU measurements were produced from source
commit [`0f9543e`](https://github.com/derekk024/colpali-triton/commit/0f9543e30e49fc30de8c476b2767e1c29073602b).

| Measurement | Result |
| --- | ---: |
| Geometric-mean latency speedup | **2.27x** |
| Benchmark cases won by Triton | **12 / 12** |
| Speedup range | **1.57x–4.31x** |
| Required CUDA parity cases | **5 / 5** |
| Local test suite | **664 passed, 5 expected CUDA skips** |
| L4 test suite | **276 passed, 1 expected MPS skip** |

![Triton MaxSim speedup over PyTorch on an NVIDIA L4](assets/benchmark-speedup.svg)

Latency was measured with CUDA events after warmup. The memory figures in the
report are changes in `torch.cuda.max_memory_allocated`, not total process,
driver, or model memory. The largest observed incremental allocation was
41,798,144 bytes for the PyTorch scorer and 1,536 bytes for the Triton scorer.

The full measurements, environment, and methodology are available in:

- [Final report](reports/phase_12_final_report.md)
- [Machine-readable GPU results](reports/phase_11_gpu_results.json)
- [Benchmark protocol](benchmarks/README.md)
- [AWS L4 reproduction guide](infra/aws/README.md)

## MaxSim

ColPali represents a query and a document page as sets of embeddings. For
query embeddings \(q_i\) and document embeddings \(d_j\), the score is:

```text
score(q, d) = sum_i max_j dot(q_i, d_j)
```

`maxsim` is the device-independent PyTorch implementation. `maxsim_triton`
executes the Triton kernel on supported CUDA inputs; `triton_is_available`
can be used to check whether the optional runtime is installed.

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

The scorer accepts unbatched `[tokens, dimension]` or batched
`[batch, tokens, dimension]` tensors. Boolean masks use `True` for valid
positions and `False` for padding. Masked query positions contribute zero;
masked document positions cannot win the maximum. Inputs are not normalized
inside the scorer because the model encoder normalizes embeddings before
indexing or scoring.

## Training objective

For a square in-batch score matrix, query `k` is paired with document `k`. The
implemented objective compares that positive with the largest off-diagonal
score in the same row:

```text
loss = mean_k softplus(hardest_negative[k] - positive[k])
```

```python
from colpali_triton import hardest_in_batch_contrastive_loss

loss = hardest_in_batch_contrastive_loss(scores)
```

This is the paper's one-way pairwise objective, not full in-batch InfoNCE and
not a document-to-query symmetric loss. A synthetic six-pair optimization test
and its recorded output are described in the
[synthetic overfit report](reports/phase_5_synthetic_overfit.md).

## Retrieval evaluation

The multimodal evaluation fixes 50 queries and 100 pages from each task using
a committed identifier-order rule. Every selected query is scored against all
100 pages for its task. Both aggregation methods below consume the same
released-model embeddings.

| Task | Scorer | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | ColPali MaxSim | 0.6720 | 0.8200 | 0.6217 |
| DocVQA | Normalized mean pooling | 0.2805 | 0.3200 | 0.2667 |
| InfoVQA | ColPali MaxSim | 0.8679 | 0.9000 | 0.8567 |
| InfoVQA | Normalized mean pooling | 0.6904 | 0.8400 | 0.6413 |

These are subset measurements, not the paper's full-task results. Selection,
qrels, fingerprints, timing, and index-size details are in the
[multimodal evaluation report](reports/phase_9_colpali_subset.md).

The repository also includes BM25 and revision-pinned BGE-M3 baselines over
the official 500-page DocVQA and InfoVQA test subsamples. Those runs use
Tesseract OCR text and are documented separately in the
[text baseline report](reports/phase_6_text_baselines.md); their metrics are
not directly compared with the smaller multimodal subsets above.

## Triton benchmark

The benchmark covers six tensor shapes in FP16 and BF16. A tuning pass evaluates
12 fixed launch configurations on four representative shapes before running
the full 12-case comparison. The selected configuration was
`tq16_td128_w4_s2`.

| Dtype | Latency speedup range | Largest absolute error |
| --- | ---: | ---: |
| FP16 | 1.60x–4.22x | 0.001039 |
| BF16 | 1.57x–4.31x | 0.009487 |

The GPU test gate compares Triton with PyTorch on five required cases,
including masks and negative similarities. The final run also captured an
Nsight Compute report for the selected configuration. Raw profiler and
benchmark outputs are excluded from Git because of their size; the committed
result record contains their SHA-256 digests.

Run the local preflight without a GPU:

```bash
python benchmarks/benchmark_maxsim.py --preflight
python benchmarks/tune_triton_maxsim.py --preflight
```

Run the benchmark on a compatible CUDA host:

```bash
python benchmarks/tune_triton_maxsim.py
python benchmarks/benchmark_maxsim.py
```

The container definition pins PyTorch 2.6.0, CUDA 12.4, Triton 3.2.0, and the
base image digest. The [AWS guide](infra/aws/README.md) documents the optional
EC2 lifecycle wrapper used for the recorded L4 run.

## Installation and tests

Python 3.9 or newer is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,evaluation,multimodal]"
python -m pytest
```

The macOS constraint files under `configs/` record the dependency sets used
for the text-baseline and released-model experiments. Model and dataset
downloads are stored under the ignored `artifacts/` directory.

To run one text baseline:

```bash
python scripts/evaluate_text_baseline.py \
  --task docvqa \
  --baseline bm25 \
  --output artifacts/phase6/docvqa_bm25.json
```

The first BGE-M3 or ColPali run downloads the corresponding pinned external
weights. The loaders verify the configured revisions and expected file
digests before inference.

## Repository layout

```text
src/colpali_triton/   scoring, model integration, loaders, and metrics
tests/                unit, gradient, contract, and workflow tests
benchmarks/           Triton tuning, correctness, timing, and profiling tools
configs/              experiment settings, revisions, and thresholds
scripts/              evaluation and validation entry points
docker/               pinned CUDA benchmark environment
infra/aws/            optional L4 lifecycle and remote-run tooling
reports/              methods, recorded results, and limitations
```

## Limitations

- The released ColPali adapter is evaluated; the original model is not trained
  from scratch in this repository.
- Multimodal retrieval metrics use 50 queries and 100 pages per task and
  should not be interpreted as full ViDoRe benchmark results.
- GPU performance was measured on one NVIDIA L4 software/hardware environment.
  Results may differ on other GPUs, CUDA versions, or Triton versions.
- The reported allocator deltas are useful for comparing these scorer calls
  but are not end-to-end application memory measurements.
- The Triton implementation covers forward inference. Training continues to
  use the differentiable PyTorch scorer.

Additional discrepancies and threats to validity are listed in the
[final report](reports/phase_12_final_report.md).

## Upstream work and licensing

This project is not the official
[`illuin-tech/colpali`](https://github.com/illuin-tech/colpali) package and
does not redistribute its source, datasets, or model weights.

Repository code is available under the [MIT License](LICENSE). External
datasets and checkpoints retain their own licenses and terms. In particular,
the referenced ColPali adapter is MIT-licensed while the PaliGemma backbone is
subject to the Gemma terms linked from its model card.
