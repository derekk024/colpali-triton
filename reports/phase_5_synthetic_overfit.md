# Phase 5: Contrastive Loss and Synthetic Overfit

## Outcome

The paper's hardest-in-batch pairwise objective is implemented and a tiny
controlled multi-vector retrieval problem was overfit successfully. This closes
phase 5 only. It does not demonstrate pretrained-backbone quality or
generalization to real documents.

Canonical command:

```text
.venv/bin/python scripts/overfit_synthetic.py
```

Configuration:

- config: `configs/synthetic_overfit.json`
- seed: `17`
- query-page pairs: `6`
- maximum query vectors: `3`
- maximum document vectors: `5`
- embedding dimension: `8`
- optimizer: Adam
- learning rate: `0.1`
- optimizer updates: `60`
- device and dtype: CPU float32

Valid query lengths alternate between 2 and 3. Valid document lengths cycle
through 3, 4, and 5. The remaining positions are explicitly masked.

## Measured result

Run on 2026-07-25:

```json
{
  "duration_seconds": 4.67341575,
  "final_loss": 0.08884117752313614,
  "final_top1_accuracy": 1.0,
  "initial_document_gradient_norm": 0.702143132686615,
  "initial_loss": 0.9450667500495911,
  "initial_query_gradient_norm": 0.858192503452301,
  "initial_top1_accuracy": 0.3333333432674408,
  "loss_reduction_fraction": 0.905994811987117,
  "maximum_padding_gradient": 0.0,
  "maximum_padding_parameter_change": 0.0,
  "minimum_positive_margin": 2.037635087966919,
  "num_pairs": 6,
  "overfit_passed": true,
  "seed": 17,
  "steps": 60
}
```

The committed acceptance thresholds require at least 85% loss reduction, 100%
paired top-1 accuracy, a minimum positive margin of 1.5, nonzero initial
gradients for both valid query and document parameters, and zero gradient or
parameter drift at padding positions.

## Objective implemented

For the all-pairs score matrix `S` of a batch containing paired examples:

```text
positive[k] = S[k, k]
negative[k] = max over l != k of S[k, l]
loss[k] = softplus(negative[k] - positive[k])
loss = mean(loss[k])
```

This matches Equation 2 of the ColPali paper. It is row-wise, uses one hardest
in-batch document negative per query, has no temperature, and is not a
document-to-query symmetric loss.

The implementation promotes float16 and bfloat16 score matrices to float32 for
the max, subtraction, softplus, and reduction. Float64 remains float64 for
high-precision tests.

## What this experiment proves

- Gradients flow through vector normalization, all-pairs MaxSim, the maximum
  reductions, and the pairwise contrastive loss.
- The objective can memorize the identity pairing in a controlled six-example
  dataset.
- The learned diagonal score is larger than every in-batch negative for all six
  queries.
- Masked vector parameters receive no gradient and do not change.
- The fixed seed and configuration produce deterministic measured statistics.

It does not prove that a PaliGemma encoder can be trained on ViDoRe, that the
paper's retrieval metrics are reproduced, or that the method generalizes.

## Environment

- MacBook Pro with Apple M3 Pro and 18 GB unified memory
- macOS 15.6.1 (`24G90`), arm64
- Python 3.9.6
- PyTorch 2.6.0
- colpali-triton 0.2.0

No model weights or datasets were downloaded.

