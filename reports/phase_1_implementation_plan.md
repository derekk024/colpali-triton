# Phase 1: Paper Notes and Implementation Plan

> Status update (2026-07-25): the phase-5 loss and synthetic overfit milestone
> described as deferred below is now complete. See
> [the phase-5 report](phase_5_synthetic_overfit.md). The remaining roadmap is
> unchanged.

## Scope and source boundary

This milestone implements only phases 1–4 from the project specification. It
does not download PaliGemma, ViDoRe, or any other model or dataset; configure
AWS; implement Triton; train LoRA adapters; or report reproduction metrics.

Primary source:

- Manuel Faysse et al., [“ColPali: Efficient Document Retrieval with Vision
  Language Models,” arXiv:2407.01449v6](https://arxiv.org/abs/2407.01449),
  published at ICLR 2025.

Paper-era implementation reference:

- [`illuin-tech/colpali` tag `v0.1.1`](https://github.com/illuin-tech/colpali/tree/v0.1.1)
  and checkpoint `vidore/colpali`, based on
  `google/paligemma-3b-mix-448`. Current library behavior and later checkpoints
  are not interchangeable with the paper configuration.

The numbers below are paper claims, not results reproduced by this repository.
Future reports must keep that distinction explicit.

## Paper-derived design

### Retrieval unit and embeddings

The paper treats each document page as one retrievable document. PaliGemma-3B
provides contextualized language-model outputs for both text tokens and image
patch tokens. A learned projection maps every output vector to `D=128`, yielding
query embeddings `E_q` with shape `[N_q, D]` and page embeddings `E_d` with
shape `[N_d, D]`.

The paper-era implementation L2-normalizes every projected vector, as required
by the project specification. Normalization will live in the future encoder,
not inside the scoring function, so indexed documents are normalized exactly
once and the MaxSim oracle remains a literal implementation of the equation.
At 448-pixel resolution the faithful page representation contains 1,024 image
patch positions plus six prompt-token positions from `Describe the image`, for
1,030 vectors rather than a patch-only 1,024.

### Late interaction

For query `q` and page `d`, Equation 1 is:

```text
LI(q, d) = sum_i max_j dot(E_q[i], E_d[j])
```

This is asymmetric: every valid query token selects its strongest page patch,
then those token-level maxima are summed. The paper does not define padding
semantics because its equation uses variable-length sets. This implementation
makes the padded equivalent explicit:

- query padding contributes zero;
- document padding is replaced by negative infinity before the maximum;
- an empty query scores zero;
- a fully masked document scores negative infinity against a nonempty query.

Using negative infinity is important. Replacing padded document scores with
zero would be incorrect whenever every real dot product is negative.

### Training objective for the next milestone

For a batch of positive query-page pairs `(q_k, d_k)`, the paper uses the
positive score `s_k+ = LI(q_k, d_k)` and the hardest other page in the batch
`s_k- = max_(l != k) LI(q_k, d_l)`. Equation 2 is the numerically stable
pairwise loss:

```text
L = mean_k softplus(s_k- - s_k+)
```

This loss is deliberately deferred to phase 5. The all-pairs score matrix
implemented now is the primitive it will consume.

### Query augmentation and optimization facts for later phases

- Five `<unused0>` tokens are appended to each query as learned soft
  expansion/reweighting positions. The paper-era text form is
  `Question: {query}` followed by those five valid scored tokens.
- The reported training set contains 118,695 English query-page pairs: 63%
  academic sources and 37% synthetically queried web PDF pages.
- The paper reports one epoch, bfloat16, LoRA rank and alpha of 32 on the
  language transformer layers, a trained random projection, paged AdamW 8-bit,
  learning rate `5e-5`, 2.5% warmup, and total batch size 32 on eight GPUs.
- The reference configuration freezes the vision encoder and adapts the
  language attention/MLP projections plus the retrieval projection.
- A 2% validation subset was used for hyperparameter selection, with PDF-level
  train/test separation to guard against leakage.

These are reference settings rather than promises that the scoped reproduction
can match the original eight-GPU run.

## Paper evaluation facts to preserve

ViDoRe evaluates page retrieval across ten tasks spanning scanned documents,
infographics, financial reports, scientific figures, tables, and practical
domain corpora in English and French. The paper uses `nDCG@5` as its primary
metric and also discusses recall, MRR, query latency, and page-indexing
throughput.

Table 2 reports an average `nDCG@5` of 81.3 for the reference 448-resolution
ColPali model across ten tasks. The paper reports 65.5 for its
Unstructured-plus-OCR BM25 baseline and 66.1 for the corresponding BGE-M3
baseline. These values are context only. Later experiments will select and
document a fixed two- or three-task subset and record only actually measured
results.

The paper attributes ColPali's gains to the retrieval-oriented data, the
language model's contextualization of visual patches, and multi-vector late
interaction. Its single-vector BiPali ablation averages 58.8 versus 81.3 for
ColPali in Table 2, motivating the required mean-pooling ablation later.

The project-required `Recall@5` is a deliberate extension: the paper discusses
recall broadly, while its headline appendix recall table reports `Recall@1`.
The reported float16 index size of 257.5 KiB per DocVQA page corresponds to
`1,030 × 128 × 2` bytes.

## Phases 1–4 implementation contract

### Public API

The package exposes:

- `maxsim_nested`: explicit query/document/token loops and `torch.dot`, used as
  the readable correctness oracle;
- `maxsim_vectorized`: broadcast matrix multiplication, masked document
  reduction, and masked query summation;
- `maxsim`: the default alias for the vectorized reference.

Inputs may be `[N, D]` or `[B, N, D]`. Two batched inputs produce an all-pairs
matrix `[B_q, B_d]`; an omitted batch dimension is removed from the result.
Masks are Boolean and match the leading token dimensions.

The vectorized implementation intentionally materializes
`[B_q, B_d, N_q, N_d]`. That makes it a clear PyTorch reference. The future
Triton implementation will tile over embeddings and document patches while
fusing the max and sum reductions to avoid this allocation.

Masked embeddings are zeroed before the matrix multiplication and document
padding is still replaced by negative infinity before reduction. Doing both
prevents masked NaN/Inf padding from contaminating either forward values or
backward gradients. Low-precision inputs use float32 for the final sum over
query tokens; float64 inputs retain float64 accumulation.

### Correctness strategy

Tests must cover:

1. hand-computed scores and all-pairs batching;
2. independent query and document masks, including negative-only matches;
3. fully masked queries and documents;
4. randomized nested/vectorized parity in float16, bfloat16, float32, and
   float64;
5. unbatched, mixed, and batched shape contracts;
6. noncontiguous tensors and masked padding containing NaNs;
7. autograd parity, finite-difference gradcheck, and zero gradients at masked
   positions;
8. validation errors for shape, dtype, mask, and device mismatches;
9. an MPS value-and-backward smoke test when Apple MPS is available.

All tests use synthetic tensors and deterministic seeds.

## Reproducibility ambiguities to resolve before real evaluation

- The paper-era left-padding query path was later corrected upstream. This
  reproduction will use explicit masks and right padding rather than reproduce
  a padding-alignment bug.
- Published training-size prose and component counts are not fully consistent;
  the paper reports 118,695 examples. The exact dataset revision and observed
  row count must be recorded before training.
- Hosted ViDoRe datasets have changed since the paper (for example, published
  and current TabFQuAD sizes differ). Every evaluated dataset revision must be
  pinned before claiming comparability.
- The [2025 independent study](https://arxiv.org/abs/2505.07730) uses a later
  three-epoch, larger-batch checkpoint. Its results are useful corroboration
  but are not a reproduction target for the paper's one-epoch
  `vidore/colpali` configuration.

## Next milestones, intentionally deferred

1. Implement the hardest-in-batch contrastive loss and overfit a controlled
   synthetic retrieval set.
2. Add metrics, fixed ViDoRe subset manifests, OCR/BM25 and dense baselines.
3. Integrate the pretrained PaliGemma backbone, projection, normalization,
   query augmentation, and LoRA without reimplementing the VLM.
4. Run and record real local reproduction measurements and ablations.
5. Only then add the AWS environment, Triton kernel, correctness comparison,
   profiler-driven tuning, and honest L4 benchmarks.
