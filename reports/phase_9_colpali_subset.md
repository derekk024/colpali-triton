# Phase 9 Scoped ColPali ViDoRe Evaluation

## Outcome

The released paper-era ColPali checkpoint completed real image retrieval on
fixed subsets of DocVQA and InfoVQA. Late-interaction MaxSim outperformed an
L2-normalized masked-mean ablation on both tasks.

This phase satisfies the local scoped-reproduction milestone. It does not
claim to reproduce the paper's full ten-task average or full-task values.

## Subset fixed before scoring

The committed configuration selects each task independently:

```text
queries per task    50
documents per task  100
selection           SHA-256(identifier, kind, committed salt)
positive policy     include every positive for every selected query
distractors         next documents in SHA-256 identifier order
evaluation          every selected query against all 100 selected pages
```

The salt is `colpali-triton-phase9-v1`. No image content, query text, model
score, or relevance rank participates in selection.

| Task | Selection fingerprint | Image-content fingerprint |
| --- | --- | --- |
| DocVQA | `2003c978f51a8591b49f2129e52d9d33c12acd4f9ecef7e87c49fee1d9d1cbaa` | `d199fe34628534923868254578395d628d7508fd07f26dc41da3fd433e769054` |
| InfoVQA | `9542196b7b4305749caa7137b374c878bd105f43e96bd099205d5977cfc8af37` | `3903702a5543afc2534d8c7999812ab19119ca9c3ce02b75f710fd846a1eb2d5` |

The complete DocVQA and InfoVQA BEIR corpus, query, and qrel Parquet files were
resolved at the Phase 6 full revisions and verified by exact byte length and
SHA-256 before selection. The selected DocVQA images contain 60,237,333
encoded bytes and are all PNG. The selected InfoVQA images contain 39,251,568
encoded bytes and are all JPEG.

## Model and scoring

Both scorers use exactly the same released model and encoded vectors:

- BF16 `vidore/colpaligemma-3b-mix-448-base` at
  `6ff0d944ea09c3ead97d2bc57427e3d4f01d192f`;
- `vidore/colpali` adapter at
  `234ecbefa176542348dc5fae4f95c9736858edc2`;
- 2,963,906,416 frozen BF16 parameters on Apple MPS;
- 127 LoRA targets and 39,292,928 adapter parameters;
- 1,030 document vectors per page, each 128-dimensional; and
- right-padded, augmented variable-length query vectors.

`late_interaction_maxsim` computes the sum, over query positions, of each
position's maximum document-vector dot product. The ablation averages only
active vectors into one float32 query vector and one float32 page vector, then
L2-normalizes each mean and uses a dot product.

Query encoding is performed once per measured query and shared by both
scorers. The reported end-to-end latency for each scorer includes that shared
encoding time plus its own scoring and ranking time.

## Retrieval quality

| Task | Scorer | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | Late interaction | 0.672027 | 0.820000 | 0.621667 |
| DocVQA | Mean pooling | 0.280474 | 0.320000 | 0.266667 |
| InfoVQA | Late interaction | 0.867856 | 0.900000 | 0.856667 |
| InfoVQA | Mean pooling | 0.690403 | 0.840000 | 0.641333 |

Measured late-interaction improvements over mean pooling:

| Task | Δ nDCG@5 | Δ Recall@5 | Δ MRR@5 |
| --- | ---: | ---: | ---: |
| DocVQA | +0.391552 | +0.500000 | +0.355000 |
| InfoVQA | +0.177453 | +0.060000 | +0.215333 |

The result supports the paper's qualitative late-interaction claim on both
fixed subsets. The much larger DocVQA gap also shows that a single-vector
summary can discard page-level matching detail even when it uses the same
model output.

Score-matrix fingerprints:

| Task | Late interaction | Mean pooling |
| --- | --- | --- |
| DocVQA | `acdfb98f9cb78c5e84d1d4ad689d1c1e345b70e0e1ba122976365406ef3e935c` | `5418abf8a680248f8056895d104f1428342cf69e0bb2764e6f6a98d99337f8db` |
| InfoVQA | `5bfb7c874bc6bfb162c4987f4a627d65c861c2ad77c236de848aef1076ced668` | `3bb75a8f5fc6055893b0d5466adbfbbcd7372164e3f446e0cf8d737d1d97107a` |

## Indexing

| Task | Seconds | Pages/s | Page p50 | Page p95 |
| --- | ---: | ---: | ---: | ---: |
| DocVQA | 192.151 | 0.5204 | 1,897.46 ms | 1,961.64 ms |
| InfoVQA | 190.153 | 0.5259 | 1,879.76 ms | 2,015.68 ms |

Logical index payloads:

| Representation | Total for 100 pages | Bytes/page |
| --- | ---: | ---: |
| BF16 multi-vector embeddings | 26,368,000 | 263,680 |
| Boolean position masks | 103,000 | 1,030 |
| Multi-vector plus mask | 26,471,000 | 264,710 |
| Float32 normalized mean vectors | 51,200 | 512 |

The embedding-only multi-vector payload is exactly 257.5 KiB per page, matching
`1,030 × 128 × 2` bytes and the paper-derived accounting. The explicit mask
adds 1,030 bytes per page. The complete logical multi-vector payload is about
517 times the mean-vector payload.

## Query performance

### DocVQA

| Scorer | Score seconds | Score pairs/s | End-to-end pairs/s | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Late interaction | 0.7645 | 6,540 | 530 | 112.73 ms | 549.98 ms |
| Mean pooling | 0.2052 | 24,363 | 564 | 108.88 ms | 476.80 ms |

Shared query encoding took 8.6531 seconds over 50 measured queries.

### InfoVQA

| Scorer | Score seconds | Score pairs/s | End-to-end pairs/s | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Late interaction | 0.8806 | 5,678 | 450 | 146.42 ms | 377.23 ms |
| Mean pooling | 0.2767 | 18,068 | 476 | 141.85 ms | 360.85 ms |

Shared query encoding took 10.2230 seconds over 50 measured queries.

Each run used one warmup query. MPS was synchronized at model, encoding, and
scoring timing boundaries. These local figures characterize this scoped
PyTorch path; they are not NVIDIA or Triton benchmarks.

## Memory

| Task | Highest synchronized current bytes | Highest synchronized driver bytes | Peak process RSS |
| --- | ---: | ---: | ---: |
| DocVQA | 6,040,276,736 | 8,891,744,256 | 1,822,277,632 |
| InfoVQA | 6,040,276,736 | 8,929,509,376 | 2,149,367,808 |

PyTorch reported a 12,884,918,272-byte recommended MPS maximum. These are
synchronized snapshots rather than continuous profiler peaks. InfoVQA's larger
CPU RSS reflects decoding much larger source JPEG pages one at a time before
the standard processor resizes them to 448 pixels.

## Baseline boundary

Phase 6 measured two genuine OCR-text baselines over each complete 500-page
test subsample:

| Task | Full-task baseline | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | BM25 | 0.3246 | 0.3653 | 0.3127 |
| DocVQA | BGE-M3 | 0.3112 | 0.3693 | 0.2929 |
| InfoVQA | BM25 | 0.5743 | 0.6518 | 0.5489 |
| InfoVQA | BGE-M3 | 0.6802 | 0.7504 | 0.6576 |

Those remain meaningful independent baselines, but they use 500 pages and all
task queries while Phase 9 uses a precommitted 100-page/50-query subset.
Therefore this report does not subtract their values from the Phase 9 numbers
or present them as a controlled head-to-head result.

## Raw records and provenance

| Task | Raw result SHA-256 |
| --- | --- |
| DocVQA | `b201d2fe37eb1a02058aa178b119aca4451d991bff38479acdc98720764895c6` |
| InfoVQA | `4421e7694a9b7290c50980022ec4994552894f0703f7b37672e6faed378eaae2` |

The ignored records are:

```text
artifacts/phase9/docvqa_colpali_subset.json
artifacts/phase9/infovqa_colpali_subset.json
```

Both records identify clean Git revision
`5f97101216ec66207d9039e4f1b28f6de46e7306`, contain all selected image
digests, query text, qrels, top-five rankings, source hashes, model artifacts,
configuration fingerprints, package versions, and measured values.

The harness was committed only after:

```text
437 passed in 3.81s
python -m pip check: no broken requirements
```

## Limitations

- Results cover fixed 20%-corpus subsets, not the complete tasks or ten-task
  ViDoRe average.
- Smaller candidate pools make quality values easier than full-task values;
  they must not be compared numerically to the paper's full-task table.
- The released adapter is evaluated; this project did not repeat the original
  eight-GPU LoRA training run.
- Mean pooling is the required single-vector ablation, not the paper's
  separately trained BiPali checkpoint.
- Timings use Apple MPS and the vectorized PyTorch scorer. Phase 11 owns
  NVIDIA/Triton correctness and benchmark claims.
- A frozen-backbone-versus-LoRA training ablation remains computationally
  infeasible locally and is not fabricated.
