# Phase 6 Test Results

## Final verification

The final post-hardening suite passed on July 25, 2026 in
America/Los_Angeles (UTC date July 26).

Command:

```text
.venv/bin/python -m pytest -vv
```

Exact result:

```text
collected 245 items
245 passed in 14.50s
```

There were 0 failures, 0 skips, 0 xfails, and 0 xpasses. Before the full run,
the focused phase-6 runner and module selection also passed all 93 of its
tests.

Test runner:

- platform: `darwin`
- Python 3.9.6
- pytest 8.4.2
- pluggy 1.6.0

## Phase-6 coverage

The new tests exercise five boundaries.

### Retrieval metrics and ranking

- hand-computed binary `nDCG@k`, `Recall@k`, and `MRR@k`;
- macro averaging and multi-positive relevance sets;
- complete-ranking invariants;
- score-descending, document-ID-descending tie handling;
- finite floating score, shape, dtype, ID, and cutoff validation; and
- rejection of mismatched query sets, incomplete corpus permutations, and
  unordered relevance inputs.

### ViDoRe data and manifest integrity

- column-selective reads from local Parquet fixtures;
- preservation of multi-positive qrels and stable string IDs;
- exact document, query, qrel, and OCR row counts;
- row-order OCR-to-qrel join guards for both query and corpus identity;
- rejection of unknown documents, query drift, non-positive qrels, NaN, and
  infinity;
- semantic content fingerprints, including a same-query OCR-row swap case;
- exact pinned 40-character dataset revisions;
- strict manifest schema and canonical built-in/config equivalence;
- duplicate JSON-key rejection;
- revision-bearing benchmark URLs; and
- frozen, validated public data objects.

### OCR-text baselines

- NFKC/casefold Unicode tokenization;
- deterministic overlapping whitespace chunks and invalid chunk settings;
- a hand-computed BM25Okapi example;
- differential page scores against `rank-bm25==0.2.2`, including empty chunks,
  common negative-IDF terms, out-of-vocabulary queries, unequal chunk lengths,
  and chunk-to-page maximum aggregation;
- dense L2 normalization and maximum chunk-to-page cosine similarity;
- blank page and blank query behavior;
- logical index accounting;
- malformed and non-finite encoder-output rejection; and
- lazy, pinned Sentence Transformers loading and offline inspection.

### Complete-corpus evaluation

- every query scored against every page;
- multi-positive qrels and canonical tie ordering;
- warm-up calls excluded from query counts and timing totals;
- separate score and postprocessing/ranking times;
- score-only and end-to-end query-document-pair throughput;
- p50/p95 latency and logical payload reporting; and
- invalid indexes, duplicate queries, out-of-corpus qrels, malformed score
  vectors, and invalid evaluation settings.

### Measured-runner safety and provenance

- duplicate configuration-key rejection;
- output-existence preflight before expensive configuration or model work;
- atomic output creation that refuses replacement unless explicitly allowed;
  and
- strict token-length summary validation and over-limit counting.

All phase 1–5 tests remain in the same full-suite command, including nested and
vectorized MaxSim agreement, masks, gradients, MPS behavior, the contrastive
loss, and deterministic synthetic overfitting.

## Real-data checks outside the unit suite

The following integration outcomes were observed from complete real-data
runs:

- DocVQA loaded 500 pages, 451 queries, and 500 qrels; its semantic content
  fingerprint was
  `3f0df908eb592425c4bc2ab68883736f6ca2f34ac9f67383a09468a600b7e580`.
- InfoVQA loaded 500 pages, 494 queries, and 500 qrels; its semantic content
  fingerprint was
  `fc39812e6d6350c6fc8894f0f37978739684a240da4fda102fd48551b7ac434d`.
- BM25 and BGE-M3 completed full-corpus evaluation on both tasks and produced
  four finite measured result artifacts.
- Hardened schema-2 BM25 reruns reproduced the original quality metrics, chunk
  counts, and logical index sizes exactly on both tasks.
- The cached pinned BGE-M3 weight file was verified at 2,271,145,830 bytes with
  SHA-256
  `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`;
  the loaded pipeline reported 1,024 dimensions, CLS pooling, float32, a final
  normalization module, and a 512-token effective maximum.
- A no-truncation tokenizer audit found two document chunks above 512 tokens
  in each task and no over-limit queries.

The exact quality, timing, index, memory, revision, and hardware records are in
[the phase-6 baseline report](phase_6_text_baselines.md). These integrations
complement the final unit suite; they are not counted among its 245 tests.

## Draft-time environment

- MacBook Pro `Mac15,6`
- Apple M3 Pro with 14-core GPU
- 18 GB unified memory
- macOS 15.6.1, arm64
- Python 3.9.6
- PyTorch 2.6.0
- pytest 8.4.2
- pluggy 1.6.0
- NumPy 1.24.3
- fsspec 2025.2.0
- PyArrow 21.0.0
- Sentence Transformers 3.0.1
- Transformers 4.41.1
- project 0.3.0, editable installation
- MPS built and available; CUDA unavailable

The virtual environment uses system-site packages. At draft time,
`pip check` reported one unrelated ambient conflict:
`alpaca-trade-api 3.2.0` requires `websockets<11`, while
`websockets 15.0.1` was installed. Neither package participates in the
retrieval implementation or its tests, but this means the environment is not
fully dependency-isolated.
