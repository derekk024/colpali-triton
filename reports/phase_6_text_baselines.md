# Phase 6: ViDoRe OCR-Text Retrieval Baselines

## Outcome

Two real, complete-corpus text retrieval baselines were evaluated on the
500-page DocVQA and InfoVQA ViDoRe test subsets:

- OCR + BM25 with maximum chunk-to-page aggregation; and
- OCR + BGE-M3 dense retrieval with maximum chunk-to-page aggregation.

All four entries below are measured local results. They are neither numbers
copied from the [ColPali paper](https://arxiv.org/abs/2407.01449) nor claimed
reproductions of a paper table. This phase establishes the text-only reference
points needed before adding the multimodal ColPali-style model in phase 7.

## Comparability boundary

The experiment deliberately uses pinned, page-level Tesseract OCR artifacts
from the [ViDoRe page-OCR artifact
collection](https://huggingface.co/collections/vidore/vidore-page-ocr-artifact).
Each page is split into deterministic 192-whitespace-word windows with a
32-word overlap. A page receives the maximum score of any of its chunks.

That is a scoped deviation from ViDoRe's official
[chunk-OCR baseline
collection](https://huggingface.co/collections/vidore/vidore-chunk-ocr-baseline),
which uses chunks produced with Unstructured and Claude-generated image
descriptions. The OCR source, chunk boundaries, and available visual
descriptions therefore differ. The local quality numbers must not be placed
next to the paper's text baselines as if they were like-for-like reproductions.

These runs also do not use page images, PaliGemma, ColPali multi-vectors,
LoRA, or MaxSim. They are phase-6 text baselines only.

The paper's headline retrieval metric is nDCG@5. Recall@5 is required by this
project specification and MRR@5 is included as an additional diagnostic;
neither is presented here as a value reproduced from the paper.

## Frozen inputs

Task membership and revisions follow the pinned
[MTEB ViDoRe task
definitions](https://github.com/embeddings-benchmark/mteb/blob/c67d000b43c841df834a53f92ccf6f2e3be77cfb/mteb/tasks/retrieval/eng/vidore_bench_retrieval.py).
The evaluated split is `test` for both tasks.

| Task | BEIR source and revision | Tesseract OCR source and revision | Documents | Queries | Qrels | Empty OCR pages |
|---|---|---|---:|---:|---:|---:|
| DocVQA | [`mteb/docvqa_test_subsampled_beir@e78cd130055fcb8d69bb0ff4115c4712a157f3d3`](https://huggingface.co/datasets/mteb/docvqa_test_subsampled_beir/tree/e78cd130055fcb8d69bb0ff4115c4712a157f3d3) | [`vidore/docvqa_test_subsampled_tesseract@9a183a7237d678eaaf68dc0f640a42f46016802f`](https://huggingface.co/datasets/vidore/docvqa_test_subsampled_tesseract/tree/9a183a7237d678eaaf68dc0f640a42f46016802f) | 500 | 451 | 500 | 4 |
| InfoVQA | [`mteb/infovqa_test_subsampled_beir@74d396242cc281c021eeb38d0ee5f6f30afc1fad`](https://huggingface.co/datasets/mteb/infovqa_test_subsampled_beir/tree/74d396242cc281c021eeb38d0ee5f6f30afc1fad) | [`vidore/infovqa_test_subsampled_tesseract@7bd56b41a40e767bba7b1467c5520c337246291c`](https://huggingface.co/datasets/vidore/infovqa_test_subsampled_tesseract/tree/7bd56b41a40e767bba7b1467c5520c337246291c) | 500 | 494 | 500 | 3 |

The loader-produced semantic content fingerprints recorded by every run were:

- DocVQA:
  `3f0df908eb592425c4bc2ab68883736f6ca2f34ac9f67383a09468a600b7e580`
- InfoVQA:
  `fc39812e6d6350c6fc8894f0f37978739684a240da4fda102fd48551b7ac434d`

The manifest also records expected Hugging Face LFS object sizes and SHA-256
values for the BEIR and OCR Parquet files. Those values identify the intended
remote objects, but the artifact-time loader used column-selective remote
Parquet reads and did not hash every complete remote file byte-for-byte.
Accordingly, they are expected source metadata, not a claim that all full
Parquet objects were locally downloaded and independently verified.

## Baseline definitions

Both baselines use seed `17`, five warm-up queries, a cutoff of five, and the
same page text and chunks. Empty OCR pages retain their corpus position and
receive stable zero-valued representations or scores.

### BM25 chunk-max

- tokenizer: Unicode NFKC normalization, case folding, then Unicode
  alphanumeric terms;
- no stop-word removal;
- BM25Okapi formula compatible with `rank-bm25==0.2.2`;
- `k1=1.5`, `b=0.75`, and negative-IDF floor coefficient
  `epsilon=0.25`;
- 192-word windows with 32-word overlap; and
- maximum BM25 chunk score per page.

The implementation is local rather than a runtime call into `rank-bm25`.

### BGE-M3 dense chunk-max

- model:
  [`BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181);
- expected `pytorch_model.bin`:
  2,271,145,830 bytes, SHA-256
  `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`;
- Sentence Transformers `3.0.1` and Transformers `4.41.1`;
- CLS pooling, 1,024 dimensions, float32 output, and L2 normalization;
- no query prefix;
- configured maximum sequence length of 512 tokens;
- MPS batch size 2; and
- maximum cosine similarity over a page's chunks.

The BGE-M3 model card documents its dense 1,024-dimensional representation and
that no query instruction is required for this model. A post-run local
verification hashed the cached `pytorch_model.bin`: its observed size was
2,271,145,830 bytes and its observed SHA-256 was
`b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`,
matching the pin. Inspection of the loaded Sentence Transformers pipeline also
confirmed a 1,024-dimensional output, CLS pooling, float32 model dtype, a final
`Normalize` module, and an effective maximum sequence length of 512.

Those checks were added after the original schema-1 result processes. The raw
dense artifacts themselves record the revision, expected digest, observed
dimension, and effective maximum length, but not the later local hash and full
pipeline inspection.

### Post-run tokenizer audit

Chunks are defined in whitespace words rather than model tokens. The pinned
BGE-M3 tokenizer was therefore run later with special tokens included and with
padding and truncation disabled to measure the untruncated lengths:

| Task and input | Count | p50 tokens | p95 tokens | Maximum tokens | Above 512 |
|---|---:|---:|---:|---:|---:|
| DocVQA document chunks | 768 | 247 | 393.6 | 570 | 2 |
| DocVQA queries | 451 | — | — | 31 | 0 |
| InfoVQA document chunks | 949 | 273 | 389.6 | 643 | 2 |
| InfoVQA queries | 494 | — | — | 38 | 0 |

Thus, two document chunks in each task were truncated by the 512-token encoder
limit during the measured dense runs. No query was truncated. The exact
affected fraction is small, but the lost suffixes can still affect retrieval
quality; the result is reported rather than silently assuming that every
192-word window fits.

## Evaluation contract

Every measured query is scored against all 500 pages; no approximate-nearest-
neighbor index or candidate prefilter is used. Rankings are complete. Higher
scores rank first and exact score ties use descending document-ID order,
matching the tie convention used by `trec_eval`/`pytrec_eval`.

Quality is macro-averaged over queries using binary qrels:

- `nDCG@5` divides binary DCG by the ideal DCG for up to five relevant pages;
- `Recall@5` divides retrieved relevant pages by all relevant pages for that
  query; and
- `MRR@5` uses the first relevant page in the first five results.

Five warm-up queries are executed and excluded. Measured per-query latency
starts immediately before query scoring and ends after score validation and
the complete deterministic ranking. The artifact field
`documents_per_second` is therefore more precisely an end-to-end
query-document-pair throughput:

```text
number of measured queries * 500 pages / total scoring-and-ranking seconds
```

It is not indexing throughput, and for the dense baseline it is not pure MPS
encoder throughput.

The reported logical index size is deterministic payload accounting, not
resident memory or an on-disk serialized file:

- BM25 counts UTF-8 vocabulary bytes, float64 IDFs, int64 posting row/frequency
  pairs, float64 chunk lengths, and the int64 chunk-to-page map.
- Dense counts float32 chunk vectors and the int64 chunk-to-page map.

Model weights, Python object overhead, raw OCR text, and downloaded datasets
are excluded from logical index size.

## Local execution environment

- MacBook Pro `Mac15,6`
- Apple M3 Pro with 14-core GPU
- 18 GB unified memory (`19,327,352,832` bytes)
- macOS 15.6.1, arm64
- Python 3.9.6
- PyTorch 2.6.0
- NumPy 1.24.3
- fsspec 2025.2.0
- PyArrow 21.0.0
- project version 0.3.0
- CUDA unavailable; MPS available

BM25 indexing and scoring run on the CPU. BGE-M3 text encoding runs on MPS.
The dense chunk similarity matrix, chunk-to-page maximum reduction, score
validation, and final ranking run on the CPU through NumPy and CPU PyTorch.
Consequently, "dense on MPS" describes encoder placement, not the whole
retrieval pipeline. These timings are not comparable to the paper's NVIDIA L4
timings or to future Triton measurements.

Peak memory is the process high-water RSS for the complete run. It is not
device-only memory and can include model loading, dataset loading, indexing,
warm-up, and evaluation.

## Measured quality

These values are copied exactly from the four local JSON result artifacts.

| Task | Baseline | nDCG@5 | Recall@5 | MRR@5 |
|---|---|---:|---:|---:|
| DocVQA | BM25 chunk-max | 0.3246009818404028 | 0.36529933481152993 | 0.3127494456762749 |
| DocVQA | BGE-M3 dense chunk-max | 0.31115215156316384 | 0.36925351071692536 | 0.2929416112342942 |
| InfoVQA | BM25 chunk-max | 0.5742823958760168 | 0.6518218623481782 | 0.5488529014844803 |
| InfoVQA | BGE-M3 dense chunk-max | 0.6802346466685157 | 0.7504048582995951 | 0.6575573549257758 |

Within this exact local setup, BGE-M3 is clearly stronger than BM25 on
InfoVQA. On DocVQA, BM25 has higher nDCG@5 and MRR@5, while BGE-M3 has slightly
higher Recall@5. These are descriptive observations from one run per
configuration, not uncertainty-aware claims about the paper's systems.

## Measured indexing and memory

| Task | Baseline | Chunks | Dataset load (s) | Model load (s) | Index build (s) | Pages/s | Logical index (bytes) | Logical bytes/page | Peak process RSS (bytes) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DocVQA | BM25 chunk-max | 768 | 10.012629167 | 0.0 | 0.17573925000000123 | 2845.1242394627066 | 1330472 | 2660.944 | 328237056 |
| DocVQA | BGE-M3 dense chunk-max | 768 | 10.139506417 | 130.046589708 | 1228.45897825 | 0.4070139979051433 | 3151872 | 6303.744 | 4475076608 |
| InfoVQA | BM25 chunk-max | 949 | 9.58550775 | 0.0 | 0.11081637500000063 | 4511.968560603044 | 1944590 | 3889.18 | 325828608 |
| InfoVQA | BGE-M3 dense chunk-max | 949 | 11.536737583 | 41.530767791 | 978.4652485420002 | 0.5110043517079879 | 3894696 | 7789.392 | 5231427584 |

Model-load timing is wall-clock encoder initialization in that individual
process and depends on cache state. It should not be interpreted as a stable
model benchmark. DocVQA dense ran before InfoVQA dense and recorded the longer
load time.

## Measured query performance

`Score time` covers the baseline's `score(query)` operation. `Rank time`
covers score validation, conversion, and a complete 500-page ranking.
`End-to-end pairs/s` uses score plus rank time, as described above.

| Task | Baseline | Queries | Score time (s) | Rank time (s) | Total (s) | p50 latency (ms) | p95 latency (ms) | End-to-end pairs/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DocVQA | BM25 chunk-max | 451 | 0.9297834290000537 | 0.26313462399995835 | 1.192918053000012 | 2.4020000000000152 | 4.6325624999994375 | 189032.26372750493 |
| DocVQA | BGE-M3 dense chunk-max | 451 | 389.93086167500064 | 1.324554577999379 | 391.255416253 | 821.0863750000499 | 1211.249500499889 | 576.3498488010284 |
| InfoVQA | BM25 chunk-max | 494 | 0.7023800059999807 | 0.1119458780000091 | 0.8143258839999898 | 1.534916999999858 | 2.669881249999672 | 303318.37026563566 |
| InfoVQA | BGE-M3 dense chunk-max | 494 | 205.7374390549992 | 1.1700766180026676 | 206.90751567300185 | 377.1135834999768 | 860.4407041501049 | 1193.7700725688505 |

The large dense latency and indexing costs are real measurements from this
implementation. Encoding uses a small MPS batch while dense candidate
similarity and ranking remain CPU operations; no effort was made in phase 6 to
turn this text baseline into a production vector-search service.

## Raw artifacts

The source-of-record files for the exact values above are:

- `artifacts/phase6/docvqa_bm25.json`, created
  `2026-07-26T00:00:57.495358+00:00`
- `artifacts/phase6/infovqa_bm25.json`, created
  `2026-07-26T00:01:18.854023+00:00`
- `artifacts/phase6/docvqa_dense_bge_m3.json`, created
  `2026-07-26T00:32:06.256142+00:00`
- `artifacts/phase6/infovqa_dense_bge_m3.json`, created
  `2026-07-26T00:53:26.721804+00:00`

The UTC timestamps correspond to July 25, 2026 in the local
America/Los_Angeles time zone.

After the reporting and provenance path was hardened, separate schema-2 BM25
runs were saved as `docvqa_bm25_schema2.json` and
`infovqa_bm25_schema2.json`. They reproduced all three quality metrics, chunk
counts, and logical index sizes above exactly, while also recording successful
semantic-fingerprint checks, requested-versus-effective settings, source-file
hashes, and environment details. Their performance timings are separate
wall-clock observations and are intentionally not substituted into the
primary four-run tables, which retain one internally consistent schema-1
measurement set.

Canonical commands:

```text
.venv/bin/python scripts/evaluate_text_baseline.py --task docvqa --baseline bm25 --output artifacts/phase6/docvqa_bm25.json --overwrite
.venv/bin/python scripts/evaluate_text_baseline.py --task infovqa --baseline bm25 --output artifacts/phase6/infovqa_bm25.json --overwrite
.venv/bin/python scripts/evaluate_text_baseline.py --task docvqa --baseline dense --device mps --output artifacts/phase6/docvqa_dense_bge_m3.json --overwrite
.venv/bin/python scripts/evaluate_text_baseline.py --task infovqa --baseline dense --device mps --output artifacts/phase6/infovqa_dense_bge_m3.json --overwrite
```

## Reproducibility limitations

- The four raw artifacts use schema version 1 from the artifact-time evaluation
  script. They record configuration, revisions, semantic content fingerprints,
  environment metadata, and measurements, but not an immutable project source
  commit or source-tree digest. The later BM25-only schema-2 validation files
  add source-tree provenance but do not retroactively supply it for the dense
  runs.
- At report time this repository had no Git `HEAD`; all project files were
  untracked. The raw results therefore cannot be tied to a committed source
  revision. A future canonical rerun should record the hardened script's source
  fingerprint and Git state.
- Hugging Face LFS hashes in the dataset manifest are expected remote metadata,
  not independently recomputed full-file hashes from the column-selective
  loader.
- The BGE-M3 hash and effective-pipeline checks, and the tokenizer-length
  audit, were performed after the schema-1 dense processes. They strengthen
  the result audit but are not embedded in those two raw JSON files.
- Timings are single-process, single-run wall-clock observations with five
  warm-ups. There are no repeat distributions or confidence intervals.
- MPS encoder work and CPU similarity/ranking are mixed in one end-to-end
  latency measurement.
- The virtual environment reuses system-site packages. `pip check` reports an
  unrelated ambient conflict: `alpaca-trade-api 3.2.0` requires
  `websockets<11`, while `websockets 15.0.1` is installed. Neither package is
  used by this project, but the environment is not fully isolated.
- Raw datasets, model caches, and generated result JSON files are intentionally
  excluded from Git. Reproduction requires network access to the pinned
  Hugging Face objects unless the cache is already populated.
- These text-only results cannot establish any claim about ColPali,
  multimodal retrieval, LoRA training, CUDA, or Triton performance.

## Phase boundary

Phase 6 demonstrates two measured real-data text baselines on two pinned
ViDoRe subsets. It does not satisfy the project's final acceptance criterion
for a scoped ColPali reproduction. The next research step is phase 7:
integrating the pretrained multimodal backbone, the 128-dimensional
token/patch projection, normalization, and the ColPali-style retrieval path
without changing this text-baseline reference point.
