# Final Report: Scoped ColPali Reproduction and Triton MaxSim

## Outcome

The scoped reproduction is complete. It implements and tests the
retrieval-specific ColPali components, records real OCR and dense baselines,
evaluates the released ColPali checkpoint against a controlled mean-pooling
ablation, and validates a custom tiled MaxSim forward kernel on an NVIDIA L4.

The final GPU run was sealed with no failure reasons. The Triton kernel matched
the vectorized PyTorch reference across the required FP16/BF16 shapes and masks,
won all 12 measured benchmark cases, delivered a 2.27x geometric-mean latency
speedup, and reduced incremental allocator peaks from as much as 41,798,144
bytes to at most 1,536 bytes.

These results are a scoped reproduction, not the paper's full eight-GPU
training run or complete ten-task ViDoRe evaluation.

## Reproduction scope

The project independently implements:

- a nested-loop MaxSim oracle and vectorized PyTorch reference;
- mask-aware batched query-document scoring;
- the paper's one-way hardest-in-batch pairwise objective;
- PaliGemma-to-128-dimensional projection, L2 normalization, query
  augmentation, and LoRA topology;
- deterministic ViDoRe subset selection, metrics, indexing, and retrieval;
- OCR/BM25 and BGE-M3 text baselines;
- a normalized mean-pooled single-vector ablation;
- a tiled Triton forward kernel that does not materialize the full
  query-document similarity tensor; and
- a fail-closed AWS workflow for hardware validation, tuning, benchmarking,
  profiling, sealing, retrieval, and termination.

It intentionally uses the public paper-era base and released adapter rather
than retraining the approximately three-billion-parameter model. A Triton
backward kernel and compressed index are outside the required scope.

## Verified environment

| Component | Recorded value |
| --- | --- |
| Cloud host | AWS `g6.xlarge`, `us-west-2` |
| GPU | NVIDIA L4, 23,661,248,512 bytes, compute capability 8.9 |
| Driver | 595.71.05 |
| Host | Linux x86-64, 4 vCPUs |
| PyTorch | 2.6.0+cu124 |
| CUDA build | 12.4 |
| Triton | 3.2.0 |
| Docker | 29.6.2 |
| Source revision | `0f9543e30e49fc30de8c476b2767e1c29073602b` |

The container base is pinned by manifest digest:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@
sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
```

The launch receipt, EC2 instance metadata, source archive, container revision,
benchmark provenance, and retrieved artifact manifest were cross-checked
before the run could be marked complete.

## Correctness

The local suite passed `664` tests with five expected CUDA-only skips. On the
L4, the CUDA image passed `276` tests; its only skip was the Apple-MPS-specific
test.

The separate CUDA assertion proved that all five mandatory GPU cases executed
without skips:

- FP16, BF16, and FP32 vectorized-reference parity;
- non-contiguous inputs and empty masks; and
- active NaN propagation across document tiles.

Every one of the 12 timed FP16/BF16 benchmark records agreed with PyTorch at
the committed tolerances. The largest observed absolute error was
`0.0094871521`, in the long-query BF16 case, below the BF16 absolute tolerance
of `0.4`. FP16's largest observed absolute error was `0.0010385513`, below its
absolute tolerance of `0.05`.

## Tuning

The tuning matrix evaluated 12 explicit launch configurations over four
representative shapes and two dtypes. It performed correctness gates before
timing, excluded cold compilation, executed 960 warmups, and collected 12,000
CUDA-event samples.

The selected cross-workload winner was:

```text
candidate                    tq16_td128_w4_s2
query-token block            16
document-token block         128
embedding-dimension block    128
warps                        4
stages                       2
```

The full benchmark and Nsight probe both consumed this exact winner record.
Per-workload optima varied, so the selected configuration is a stable
cross-workload compromise rather than a claim that one tile is optimal for
every shape.

## GPU benchmark

Each provider/case combination used 10 warmups followed by five trials of 50
measured calls. Timings use CUDA events and synchronize at trial boundaries.
Correctness is checked before timing. Speedup is PyTorch mean latency divided
by Triton mean latency.

| Shape | Dtype | PyTorch mean | Triton mean | Speedup | PyTorch incremental peak | Triton incremental peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q15, d1 | FP16 | 0.2326 ms | 0.1050 ms | 2.21x | 332,288 B | 1,024 B |
| q15, d8 | FP16 | 0.2392 ms | 0.1045 ms | 2.29x | 2,617,344 B | 1,024 B |
| q15, d32 | FP16 | 0.2437 ms | 0.1027 ms | 2.37x | 10,454,016 B | 1,024 B |
| q15, d128 | FP16 | 0.4706 ms | 0.1114 ms | 4.22x | 41,798,144 B | 1,024 B |
| padded q32, d64 | FP16 | 0.2379 ms | 0.1470 ms | 1.62x | 25,388,032 B | 1,536 B |
| long q50, d32 | FP16 | 0.2368 ms | 0.1483 ms | 1.60x | 15,794,688 B | 1,536 B |
| q15, d1 | BF16 | 0.2273 ms | 0.1022 ms | 2.22x | 332,288 B | 1,024 B |
| q15, d8 | BF16 | 0.2323 ms | 0.1020 ms | 2.28x | 2,617,344 B | 1,024 B |
| q15, d32 | BF16 | 0.2354 ms | 0.1011 ms | 2.33x | 10,454,016 B | 1,024 B |
| q15, d128 | BF16 | 0.4754 ms | 0.1102 ms | 4.31x | 41,798,144 B | 1,024 B |
| padded q32, d64 | BF16 | 0.2317 ms | 0.1476 ms | 1.57x | 25,388,032 B | 1,536 B |
| long q50, d32 | BF16 | 0.2676 ms | 0.1500 ms | 1.78x | 15,794,688 B | 1,536 B |

Summary:

| Measure | Result |
| --- | ---: |
| Triton wins | 12 / 12 |
| Geometric-mean speedup | 2.2697x |
| Speedup range | 1.5702x–4.3134x |
| Triton mean-latency range | 0.1011–0.1500 ms |
| Maximum absolute error | 0.0094871521 |

The speedup grows to more than 4.2x for 128 documents because PyTorch's
vectorized path allocates and reduces a large intermediate tensor, while the
Triton kernel keeps the document-token maximum tiled. The smaller advantage on
long or padded queries reflects additional masked query tiles and reduction
work. No measured canonical case was slower than PyTorch.

The allocator figures are incremental PyTorch CUDA allocator peaks, not total
GPU board usage. They demonstrate the avoided intermediate allocation but
should not be read as whole-process or driver memory.

## Profiling

Nsight Compute completed one bounded capture using the exact tuning winner:

```text
query shape       [1, 15, 128]
document shape    [32, 1030, 128]
output shape      [1, 32]
dtype             FP16
report size       121,329 bytes
```

The profiler container was not privileged; it received only `SYS_ADMIN`, and
the workflow removed all exact-run containers before sealing. The report's
metadata matched the winner hash. Nsight Systems fallback was therefore not
needed.

## Retrieval reproduction and ablations

The released ColPali checkpoint was evaluated on precommitted 50-query,
100-page subsets:

| Task | Scorer | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | ColPali MaxSim | 0.6720 | 0.8200 | 0.6217 |
| DocVQA | Mean pooling | 0.2805 | 0.3200 | 0.2667 |
| InfoVQA | ColPali MaxSim | 0.8679 | 0.9000 | 0.8567 |
| InfoVQA | Mean pooling | 0.6904 | 0.8400 | 0.6413 |

This controlled ablation supports the paper's qualitative claim that
multi-vector late interaction preserves matching detail lost by a
single-vector summary. The complete multi-vector payload was 264,710 bytes per
page including its mask, versus 512 bytes for the mean vector: a substantial
quality/storage tradeoff.

The independent OCR-text baselines used complete 500-page task subsamples:

| Task | Baseline | nDCG@5 | Recall@5 | MRR@5 |
| --- | --- | ---: | ---: | ---: |
| DocVQA | BM25 | 0.3246 | 0.3653 | 0.3127 |
| DocVQA | BGE-M3 | 0.3112 | 0.3693 | 0.2929 |
| InfoVQA | BM25 | 0.5743 | 0.6518 | 0.5489 |
| InfoVQA | BGE-M3 | 0.6802 | 0.7504 | 0.6576 |

Because those baselines and the ColPali evaluation use different fixed pools,
their numbers are not presented as a direct head-to-head comparison.

The required mean-pooling ablation was feasible and completed. A new
frozen-backbone-versus-LoRA training ablation was not computationally feasible
within this scoped reproduction, so no result is invented for it.

## Findings relative to the paper

Reproduced or supported:

- the retrieval representation and late-interaction equation are implemented
  directly and tested against an explicit oracle;
- the released checkpoint gives strong retrieval results on both fixed
  multimodal subsets;
- late interaction materially outperforms mean pooling on identical model
  vectors and candidate pools; and
- a fused/tiled MaxSim kernel can remove the large similarity intermediate and
  improve GPU scoring latency.

Not reproduced:

- the original eight-GPU LoRA training run;
- the paper's full ten-task ViDoRe average and full-corpus task values;
- the separately trained BiPali baseline;
- the paper's OCR-plus-caption text pipeline; or
- training-time throughput and backward-kernel performance.

New project-specific result:

- on one NVIDIA L4, this Triton forward kernel was 1.57x–4.31x faster than the
  committed PyTorch vectorized reference over the canonical matrix, with a
  2.27x geometric mean and numerical agreement at all committed tolerances.

These scopes prevent the subset results or kernel microbenchmarks from being
misrepresented as a complete replication of the paper.

## Provenance and raw artifacts

The compact committed record is
[`phase_11_gpu_results.json`](phase_11_gpu_results.json). Generated raw
artifacts remain ignored:

```text
artifacts/phase11/remote/
  0f9543e30e49fc30de8c476b2767e1c29073602b/
  phase11-l4-0f9543e-20260729-final/
```

Integrity records:

| Artifact | SHA-256 |
| --- | --- |
| Sealed artifact manifest | `b2b66c56a8227ec9cd4b082ee4cab937591ee4548eb14ac6193e6ab1cb568e53` |
| Retrieved remote bundle | `8d63be08908dcc6ffe893913e8dbe985ff8d2311ce15d80c20c52d691f599fd5` |
| Benchmark JSON | `88058b563bdfd38b226e8bd5ff40f88bab0e3e34a1f65d0858c8d814e021d221` |
| Tuning JSON | `74535b5388d45194fc7dbe2b4722e2ece14b7ef90500987496cfeb3f0a210a5a` |
| Nsight Compute report | `4141ff9c485a25e012a9a94f6be06d8eed3cfdf1f4cc591327dc75da1fa17aaf` |

The raw bundle contains 37 declared files: hardware and host gates, exact
commands, build/test logs, JUnit, tuning records, derived benchmark config,
benchmark output, profile and metadata, result-contract validation, manifest,
seal, and independent retrieval receipt.

## AWS lifecycle

The final host launched at `2026-07-29T18:31:49Z`. Termination was requested
immediately after verified retrieval at `18:38:13Z` and AWS confirmed the
terminal state at `18:45:02Z`. The 100 GiB encrypted root volume was configured
for deletion on termination.

Post-run checks found zero active project instances and zero project volumes.
The temporary EC2 Instance Connect endpoint and both benchmark security groups
were deleted. The no-cost EC2 key-pair registration and local private key were
retained so a future authorized rerun does not require generating new
credentials.

## Limitations

- GPU numbers come from one short run on one L4; they do not characterize
  device-to-device or temporal variance.
- Timings are kernel-path microbenchmarks with resident synthetic normalized
  inputs. They exclude model encoding, host-to-device transfer, index loading,
  distributed search, and end-to-end retrieval.
- Shapes are representative of the released 448-pixel ColPali encoder and
  several query/document batch patterns, but do not cover every possible
  sequence length or embedding dimension.
- PyTorch allocator statistics do not include every driver or compiler
  allocation.
- The selected tuning winner optimizes the committed workload matrix, not an
  arbitrary production distribution.
- Retrieval quality uses small deterministic subsets; full-task comparisons
  would require materially more compute and storage.
- The public base/adapter revisions are pinned, but the project does not claim
  the currently gated Google checkpoint is the paper's original training
  revision.

## Acceptance conclusion

All original acceptance criteria are now satisfied:

- local unit tests pass;
- nested-loop and vectorized PyTorch MaxSim agree;
- the controlled objective overfits;
- two real retrieval baselines are recorded;
- scoped released-ColPali results and an ablation are recorded;
- Triton executes on NVIDIA hardware and matches PyTorch;
- reproducible GPU benchmarks identify exact hardware and software; and
- the README, phase reports, compact GPU record, and raw sealed bundle preserve
  methods, deviations, results, and limitations.
