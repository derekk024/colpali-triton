# Phase 11 Triton MaxSim tuning protocol

This tuning pass is prepared locally and runs only on an NVIDIA CUDA host. It
does not need an AWS instance for configuration validation or preflight.

## Fixed experiment

`configs/triton_tuning.json` commits the complete experiment:

- 12 named `TritonMaxSimConfig` candidates. Every launch tuple is explicit;
  the runner does not use Triton autotuning or silently expand a search space.
- FP16 and BF16, with separate numerical-agreement tolerances.
- Four representative ColPali workloads: single-document latency, 32-document
  retrieval, padded 64-document retrieval, and a 50-token long query.
- 10 unmeasured warmups followed by five trials of 25 measurements.
- deterministic inputs, normalized before casting from FP32, with TF32 and
  cuDNN benchmarking disabled.

The candidate matrix varies query/document tiling, warp count, and stage count.
The embedding block remains 128 because the scoped ColPali model emits
128-dimensional vectors.

## Safe local preflight

Run this on any development machine:

```bash
python benchmarks/tune_triton_maxsim.py --preflight
```

Preflight validates the strict JSON schema, instantiates every selected launch
tuple with the production `TritonMaxSimConfig` class, and reports CUDA/Triton
readiness as JSON. A non-CUDA Mac reports `blocked` but exits successfully. It
does not compile a kernel, allocate a CUDA tensor, create directories, or
write an artifact.

## NVIDIA run

Inside the pinned Phase 10 CUDA container:

```bash
python benchmarks/tune_triton_maxsim.py \
  --output artifacts/phase11/triton_tuning.json
```

The default output path is the same path shown above. An existing artifact is
refused before GPU work begins. Replacement requires an explicit
`--overwrite`; use a new path for independent repetitions.

Candidate, dtype, and case filters exist for diagnostics:

```bash
python benchmarks/tune_triton_maxsim.py \
  --candidate tq16_td64_w4_s2 \
  --dtype float16 \
  --case retrieval_q15_d32 \
  --output artifacts/phase11/diagnostic.json
```

Filtered runs are marked `canonical_full_matrix: false` and are not a
canonical tuning result.

## Correctness and timing

For each shape/dtype workload, PyTorch vectorized MaxSim produces one reference
from the exact candidate inputs. Every Triton candidate then performs its cold
compile/call and must agree with that reference before it can warm up or enter
timing. Compilation failures, non-finite output, shape differences, numerical
rejections, warmup failures, and timing failures stay in the result; rejected
candidates are never timed. Non-finite error magnitudes are represented as
JSON `null`, and artifact serialization rejects non-standard `NaN` or
`Infinity` values. If failure cleanup cannot synchronize CUDA, the original
failure remains primary and the cleanup error is recorded separately.

Steady-state calls use one pair of `torch.cuda.Event` objects per invocation.
One synchronization follows each candidate trial. Candidate order is
deterministically reshuffled for every trial to reduce fixed ordering bias,
and every execution order and raw event sample is retained.

The per-workload winner has the lowest pooled p50 CUDA-event latency. The
overall winner must complete every selected workload. Its primary score is the
median of its workload-relative p50 latencies, giving each committed workload
equal weight. Ties use aggregate measured CUDA-event latency and then candidate
ID. The artifact also reports per-dtype winners and the complete ranking.

## Provenance contract

The artifact records the raw and semantic config digests, per-source digests,
an aggregate source/config fingerprint, Git state, container source markers,
package versions, PyTorch/CUDA/cuDNN details, exact GPU identity and properties,
`nvidia-smi` metadata, inputs, seeds, launch tuples, all candidate outcomes,
all timing samples, and allocator peaks. Environment snapshots are captured
once before any workload runs and once after all workloads complete; each
snapshot records its own capture time and current free-memory state.

Remote source provenance must agree across the Git checkout,
`COLPALI_SOURCE_COMMIT`, and `.source_commit` whenever those sources are
present. A disagreement invalidates the run instead of producing an artifact.
The source deployment must therefore be committed before the NVIDIA run.
