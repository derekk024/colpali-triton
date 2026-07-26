# Benchmarks

`benchmark_maxsim.py` is the Phase 10/11 CUDA benchmark harness. Its canonical
matrix is fixed by `configs/maxsim_benchmark.json`:

- PyTorch vectorized and Triton fused providers
- FP16 and BF16
- six shapes using the ColPali embedding dimension (128) and document length
  (1,030), with 1–128 candidate documents, 15–50 query tokens, and one padded
  case
- a fixed Triton launch configuration of 16 query tokens, 64 document tokens,
  and 128 embedding values per block, with four warps and two stages
- 10 untimed warmups followed by five trials of 50 CUDA-event measurements

Run a non-mutating readiness check first:

```bash
python benchmarks/benchmark_maxsim.py --preflight
```

The check exits successfully and prints structured JSON even when the machine
has no CUDA device or Triton installation. A benchmark run itself fails before
creating an artifact unless the selected device and every selected provider
are available.

Run the complete canonical matrix inside the pinned CUDA environment:

```bash
python benchmarks/benchmark_maxsim.py
```

The default result is
`artifacts/phase10/maxsim_cuda_benchmark.json`. Existing results are preserved
unless `--overwrite` is explicit. Provider, dtype, and case filters are
available for diagnostics:

```bash
python benchmarks/benchmark_maxsim.py \
  --provider triton \
  --dtype float16 \
  --case colpali_q15_d32 \
  --output artifacts/phase10/diagnostic.json
```

Filtered output is marked `canonical_full_matrix: false` and must not be
reported as the canonical comparison.

## Measurement contract

Each provider must numerically agree with the identical PyTorch output before
timing starts. The cold first call is synchronized and recorded separately so
Triton compilation is not hidden or mixed into steady-state latency. Warmups
are also separate.

Every measured call has its own start/end CUDA events. Events for one trial are
enqueued and followed by one explicit CUDA synchronization. The report retains
all samples plus p50/p95 latency. Headline throughput is total work divided by
total event time; mean and median instantaneous rates are labeled separately.
Peak allocated and reserved CUDA memory are reset and recorded per trial, both
as totals and as increments above the resident input/reference tensors.

The JSON record includes:

- the raw config SHA-256, semantic config fingerprint, source-file digests,
  aggregate source/config fingerprint, Git commit and dirty-state digest
- exact selection, seeds, shapes, masks, input storage, tolerances, provider
  entry points, and Triton launch parameters
- Python, package, PyTorch, CUDA, cuDNN, GPU, driver, UUID, PCI, memory, clock,
  power, and relevant environment-variable metadata when exposed by the host
- cold-call time, warmup time, all CUDA-event samples, latency summaries,
  throughput summaries, correctness error, and allocator peaks

The harness does not infer or extrapolate missing measurements. A completed
NVIDIA artifact is required before GPU performance claims can be made.
