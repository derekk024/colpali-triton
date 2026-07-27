# Phase 11 Pre-GPU Readiness

## Outcome

The Triton MaxSim kernel and the complete NVIDIA execution path are ready for
one controlled L4 session. This is still a readiness result, not a GPU result:
the kernel has not compiled or executed on NVIDIA hardware, and no latency,
throughput, CUDA memory, tuning winner, or profiler claim is reported here.

The overall project remains incomplete until the L4 artifacts are retrieved
and the Phase 12 report is written.

## Local verification

The clean Apple M3 Pro environment, using Python 3.9.6, produced:

```text
664 passed, 5 skipped in 5.44s
python -m pip check: no broken requirements
all Python sources compiled successfully
all five shell entry points passed bash syntax checks
benchmark preflight: blocked only by absent CUDA/Triton
tuning preflight: 12 candidates validated, blocked only by absent CUDA/Triton
```

The five skips are exactly the real CUDA/Triton cases:

- FP16, BF16, and FP32 randomized parity with masks;
- non-contiguous inputs and empty-mask behavior; and
- active-NaN propagation across document tiles.

Those tests must all execute with zero skips on the L4. A separate JUnit
validator enforces that contract; a successful generic pytest exit is not
enough.

## Kernel audit

The forward path tiles query tokens, document tokens, and the 128-dimensional
embedding axis. It accumulates dot products and final scores in FP32 and stores
only one partial score per query block and query-document pair rather than the
complete four-dimensional similarity tensor.

The final local audit added:

- explicit NaN propagation matching `torch.amax`;
- a direct partial-buffer result for the common one-query-block path, avoiding
  a redundant allocation and reduction launch;
- non-contiguous stride and empty-mask handling;
- an autograd fallback to the vectorized PyTorch implementation; and
- explicit, validated query/document/dimension blocks, warp counts, and stage
  counts.

Actual SM89 compilation, register/shared-memory use, numerical error, and
performance remain GPU-only evidence.

## Fixed tuning and benchmark contracts

The tuning configuration has:

```text
raw config SHA-256       25b0706ea22bc610791070ef1feee8987c9779f39269b09c861337c70c297f3a
semantic SHA-256         490975369fac2bbd58ce70332057227ac2e78b65155d95ce6cf729f55d40bb56
candidates               12
workloads                4 shapes × 2 dtypes
cold correctness gates   96
warmups                   960
measured CUDA calls      12,000
```

Every candidate is correctness-gated before timing. Candidate order is
deterministically reshuffled per trial, rejected candidates remain in the
artifact, non-finite values cannot produce non-standard JSON, and environment
snapshots are taken before and after the measured work.

The full benchmark configuration has:

```text
raw config SHA-256       4420acdcff533745c3533ba1170934886d6d3ba3340b9725531412192d652f25
semantic SHA-256         a512ce1e9acc8acb2a05d7e99cfaf978fe5a541261bad9816c5aca4b04acfe2c
matrix                   6 shapes × 2 dtypes × 2 providers
timing                   10 warmups; 5 trials × 50 CUDA events
```

The remote workflow derives a new valid benchmark configuration containing the
tuning winner. It then cross-checks the tuning result, winner record, derived
configuration, completed 12-workload benchmark, source commit, and every
relevant SHA-256 both before sealing and again after local retrieval. The
profiler probe also verifies that it used the same winner.

The execution controller additionally requires the exact ignored receipt
written by the guarded lifecycle launch. It checks the receipt schema, source
commit, environment fingerprint, user-data digest, container/bootstrap pins,
resolved AMI, AWS account, region, instance, subnet, and security group.
Remote IMDS evidence must match that receipt, and local retrieval recomputes
the host validation before accepting a complete result.
The lifecycle refuses billable launch from an unresolved or dirty worktree and
rechecks the captured commit and user-data digest immediately before
`run-instances`.

## L4 and profiler gates

Before tests or tuning, the container must report:

```text
device name              NVIDIA L4
compute capability       8.9
reported memory          between 21 and 25 GiB
```

The CUDA image is digest-pinned and asserts PyTorch 2.6.0, CUDA 12.4, Triton
3.2.0, x86-64, and the presence of `ncu` or `nsys`. Its immutable base metadata
includes the pinned CUDA 12.4 Nsight Compute package.

Profiling first tries one bounded Nsight Compute launch, then Nsight Systems
only if needed. Each attempt is capped at 300 seconds. Only profiler
containers receive `SYS_ADMIN`, as required for NVIDIA hardware counters; no
container is privileged. Completion requires exit code zero, a nonempty
report, and metadata matching the exact tuning winner.

## Failure and lifecycle behavior

The paid job is detached and resumable, with a four-hour hard cap. Control
commands, source upload, bundle creation, and artifact download have separate
bounds; only ambiguous SSH transport loss is retried. A deterministic remote
command failure stops immediately instead of consuming the reconnect window.

After the remote result directory is created, a failed build, hardware check,
test, tuning pass, benchmark, or profile is still sealed and retrieved. If the
outer job bound or an unexpected host-side exit interrupts normal sealing, a
separately bounded, locked finalizer removes only containers labeled for the
exact run/source, quarantines partial contract files, recomputes the failed
result contract, and atomically publishes an `incomplete` manifest and seal.
Both the normal seal path and recovery must complete a final exact-label query
proving that no run container remains; uncertainty retains the cidfiles and
leaves the result unsealed. Polling can invoke the same idempotent recovery
after a disconnect. The local command returns nonzero only after the recovered
bundle and every declared file have been verified. A finalizer failure remains
fail-closed and resumable; it cannot be reported as a retrieved result.

The remote workflow never launches, stops, or terminates an instance. Launch
requires a separate exact cost acknowledgement. Termination separately
verifies the instance ID and managed tags, performs an AWS permission dry run,
waits for `instance-terminated`, records the final state, and refuses to claim
confirmation if AWS does not reach a terminal state.

## AWS decision

An NVIDIA rental is now necessary. `g6.xlarge` is the recommended first run:
it provides the same single L4 as `g6.2xlarge`, while this kernel-only workflow
does not need the larger host CPU/RAM allocation. Current EC2 and EBS prices
must be checked for the exact selected region before launch.

No AWS profile, access key, secret key, or region is configured on the local
Mac, and STS cannot locate credentials. No resource has been created. Launch
also still requires an explicit region, subnet ID, security-group ID, EC2
key-pair name, reachable SSH path, readable private-key path, and explicit
acknowledgement of the current regional EC2 and EBS cost.

## Remaining acceptance work

1. Launch one guarded `g6.xlarge` in the user-selected region.
2. Run the sealed Phase 11 workflow and retrieve a `complete` bundle.
3. Inspect and preserve the launch, host, container, test, tuning, benchmark,
   profiler, retrieval, and confirmed-termination records.
4. Commit compact sanitized GPU result records and hashes for any large trace.
5. Write Phase 12 with numerical errors, per-shape wins and losses, tuning and
   profiling findings, exact hardware/software identity, limitations, and
   discrepancies.

The Phase 9 ColPali and mean-pooling comparison is controlled on identical
subsets. The BM25 and BGE-M3 runs remain meaningful real-data baselines but use
the complete 500-page task subsamples, not the Phase 9 100-page pools. Phase 12
must preserve that comparability boundary rather than present a fabricated
head-to-head result.
