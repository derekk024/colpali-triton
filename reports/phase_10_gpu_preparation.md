# Phase 10 GPU Preparation and Phase 11 Readiness

## Outcome

The CUDA container, guarded AWS L4 lifecycle, committed-source deployment
path, benchmark protocol, and initial Triton forward kernel are implemented
and locally verified. No AWS resource was created.

This is a readiness result, not a GPU result. The Triton kernels have not yet
been compiled or executed on NVIDIA hardware, so numerical agreement,
profiling, tuning, latency, throughput, and CUDA memory numbers remain pending.

## Pinned GPU environment

| Component | Pinned value |
| --- | --- |
| Platform | `linux/amd64` |
| Base image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` |
| Base manifest | `sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56` |
| PyTorch | `2.6.0` |
| CUDA build | `12.4` |
| Triton | `3.2.0` |
| AWS instance types | `g6.xlarge`, `g6.2xlarge` |
| GPU contract | one NVIDIA L4 |
| DLAMI selector | AWS public x86-64 Ubuntu 22.04 Base OSS NVIDIA-driver parameter |

The Docker registry manifest was independently checked as an amd64 Linux
manifest. The Dockerfile retains the readable tag and immutable digest,
asserts the expected software versions during build, and records the source
commit in the image metadata.

The environment manifest fingerprint is:

```text
b06d77f646da2d73eed72f171908f2232aae9b1e3c1ec9c3badf4b124817c6be
```

Important file digests at this milestone:

| File | SHA-256 |
| --- | --- |
| `infra/aws/l4_environment.json` | `1fa7c0e698f5486e4574fc1f51b5e7ed8aa448daad9340be27ce3811594e88b0` |
| `docker/Dockerfile.cuda` | `7effc59c8cb6c8cf13f94c3956cb6a16de6f87b0cf3a32be624664cb45b925f4` |
| `infra/aws/user_data.sh` | `d0bc2fe70c06a09c11f4501ef0abeb18e7ca4bea796146f1cd71b7786d4baa98` |

## AWS safety and reproducibility

The lifecycle interface has four operations:

- `plan` validates and prints a complete launch plan without calling AWS;
- `launch` is also offline unless `--execute` and an exact cost
  acknowledgement are both present;
- `status` describes one exact instance ID; and
- `terminate` is offline by default, requires the ID twice for execution, and
  verifies the component's managed tags before mutation.

There are no default-network fallbacks. Region, subnet, security group, and
EC2 key name are mandatory. Public IPv4 is explicitly disabled unless
requested. Launch is limited to one approved L4 instance, uses an encrypted,
delete-on-termination `gp3` root volume, requires IMDSv2, and derives a stable
EC2 client token from the exact plan.

If an explicit AMI is not supplied, execution resolves AWS's documented
public DLAMI parameter, verifies that the result is an available Amazon-owned
x86-64 Ubuntu 22.04 GPU Base image, and records the immutable AMI ID in the
protected launch receipt. Receipts are created atomically and never
overwritten.

The deployment helper refuses a dirty worktree and streams
`git archive HEAD`. It creates a commit-specific remote directory, records the
commit in the image label, builds the pinned image, and runs the
PyTorch/Triton CUDA tests. It does not copy ignored artifacts, credentials,
caches, or uncommitted files.

## Canonical benchmark protocol

The committed benchmark matrix contains:

```text
providers                 PyTorch vectorized, Triton fused
dtypes                    FP16, BF16
query tokens              15, 32, 50
document tokens           1,030
candidate documents       1, 8, 32, 64, 128
embedding dimension       128
warmups                    10
measured calls per trial  50
trials                     5
Triton launch              Q16 / D64 / K128 / 4 warps / 2 stages
```

One case uses both query and document padding. The benchmark generates
deterministic normalized inputs, computes an identical PyTorch reference,
requires numerical agreement before timing, separates the cold compile call,
uses one pair of CUDA events per measured invocation, synchronizes after each
trial, and retains all samples.

Each output records p50/p95 latency, aggregate documents and query-document
pairs per second, total and incremental peak allocated/reserved CUDA memory,
exact GPU properties and UUID, driver and package versions, clocks and power
when exposed, relevant environment variables, Git state, per-source hashes,
raw and semantic config hashes, and the exact Triton launch parameters.
Existing result files are protected unless overwrite is explicit.

Benchmark configuration fingerprints:

| Kind | SHA-256 |
| --- | --- |
| Raw `configs/maxsim_benchmark.json` | `4420acdcff533745c3533ba1170934886d6d3ba3340b9725531412192d652f25` |
| Validated semantic configuration | `a512ce1e9acc8acb2a05d7e99cfaf978fe5a541261bad9816c5aca4b04acfe2c` |

## Initial Triton implementation

`maxsim_triton` tiles query tokens, document tokens, and the embedding
dimension. Each program computes document-token maxima for one query tile and
query-document pair. A second small kernel sums the query-tile partials.

The native path therefore stores:

```text
[query-document pairs, ceil(query tokens / query block)]
```

rather than the PyTorch reference's complete:

```text
[query batch, document batch, query tokens, document tokens]
```

The implementation supports masks, all-pairs batching, noncontiguous strides,
FP16/BF16 inputs with FP32 accumulation, and explicit block, warp, and stage
tuning. Because a backward Triton kernel is a stretch goal, autograd-enabled
calls deliberately retain the tested vectorized PyTorch path.

## Local verification

The clean Apple M3 Pro environment produced:

```text
510 passed, 4 skipped in the full pytest suite
python -m pip check: no broken requirements
17 focused container/AWS tests passed
all four shell entry points passed bash syntax checks
the lifecycle module compiled under Python 3.9
the installed AWS CLI v1.44 accepted the generated argument schema
```

The four skips are exactly the committed CUDA/Triton numerical tests: FP16,
BF16, and FP32 randomized parity plus a noncontiguous/empty-mask case.

Local benchmark preflight correctly reports:

```text
PyTorch build includes CUDA  false
CUDA available               false
Triton package available     false
```

Docker is not installed on the Mac. AWS CLI `1.44.87` is installed, but no
profile, access key, secret key, or region is configured; STS reports that it
cannot locate credentials. These facts block real execution but do not weaken
the offline plan, command-construction, safety, or provenance tests.

## Remaining acceptance gate

Phase 10 preparation is complete. Phase 11 and the overall project remain
incomplete until an NVIDIA L4 run:

1. launches and records the exact AMI/instance environment;
2. builds the digest-pinned source image;
3. runs every CUDA-gated numerical test for FP16/BF16 shapes and masks;
4. executes the complete canonical benchmark;
5. profiles the kernel and evaluates multiple explicit tuning configurations;
6. records honest wins and losses versus PyTorch; and
7. terminates the instance and records the verified termination receipt.

Only those measured artifacts can support the final Phase 12 report and a
claim that every acceptance criterion is satisfied.

## Primary sources

- [AWS accelerated-instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS x86-64 Ubuntu 22.04 GPU Base DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html)
- [AWS `run-instances` reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html)
- [AWS EC2 user-data behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [PyTorch 2.6 CUDA 12.4 installation matrix](https://pytorch.org/get-started/previous-versions/)
- [Triton matrix-multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
