# Phase 10 AWS NVIDIA L4 environment

This directory prepares one deliberately scoped EC2 benchmark host. It does
not create a VPC, subnet, security group, key pair, IAM role, or any other
shared infrastructure. The lifecycle tool only accepts `g6.xlarge` and
`g6.2xlarge`, which AWS documents as one NVIDIA L4 each with 22 GiB of reported
GPU memory.

The CUDA container is always `linux/amd64`. Its PyTorch base is both
human-readable and immutable:

```text
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56
```

The image asserts PyTorch 2.6.0, CUDA 12.4, and Triton 3.2.0 while building.
An Apple Silicon build must still request `linux/amd64`; an emulated Mac build
is not a GPU benchmark environment.

## Prerequisites

Before spending anything, supply all four placement/access values explicitly:

- an AWS region where the chosen G6 type is available and your account has
  quota;
- a subnet ID;
- a security group ID with only the access your workflow needs;
- an existing EC2 key-pair name.

The tool never falls back to a default VPC or default security group. A public
IPv4 address is disabled unless `--associate-public-ip` is explicit. Check the
current on-demand EC2 and EBS prices for the exact region before launch; this
repository intentionally does not freeze a price that will become stale.

AWS CLI v1 or v2 is supported. Configure credentials outside the repository.
No credential, private key, token, or provider cache belongs in Git.

## Offline plan

`plan` validates inputs and prints the exact command without invoking AWS, so
it works without credentials. Replace the example IDs and key name:

```bash
python3 infra/aws/l4_lifecycle.py plan \
  --region us-west-2 \
  --subnet-id subnet-0123456789abcdef0 \
  --security-group-id sg-0123456789abcdef0 \
  --key-name colpali-benchmark
```

By default, execution resolves AWS's public x86-64 Ubuntu 22.04 GPU Base DLAMI
SSM parameter and records the resulting immutable AMI ID. Pass `--ami-id`
from an earlier receipt to rerun the exact image. Planning with neither form
uses an obvious non-executable placeholder.

The launch request uses an encrypted 100 GiB `gp3` root volume that is deleted
with the instance, IMDSv2-only metadata, managed tags on the instance and
volume, stopped-on-guest-shutdown behavior, and a deterministic EC2 client
token. Override the token only when an intentional second instance with the
same plan is required.

## Guarded launch

`launch` is also offline unless both safety inputs are present. The
acknowledgement must match exactly:

```bash
python3 infra/aws/l4_lifecycle.py launch \
  --region us-west-2 \
  --subnet-id subnet-0123456789abcdef0 \
  --security-group-id sg-0123456789abcdef0 \
  --key-name colpali-benchmark \
  --associate-public-ip \
  --execute \
  --acknowledge-cost "I UNDERSTAND THIS CREATES A BILLABLE EC2 INSTANCE"
```

Execution resolves and verifies the Amazon-owned DLAMI, performs AWS's
permission dry-run, launches exactly one on-demand instance, and writes an
ignored receipt to `artifacts/phase10/aws_launch_receipt.json`. The receipt
contains the immutable AMI, idempotency token, configuration fingerprint,
source commit, user-data digest, instance response, and exact container pins.
It refuses to overwrite an existing receipt.

Cloud-init checks the DLAMI's NVIDIA driver, Docker, and NVIDIA Container
Toolkit; configures the Docker GPU runtime; pulls the digest-pinned base; and
runs a real CUDA import/device check. On the host:

```text
/var/log/colpali-triton-bootstrap.log
/var/lib/colpali-triton/bootstrap-environment.json
/var/lib/colpali-triton/bootstrap.ready
```

The ready file is created last. Its absence means bootstrap did not complete.

## Status and committed-source deployment

Read one exact instance:

```bash
python3 infra/aws/l4_lifecycle.py status \
  --region us-west-2 \
  --instance-id i-0123456789abcdef0
```

After bootstrap and only when a public/private route permits SSH, deploy the
clean committed `HEAD`, build the pinned image, and run CUDA checks:

```bash
infra/aws/deploy_committed_source.sh \
  --host 203.0.113.10 \
  --identity-file /absolute/path/to/key.pem
```

The helper refuses a dirty worktree. It streams `git archive HEAD` rather than
the full working directory, creates a new commit-specific remote directory,
records the commit in the image label, and never deletes an earlier checkout.

To build or run manually on an NVIDIA Linux host:

```bash
docker/build_cuda.sh
docker/run_cuda_checks.sh
```

Pass a custom command after `run_cuda_checks.sh` to use the same GPU container,
for example a later benchmark runner.

## One-command Phase 11 run and retrieval

Once the L4 host is running and bootstrap is ready, the Phase 11 workflow
deploys one clean committed `HEAD`, builds its pinned image, runs every test,
runs the fixed tuning matrix, runs the full canonical benchmark, and attempts
one bounded profiler capture. It then seals every remote artifact with
SHA-256, downloads a tar bundle, verifies every file, and creates a new local
directory under `artifacts/phase11/remote/`.

First print the complete offline plan. This makes no AWS, SSH, or SCP call:

```bash
python3 infra/aws/phase11_remote.py \
  --host 203.0.113.10 \
  --identity-file /absolute/path/to/key.pem \
  --run-id phase11-l4-20260726
```

The plan blocks a dirty worktree, a missing private-key file, an existing
local result directory, and unsafe paths. Omit `--run-id` to derive a unique
UTC-stamped ID from the source commit.

Run that plan with the explicit execution acknowledgement:

```bash
python3 infra/aws/phase11_remote.py \
  --host 203.0.113.10 \
  --identity-file /absolute/path/to/key.pem \
  --run-id phase11-l4-20260726 \
  --execute \
  --acknowledge-run "RUN PHASE 11 ON THIS EXISTING INSTANCE"
```

The remote source and result directories are commit- and run-specific. Neither
is replaced when it already exists. A successful local directory contains the
remote `artifact_manifest.json`, its `COMPLETE.sha256` marker, test and
benchmark records, profiler status, exact host/container metadata, and a
`_retrieval/` receipt plus the verified original bundle.

The profiler probe uses one fixed FP16 ColPali shape. It attempts one Nsight
Compute launch, or a bounded Nsight Systems trace when only that tool is
present. Profiling is capped at 300 seconds and a missing or permission-blocked
profiler is recorded instead of being mistaken for benchmark failure.

This workflow never launches, stops, or terminates an EC2 instance. In
particular, successful retrieval deliberately leaves the host running so the
result can be inspected. Termination remains the separate, twice-confirmed
lifecycle command below.

## Guarded termination

The default is another offline plan:

```bash
python3 infra/aws/l4_lifecycle.py terminate \
  --region us-west-2 \
  --instance-id i-0123456789abcdef0
```

Execution requires the target ID twice. Before any mutation, it describes the
instance and refuses to proceed unless `Project`, `Phase`, and `ManagedBy`
exactly match this component's contract. It then performs an AWS permission
dry-run and writes a local termination receipt:

```bash
python3 infra/aws/l4_lifecycle.py terminate \
  --region us-west-2 \
  --instance-id i-0123456789abcdef0 \
  --execute \
  --confirm-instance-id i-0123456789abcdef0
```

## Primary references

- [AWS G6 instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS public DLAMI parameter lookup](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html)
- [AWS `run-instances` reference](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html)
- [AWS EC2 user data behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)
- [NVIDIA Container Toolkit installation/configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Official PyTorch 2.6 CUDA 12.4 installation matrix](https://pytorch.org/get-started/previous-versions/)
- [Official PyTorch container tags](https://hub.docker.com/r/pytorch/pytorch/tags)
