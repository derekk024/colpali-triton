#!/usr/bin/env python3
"""Capture the immutable host/container metadata for one Phase 11 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence
from urllib import error, request


IMDS_BASE = "http://169.254.169.254/latest"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _command(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": list(command),
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _imds_identity() -> dict[str, Any]:
    token_request = request.Request(
        f"{IMDS_BASE}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with request.urlopen(token_request, timeout=2) as response:
            token = response.read().decode("utf-8")
        identity_request = request.Request(
            f"{IMDS_BASE}/dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with request.urlopen(identity_request, timeout=2) as response:
            raw = response.read()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("identity document is not a JSON object")
        allowed = (
            "accountId",
            "architecture",
            "availabilityZone",
            "imageId",
            "instanceId",
            "instanceType",
            "privateIp",
            "region",
            "version",
        )
        return {
            "available": True,
            "identity_document": {
                key: value[key] for key in allowed if key in value
            },
        }
    except (
        error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument(
        "--bootstrap-environment",
        type=Path,
        default=Path(
            "/var/lib/colpali-triton/bootstrap-environment.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bootstrap: dict[str, Any]
    if args.bootstrap_environment.is_file():
        bootstrap = {
            "available": True,
            "path": str(args.bootstrap_environment),
            "sha256": _sha256(args.bootstrap_environment),
            "record": json.loads(args.bootstrap_environment.read_text()),
        }
    else:
        bootstrap = {
            "available": False,
            "path": str(args.bootstrap_environment),
        }

    record = {
        "format": "colpali-triton-phase11-host-metadata-v1",
        "captured_at": _utc_now(),
        "source_commit": args.source_commit,
        "run_id": args.run_id,
        "container_image_name": args.image_name,
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "ec2": _imds_identity(),
        "nvidia_smi": _command(
            (
                "nvidia-smi",
                (
                    "--query-gpu=index,name,uuid,pci.bus_id,driver_version,"
                    "memory.total,compute_cap,pstate,"
                    "clocks.current.sm,clocks.current.memory,power.limit"
                ),
                "--format=csv,noheader",
            )
        ),
        "docker_version": _command(
            ("docker", "version", "--format", "{{json .}}")
        ),
        "docker_image": _command(
            ("docker", "image", "inspect", args.image_name)
        ),
        "bootstrap": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
