# Local Test Results

## Run: 2026-07-25

Command:

```text
.venv/bin/python -m pytest -vv
```

Result:

```text
collected 76 items
76 passed in 0.96s
```

Environment:

- MacBook Pro (`Mac15,6`)
- Apple M3 Pro with 14-core GPU
- 18 GB unified memory
- macOS 15.6.1 (`24G90`), arm64
- Python 3.9.6
- PyTorch 2.6.0
- pytest 8.4.2
- MPS built: yes
- MPS available: yes

The run includes an MPS forward/value comparison against CPU and MPS backward
test. All other tests run on CPU. The suite uses only deterministic synthetic
tensors; no model or dataset was downloaded.

Coverage by behavior includes hand-computed values, Cartesian batching,
unbatched and mixed shapes, right, left, and interior masks, negative-only
matches, fully masked inputs, masked NaN padding in forward and backward,
float16, bfloat16, float32, and float64 oracle parity (including the paper's
`D=128` embedding width), noncontiguous inputs, autograd parity, gradcheck, tied
maxima, accumulation dtypes, mixed empty rows, supported dtype/layout/device
validation, alias forwarding, and masked-position gradients.

Packaging checks:

- Editable installation from `pyproject.toml`: passed.
- Python bytecode compilation for `src/` and `tests/`: passed.
- Git whitespace check: passed.

The test environment was created with `--system-site-packages` to reuse the
machine's existing PyTorch installation. A global `pip check` therefore reports
an unrelated pre-existing `alpaca-trade-api`/`websockets` version conflict.
Neither package is a dependency of this project and the conflict did not affect
collection or execution.
