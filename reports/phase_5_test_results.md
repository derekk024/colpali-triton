# Phase 5 Test Results

## Run: 2026-07-25

Command:

```text
.venv/bin/python -m pytest -vv
```

Exact result:

```text
collected 116 items
116 passed in 23.21s
```

Environment:

- MacBook Pro (`Mac15,6`)
- Apple M3 Pro with 14-core GPU
- 18 GB unified memory
- macOS 15.6.1 (`24G90`), arm64
- Python 3.9.6
- PyTorch 2.6.0
- pytest 8.4.2
- colpali-triton 0.2.0, editable installation
- MPS built and available

The suite includes:

- all phases 1–4 MaxSim value, mask, shape, dtype, validation, gradient,
  gradcheck, tied-maximum, and MPS tests;
- hand-computed and direct-reference tests for Equation 2;
- loss reductions, low-precision promotion, large-value softplus stability,
  tied-negative gradients, non-finite score validation, and MPS backward;
- MaxSim-to-loss integration, including rejection of fully masked training
  documents;
- deterministic synthetic overfitting with variable masks;
- ambient-default-dtype independence;
- canonical JSON configuration and CLI exit-code validation.

The canonical overfit command was also run separately after the suite:

```text
.venv/bin/python scripts/overfit_synthetic.py
```

It exited successfully with `overfit_passed: true`. Detailed measurements are
recorded in [the phase-5 synthetic report](phase_5_synthetic_overfit.md).

No model or dataset was downloaded. The environment reuses the machine's
existing PyTorch installation through `--system-site-packages`; the unrelated
global `alpaca-trade-api`/`websockets` conflict noted in the earlier report
remains outside this project's dependency graph.

