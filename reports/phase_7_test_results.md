# Phase 7 Test Results

## Final verification

The Phase 7 suite passed on July 26, 2026 in America/Los_Angeles.

Command:

```text
.venv-phase7/bin/python -m pytest -q
```

Exact result:

```text
388 passed in 4.21s
```

There were 0 failures, 0 skips, 0 xfails, and 0 xpasses. The isolated
environment also passed `python -m pip check`, and all source, test, script,
and benchmark files passed bytecode compilation.

## New Phase 7 coverage

The Phase 7 tests add:

- strict, immutable checkpoint-manifest parsing and canonical fingerprinting;
- revision, artifact-size, SHA-256, schema, duplicate-key, and unknown-key
  validation;
- typed query/document batch contracts and device transfer behavior;
- safe L2 normalization, exact-zero padding, finite-value checks, and
  mask-bound multi-vector encodings;
- shared projection gradients and frozen-backbone behavior;
- query/document role enforcement in late-interaction scoring;
- exact document prompts, query augmentation, dummy-image prefix removal,
  right padding, and processor-output validation;
- exact LoRA configuration, target selection, vision-module exclusion,
  trainability reporting, and frozen-projection ablation behavior;
- a miniature real Transformers PaliGemma forward path with the released
  state-key layout;
- PEFT insertion into a miniature PaliGemma model and an end-to-end adapted
  forward pass;
- verified checkpoint size/hash failure paths and download controls; and
- cross-module guards proving that implementation constants match the
  committed Phase 7 manifest.

All phases 1–6 remain in the same command, including MaxSim value, masking,
batching, gradient, MPS, loss, synthetic-overfit, dataset, metric, baseline,
and runner tests.

## Integration checks outside the unit suite

- The real pinned PaliGemma processor produced a `[1, 1030]` document batch
  with 1,030 active positions and a right-padded augmented query batch.
- Processor IDs matched the expected image token `257152` and `<unused0>`
  token `7`.
- The real pinned adapter matched its 78,625,112-byte and SHA-256 contract.
- Its safetensors file contained exactly 127 LoRA A tensors, 127 LoRA B
  tensors, and 39,292,928 adapter parameters.
- The pinned public-base index exposed the expected projection weight/bias,
  PaliGemma state prefix, hidden width, and image geometry.

These are Phase 7 integration checks and are not counted among the 388 unit
tests. The full released-model MPS forward pass belongs to Phase 8.

## Test environment

- platform: macOS 15.6.1, arm64
- Python 3.9.6
- pytest 8.4.2
- PyTorch 2.6.0
- Transformers 4.42.4
- PEFT 0.11.1
- project 0.4.0, editable installation
- MPS built and available; CUDA unavailable
- `pip check`: no broken requirements
