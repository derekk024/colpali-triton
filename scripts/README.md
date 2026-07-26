# Scripts

Run the phase-5 synthetic overfit check from the project root:

```text
.venv/bin/python scripts/overfit_synthetic.py
```

Run one Phase 6 complete-corpus OCR-text evaluation:

```text
.venv/bin/python scripts/evaluate_text_baseline.py \
  --task docvqa \
  --baseline bm25 \
  --output artifacts/phase6/docvqa_bm25.json
```

Supported tasks are `docvqa` and `infovqa`; supported baselines are `bm25` and
`dense`. For the dense baseline, select `--device mps`, `cuda`, or `cpu`.
The runner validates the dataset manifest, semantic content fingerprint,
effective encoder configuration, and pinned model artifact. It records exact
quality and performance measurements plus source/config/environment provenance.
Existing result files are protected unless `--overwrite` is passed.
