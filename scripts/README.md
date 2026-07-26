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

Run the Phase 8 released-checkpoint smoke test on Apple MPS:

```text
.venv-phase7/bin/python scripts/smoke_colpali_mps.py
```

The first run downloads and byte-verifies approximately 5.93 GB of pinned
public base and adapter weights in the ignored artifact cache. It executes one
deterministic generated document page and one query, validates the
multi-vector shapes and norms, computes a real PyTorch MaxSim score, and writes
a provenance-rich record to `artifacts/phase8/colpali_mps_smoke.json`.
Existing output records are protected unless `--overwrite` is passed.

Run one Phase 9 released-ColPali subset evaluation:

```text
.venv-phase7/bin/python scripts/evaluate_colpali_subset.py \
  --task docvqa \
  --local-files-only
```

Use `--task infovqa` for the second fixed task. Each run verifies the pinned
model, adapter, image corpus, queries, and qrels; checks the precommitted
100-page/50-query subset fingerprints; indexes every selected page once; and
evaluates both late-interaction MaxSim and the L2-normalized masked-mean
ablation over the complete selected corpus. Results are written under
`artifacts/phase9/` and protected against replacement by default.
