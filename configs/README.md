# Configurations

`synthetic_overfit.json` records the deterministic phase-5 experiment and its
acceptance thresholds.

Phase 6 adds:

- `vidore_phase6.json`: strict dataset/source manifest with full revisions,
  expected LFS metadata, exact row counts, the OCR/qrel join contract, and
  semantic content fingerprints for DocVQA and InfoVQA.
- `text_baselines.json`: chunking, retrieval, BM25Okapi, BGE-M3, seed, and
  evaluation settings. The dense checkpoint revision, weight size, and
  SHA-256 are mandatory and verified by the runner.
- `phase6_macos_arm64_constraints.txt`: the exact dependency versions observed
  for the Apple-arm64 evaluation environment. Use it as a constraints file,
  not as a standalone requirements file.
