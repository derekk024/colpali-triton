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

Phase 7 adds:

- `colpali_phase7.json`: a strict checkpoint manifest containing the current
  gated Google source reference, the public BF16 ViDoRe runtime base, the
  paper adapter's initial upload revision, artifact sizes and SHA-256 digests,
  and the exact architecture, processing, and LoRA contracts. The public
  runtime base is the reproducible execution path; the Google revision is not
  claimed to be the paper's original training revision.
- `phase7_macos_arm64_constraints.txt`: the exact dependency versions in the
  clean Apple-arm64 Phase 7 test and inference environment. Use it as a
  constraints file, not as a standalone requirements file.
