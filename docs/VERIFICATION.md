# Verification performed in the artifact build environment

Executed with Python 3.13.5 and `PYTHONPATH=src` in the project root.

* `pytest -q`: **10 passed** (API query validation, streamed fake ZIP download, malicious ZIP
  rejection, feature extraction, dataset serialization, labels, synthetic grouped splitting,
  mock Digital RF reads, and XGBoost/Isolation Forest benchmark).
* `rf-quality synthetic ...`: **passed** (created portable NPZ dataset).
* `rf-quality ingest --station S000028 --start 2022-12-09 --end 2022-12-10 --frequency 10 --dry-run`:
  **passed**, produced documented API GET query (no external request sent).
* `rf-quality train ... --models xgboost isolation_forest`: **passed**.
* `rf-quality train ... --models autoencoder --epochs 1`: **passed**.
* `rf-quality train ... --models all --epochs 1`: **passed runner integration**;
  `tsai` (InceptionTime) and `aeon` (MultiROCKET) marked `missing_dependency`, not trained.
* `python -m compileall -q src tests`: **passed**.

Not verified in this environment:

* Real PSWS HTTP response/archive format or live API availability.
* Actual real-world Digital RF/HDF5 file ingestion (mock reader tested).
* Training/InceptionTime imports using installed `tsai` and MultiROCKET using installed `aeon`.
* Packaging installation with Hatchling (not installed offline).
* Accuracy for any real station or scientific validity of annotations.

Use the README's quick start to install extras and execute live smoke tests on your own
computer/WSL. These are explicit limitations, not failures hidden by the test suite.

## v0.2.0 development checks (November 18, 2026)

* `PYTHONPATH=src pytest -q`: **26 passed** locally in the build environment.
* `rf-quality synthetic` + `rf-quality train --models inceptiontime` with an
  offline 6-station synthetic dataset, local `filters=4`, one epoch: **passed**;
  generated an independently reloadable native PyTorch artifact, PR curve and
  confusion matrix. Synthetic scores are not real PSWS evidence.
* `rf-quality review-gallery` on synthetic NPZ: **passed**, 3 preview figures and
  a browser annotation page generated.
* Nested ZIP extraction, CSV template generation, SHA-256 provenance, source
  fingerprints, generic HTTPS alert dispatch opt-in and idempotency: **tested**.
* MultiROCKET on `aeon` and real DRF parsing: **still not executed here**.
* `uv.lock` generation: **blocked by incomplete offline dependency cache**;
  generate on a network-enabled WSL machine and commit it before final experiments.
* Actual Vector API integration: **not possible without its schema and credentials**;
  generic HTTP transport is available but disabled by default.
