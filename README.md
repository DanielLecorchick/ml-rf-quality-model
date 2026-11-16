# ML-RF-Quality-Model — CS523 Team 4

Starter research code for **binary RF observation quality classification** using the University of
Alabama PSWS/GRAPE network. This is **quality assessment, not RF signal-type recognition**.

The project ships working Python integration code for the **five requested models**:

| Model | Upstream implementation | Input | Label use |
|---|---|---|---|
| XGBoost | `xgboost.XGBClassifier` | 18 RF signal features | Binary labels |
| InceptionTime | `tsai.models.InceptionTime` (PyTorch) | (N,2,T) I/Q | Binary labels |
| MultiROCKET | `aeon.MultiRocketClassifier` | (N,2,T) I/Q | Binary labels |
| Convolutional autoencoder | Small original PyTorch baseline in `models/adapters.py` | (N,2,T) I/Q | Train on acceptable-only |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | 18 RF signal features | Train on acceptable-only |

An optional `dask` extra enables threaded parallel feature extraction (`rf-quality build --feature-workers 4`).

The **data downloader** calls the [official PSWS REST API](https://pswsnetwork.eng.ua.edu/about/) (`GET /observations/downloadapi/`). It validates query arguments, streams ZIP files, records SHA-256 provenance, rejects invalid ZIPs, and optionally safely extracts archives. No credentials are required according to the public API documentation, but the site documents a **100 requests/day** limit. The endpoint has not been live-tested in the artifact-building environment because outbound network connectivity is unavailable here.

## Quick start

Python 3.11 or 3.12 recommended, ideally on Linux/WSL. Use `uv` if available:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,xgboost,rocket,inception,drf]'
rf-quality doctor
```

If you use pip instead:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev,xgboost,rocket,inception,drf]'
```

The `digital-rf` package may require system-level HDF5 libraries depending on the platform. For a small smoke test without installing every optional model, install `pip install -e '.[dev,xgboost,autoencoder]'`.

### A. Test the complete harness WITHOUT claiming real data quality

```bash
rf-quality synthetic --out data/processed/synthetic.npz
rf-quality train --data data/processed/synthetic.npz \
  --models xgboost isolation_forest autoencoder --epochs 2 --out results/synthetic-smoke
# With all optional libraries installed:
rf-quality train --data data/processed/synthetic.npz --models all \
  --epochs 5 --rocket-kernels 2000 --out results/all-synthetic
```

Results contain `split.json` (held-out **stations**), per-model JSON, per-window CSV predictions,
and `summary.csv`. **Never report synthetic-data scores as PSWS performance.** Missing optional
model dependencies are reported as `missing_dependency` with installation guidance; add `--strict`
to fail instead.

### B. Download actual PSWS observations

The following **station/date combination is from the PSWS documentation**, not a guarantee that a
matching downloadable archive will remain available:

```bash
# Print/validate API request without contacting the server.
rf-quality ingest --station S000028 --start 2022-12-09 --end 2022-12-10 --frequency 10 --dry-run

# Actual request; extract after download.
rf-quality ingest --station S000028 --start 2022-12-09 --end 2022-12-10 \
  --frequency 10 --out data/raw --extract
```

Alternative region query:

```bash
rf-quality ingest --bbox 32.0 35.0 -88.0 -84.0 \
  --start 2024-01-01 --end 2024-01-02 --dry-run
```

Download outcomes: **404** means no matching observations; **429** means rate-limited. The
client retries some temporary failures. Downloaded ZIPs receive matching `.json` provenance
manifests. For nested ZIP archives, extract the inner ZIP(s) into the same `data/extracted/` tree
before running `build`. Do not commit large raw or derived observations.

### C. Convert downloaded Digital RF into unlabeled windows

The Digital RF reader expects a root **above** a `ch0/` directory (the code discovers roots).
It reads the actual sample rate from DRF channel metadata and skips missing/unreadable windows,
instead of zero-filling discontinuities. Observations with different samples/window are currently
skipped when they conflict with the first observed rate. The default is 60-second, non-overlapping
windows; change `--seconds`, `--stride-seconds` to explore other durations.

```bash
rf-quality build --root data/extracted --out data/processed/psws.npz \
  --seconds 60 --stride-seconds 60 --max-windows 2000
rf-quality label-template --data data/processed/psws.npz --out data/labels/review.csv
```

The NPZ contains `signals`, `features`, `feature_names`, `station_ids`, `observation_ids`,
`window_ids`, and `labels` (initially **-1**, meaning unknown).

### D. Review quality labels and train

Manually enter `0` (acceptable) or `1` (poor) in the `label` column of `review.csv`. Leave
uncertain observations blank; blank windows are excluded from supervised benchmarking.
Do **not** treat frequency shifts or novel geophysical events as station faults without review.

```bash
rf-quality apply-labels --data data/processed/psws.npz \
  --labels data/labels/review.csv --out data/processed/psws_labeled.npz
rf-quality train --data data/processed/psws_labeled.npz \
  --models all --out results/psws --epochs 10
```

Evaluation requires a sufficient number of labeled **stations** to produce station-disjoint
training/testing sets containing both classes. If you have only one or two stations, gather more
before claiming generalization. Anomaly thresholds use only normal **training** windows;
anomaly scores are not calibrated probabilities. The deep models use global training-set channel
normalization, preserving relative amplitude instead of independently normalizing each window.

## Repo layout

```text
src/rf_quality/
  ingestion/psws_client.py         # public PSWS download API, provenance, safe unzip
  ingestion/digital_rf_reader.py   # real DRF channel reader + window iteration
  features.py                      # RF features for XGBoost + Isolation Forest
  dataset.py                       # NPZ data contract and serialization
  synthetic.py                     # demo generator ONLY; explicit synthetic origin
  labels.py                        # reviewer CSV export/import
  models/adapters.py               # five models, common fit/score contract
  evaluation.py                    # station-disjoint split and metric exports
  cli.py                           # ingest/build/synthetic/labels/train/doctor
scripts/fetch_upstream_sources.sh   # clone source repos under third_party/
third_party/sources.json            # source URLs and provenance
third_party/README.md              # license notes
```

## Pull the *actual upstream source repositories*

Model architectures are **not rewritten from scratch** in this repo; their actual code comes
from installed upstream packages. To inspect and pull their full source repos locally, run:

```bash
# Lightweight priority sources
bash scripts/fetch_upstream_sources.sh HamSCI_DRF_processing InceptionTime_original tsai aeon
# Full set (also fetches XGBoost, scikit-learn, digital-rf source)
bash scripts/fetch_upstream_sources.sh
```

Clones are shallow, stored in ignored `third_party/` directories, and retain original licenses.
**Note:** this distribution does not bundle the upstream Git repositories, because GitHub cannot
be accessed from the environment used to generate this artifact. Cloning them on your own
machine with the above script is necessary if you want local copies of all upstream sources.
The Git repositories are NOT required for runtime if you install the published dependencies.
The HamSCI/DRF_processing repo is GPL-3.0; avoid copying its source into a differently licensed
project without checking GPL obligations.

## Reproducibility and outstanding work

* Status: ingestion client and all model adapters are coded; synthetic tests validate the local
  components available at build time. **No live PSWS request or real Digital RF archive validation
  has been completed here**. A real-file smoke test is the first next step.
* Dataset: RF windows are complex I/Q split into 2 real channels. They remain in original units.
  Feature estimates are exploratory proxies, not reference-calibrated SNR measurements.
* Validation: hold out stations. Also add time-based held-out periods for within-station
  monitoring, multiple seeded repeats, class imbalance checks, precision-recall operating points,
  and expert-reviewed test labels before a final paper.
* API serving and Vector alerting are **not yet implemented**. Do not deploy alerting before
  assessing false-positive rates on unseen stations and geophysical events.
* Source provenance: [PSWS API docs](https://pswsnetwork.eng.ua.edu/about/),
  [Digital RF Python library](https://pypi.org/project/digital-rf/),
  [HamSCI DRF examples](https://github.com/HamSCI/DRF_processing),
  [original InceptionTime paper](https://arxiv.org/abs/1909.04939),
  [tsai PyTorch implementation](https://timeseriesai.github.io/tsai/models.inceptiontime.html),
  [aeon MultiROCKET](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MultiRocketClassifier.html),
  [XGBoost](https://xgboost.readthedocs.io/), [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
