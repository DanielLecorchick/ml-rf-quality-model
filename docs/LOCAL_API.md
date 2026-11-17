# Local scoring API and station alerts

**The service is a local demo, not an operational safety or emergency system.**
It never calls the external Vector API and does not send any alert outside the machine.

Install dependencies with `uv pip install -e '.[all,dev]'` or at least `.[serve,xgboost,dev]`
and `rf-quality train ...` first so that a trusted model artifact exists.

Start on loopback only:

```bash
rf-quality serve \
  --model results/synthetic-validation/seed-42/xgboost/artifact \
  --alert-db results/local_alerts.sqlite --port 8000 \
  --min-windows 5 --poor-fraction 0.6 \
  --history-windows 20 --cooldown-seconds 3600
```

Available routes:

- `GET /health` and `GET /model` for service/model metadata.
- `POST /predict` for one observation; `POST /predict/batch` for 1–64 observations.
- `GET /stations/{station_id}` for recent local station health.
- `GET /alerts?limit=100` for local SQLite event history.
- `GET /docs` for FastAPI's interactive OpenAPI documentation.

Request body (`POST /predict`):

```json
{"station_id":"S000028", "window_id":"S000028:2026-11-16:1200", "i":[0.1,0.2], "q":[0.1,0.2]}
```

**Important:** The two shown samples are only illustrative. The real `i` and `q` arrays must
contain exactly the **trained window length** (e.g., 600 samples per channel), with finite numeric
values, or the API returns HTTP 422. It extracts the 18 RF features automatically; raw Digital RF
files must first be decoded into I/Q windows. The optional `window_id` should be unique per
station/window to suppress accidental repeated alert counts.

Example client using an already prepared NPZ:

```python
import numpy as np
import requests
with np.load("data/processed/synthetic.npz") as d:
    signal = d["signals"][0]
    payload = {"station_id": str(d["station_ids"][0]),
               "window_id": str(d["window_ids"][0]),
               "i": signal[0].tolist(), "q": signal[1].tolist()}
print(requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=15).json())
```

Alert policy uses a sliding window of the latest `history_windows` station predictions and emits
an event when at least `min_windows` exist and the predicted poor fraction is above
`poor_fraction`. A cooldown suppresses repeated events. `window_id` deduplication is persisted
in the local SQLite database; there are no webhooks, emails, or Vector integration.
For a real deployment, add authentication, centralized data controls, independent alert validation,
monitoring, and false-alarm analysis before allowing access outside loopback.
