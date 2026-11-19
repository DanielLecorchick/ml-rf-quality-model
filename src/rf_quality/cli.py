"""Command-line interface: API ingestion -> Digital RF windows -> labels -> five-model benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .dataset import WindowDataset, assemble_windows
from .experiments import run_experiments
from .ingestion.psws_client import ObservationQuery, download_observations, safe_unzip, API_URL
from .labels import apply_labels, create_label_template
from .models.adapters import AVAILABLE_MODELS
from .synthetic import generate_synthetic


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rf-quality", description="CS523 PSWS RF quality harness")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Download ZIP archive from the public PSWS REST API")
    ingest.add_argument("--station", default=None, help="e.g. S000028 (or provide --bbox)")
    ingest.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    ingest.add_argument("--start", required=True, help="YYYY-MM-DD")
    ingest.add_argument("--end", required=True, help="YYYY-MM-DD")
    ingest.add_argument("--frequency", type=float, help="MHz")
    ingest.add_argument("--instrument", type=int, help="Optional instrument ID")
    ingest.add_argument("--out", type=Path, default=Path("data/raw"))
    ingest.add_argument("--extract", action="store_true", help="Extract ZIP to data/extracted/<query ID>")
    ingest.add_argument("--dry-run", action="store_true", help="Print the URL and do not contact PSWS")
    ingest.add_argument("--overwrite", action="store_true")
    ingest.add_argument("--max-mb", type=int, default=2048, help="Maximum download size in MB")

    build = sub.add_parser("build", help="Convert extracted Grape Digital RF into a window dataset")
    build.add_argument("--root", type=Path, required=True, help="Root containing extracted ch0/ trees")
    build.add_argument("--out", type=Path, default=Path("data/processed/psws.npz"))
    build.add_argument("--seconds", type=float, default=60)
    build.add_argument("--stride-seconds", type=float, default=60)
    build.add_argument("--channel", default="ch0")
    build.add_argument("--station", default=None, help="Override station ID (only for a single station)")
    build.add_argument("--max-windows", type=int, default=5000)
    build.add_argument("--gap-report", type=Path, default=None,
                       help="CSV path for unreadable/missing windows (default beside NPZ)")
    build.add_argument("--feature-workers", type=int, default=1, help="Dask threaded workers; requires .[dask]")

    demo = sub.add_parser("synthetic", help="Generate labeled *synthetic* RF data (not real PSWS labels)")
    demo.add_argument("--out", type=Path, default=Path("data/processed/synthetic.npz"))
    demo.add_argument("--stations", type=int, default=8)
    demo.add_argument("--per-station", type=int, default=36)
    demo.add_argument("--length", type=int, default=600)
    demo.add_argument("--seed", type=int, default=42)

    labels = sub.add_parser("label-template", help="Export CSV for reviewer annotations")
    labels.add_argument("--data", type=Path, required=True)
    labels.add_argument("--out", type=Path, default=Path("data/labels/review.csv"))

    gallery = sub.add_parser("review-gallery", help="Create offline RF spectrogram review gallery")
    gallery.add_argument("--data", type=Path, required=True)
    gallery.add_argument("--out", type=Path, default=Path("data/review"))
    gallery.add_argument("--max-windows", type=int, default=100)
    gallery.add_argument("--seed", type=int, default=42)

    apply = sub.add_parser("apply-labels", help="Merge reviewer labels back into NPZ")
    apply.add_argument("--data", type=Path, required=True)
    apply.add_argument("--labels", type=Path, required=True)
    apply.add_argument("--out", type=Path, default=Path("data/processed/psws_labeled.npz"))

    train = sub.add_parser("train", help="Train/tune/save models using train/validation/test splits")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--models", nargs="+", choices=[*AVAILABLE_MODELS, "all"], default=["all"])
    train.add_argument("--out", type=Path, default=Path("results/demo"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--rocket-kernels", type=int, default=2000)
    train.add_argument("--strict", action="store_true", help="Fail if optional model dependency missing")
    train.add_argument("--seeds", nargs="+", type=int, default=None, help="Multiple seeded repeats")
    train.add_argument("--split-mode", choices=["station", "time"], default="station")
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--embargo-seconds", type=float, default=0,
                       help="Temporal split: exclude windows near boundaries")
    train.add_argument("--grid-file", type=Path, help="JSON mapping model name to list of hyperparameter dicts")
    train.add_argument("--threshold-metric", choices=["f1", "balanced_accuracy", "recall_at_precision"],
                       default="f1")
    train.add_argument("--min-precision", type=float, default=0.8,
                       help="For recall_at_precision tuning")
    train.add_argument("--no-save-models", action="store_true")

    pred = sub.add_parser("predict-file", help="Score NPZ windows using a saved trusted model bundle")
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--data", type=Path, required=True)
    pred.add_argument("--out", type=Path, required=True)

    serve = sub.add_parser("serve", help="Run local-only FastAPI prediction/alert service")
    serve.add_argument("--model", type=Path, required=True, help="Directory with manifest.json")
    serve.add_argument("--alert-db", type=Path, default=Path("results/local_alerts.sqlite"))
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--min-windows", type=int, default=5)
    serve.add_argument("--poor-fraction", type=float, default=0.6)
    serve.add_argument("--history-windows", type=int, default=20)
    serve.add_argument("--cooldown-seconds", type=int, default=3600)

    dispatch = sub.add_parser("alert-dispatch", help="Review or explicitly send local alerts via generic HTTPS")
    dispatch.add_argument("--alert-db", type=Path, default=Path("results/local_alerts.sqlite"))
    dispatch.add_argument("--endpoint", default=None,
                          help="Generic HTTPS receiver; NOT a verified Vector API schema")
    dispatch.add_argument("--token-env", default="RF_QUALITY_ALERT_TOKEN",
                          help="Environment variable containing receiver token")
    dispatch.add_argument("--confirm-send", action="store_true",
                          help="Permit actual HTTP delivery (off by default)")
    dispatch.add_argument("--limit", type=int, default=25)

    sub.add_parser("doctor", help="Check which external model and RF packages are available")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if args.command == "ingest":
        bbox = args.bbox or [None, None, None, None]
        q = ObservationQuery(args.start, args.end, args.station, *bbox,
                             frequency=args.frequency, instrument_id=args.instrument)
        params = q.params()  # Validate before accessing the network.
        if args.dry_run:
            from urllib.parse import urlencode
            print(f"GET {API_URL}?{urlencode(params)}")
            print("API limit: documented 100 requests/day; no authentication")
            return 0
        result = download_observations(q, args.out, overwrite=args.overwrite,
                                       max_bytes=args.max_mb * 1024 * 1024)
        print(f"Downloaded: {result}")
        if args.extract:
            dest = args.out.parent / "extracted" / q.identifier()
            from .ingestion.archives import extract_observation_archives
            extracted = extract_observation_archives(result, dest)
            print(f"Extracted {len(extracted)} ZIP layer(s): {dest}")
        return 0
    if args.command == "build":
        from .ingestion.digital_rf_reader import find_observation_roots, iter_drf_windows
        roots = find_observation_roots(args.root, channel=args.channel)
        if not roots:
            raise SystemExit("No Digital RF channel directory found. Extract ZIPs first; inspect nested ZIPs")
        windows, stations, observations, ids, timestamps = [], [], [], [], []
        gaps: list[dict] = []
        expected_length = None
        for root in roots:
            if len(windows) >= args.max_windows:
                break
            logging.info("Reading Digital RF: %s", root)
            for window in iter_drf_windows(
                root, channel=args.channel, seconds=args.seconds,
                stride_seconds=args.stride_seconds, station_id=args.station,
                max_windows=args.max_windows - len(windows), on_gap=gaps.append,
            ):
                if expected_length is None:
                    expected_length = window.iq.shape[1]
                if window.iq.shape[1] != expected_length:
                    logging.warning("Skipping inconsistent sample rate/window size in %s", root)
                    continue
                windows.append(window.iq)
                stations.append(window.station_id)
                observations.append(window.observation_id)
                ids.append(window.window_id)
                timestamps.append(window.timestamp_unix)
        ds = assemble_windows(windows, stations, observations, ids, origin="psws",
                              feature_workers=args.feature_workers, timestamps=timestamps)
        ds.save(args.out)
        import csv
        gap_path = args.gap_report or args.out.with_name(args.out.stem + "_gaps.csv")
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["station_id", "observation_id", "sample_index", "sample_count",
                  "sample_rate_hz", "timestamp_unix", "reason"]
        with gap_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(gaps)
        print(f"Missing/unreadable window audit: {gap_path} ({len(gaps)} windows)")
        print(f"Saved {len(ds.signals)} unlabeled observations to {args.out}; "
              f"shape={ds.signals.shape}. Review labels before training.")
        return 0
    if args.command == "synthetic":
        ds = generate_synthetic(stations=args.stations, per_station=args.per_station,
                                length=args.length, seed=args.seed)
        ds.save(args.out)
        print(f"Saved SYNTHETIC dataset ({len(ds.labels)} windows) to {args.out}")
        return 0
    if args.command == "label-template":
        ds = WindowDataset.load(args.data)
        create_label_template(ds, args.out)
        print(f"Created human-annotation CSV: {args.out} (0=acceptable, 1=poor, blank=unlabeled)")
        return 0
    if args.command == "review-gallery":
        from .annotation import build_review_gallery
        result = build_review_gallery(WindowDataset.load(args.data), args.out,
                                      max_windows=args.max_windows, seed=args.seed)
        print(f"Open {result.resolve()} in a browser; review and export review.csv")
        return 0
    if args.command == "apply-labels":
        ds = apply_labels(WindowDataset.load(args.data), args.labels)
        ds.save(args.out)
        print(f"Saved {int((ds.labels >= 0).sum())} labeled windows to {args.out}")
        return 0
    if args.command == "train":
        ds = WindowDataset.load(args.data)
        names = list(AVAILABLE_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
        grid = json.loads(args.grid_file.read_text()) if args.grid_file else None
        reports = run_experiments(
            ds, names, args.out, seeds=args.seeds or [args.seed], split_mode=args.split_mode,
            validation_fraction=args.validation_fraction, test_fraction=args.test_fraction,
            embargo_seconds=args.embargo_seconds, grid=grid, epochs=args.epochs,
            rocket_kernels=args.rocket_kernels, threshold_metric=args.threshold_metric,
            min_precision=args.min_precision, save_models=not args.no_save_models,
            strict=args.strict,
        )
        from .provenance import write_run_provenance
        write_run_provenance(args.data, args.out, arguments=vars(args), grid_path=args.grid_file)
        print(json.dumps(reports, indent=2))
        print(f"Saved results under {args.out}")
        return 0
    if args.command == "predict-file":
        import csv
        from .artifacts import load_model_bundle, predict_bundle
        ds = WindowDataset.load(args.data)
        model, manifest = load_model_bundle(args.model)
        scored = predict_bundle(model, manifest, ds.signals, ds.features, ds.feature_names)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["window_id", "station_id", "predicted_label", "score"])
            for window_id, station, pred, score in zip(
                ds.window_ids, ds.station_ids, scored.predicted, scored.score
            ):
                writer.writerow([window_id, station, int(pred), float(score)])
        print(f"Saved {len(scored.score)} scored observations to {args.out}")
        return 0
    if args.command == "serve":
        from .alerts import AlertPolicy
        from .api import create_app
        import uvicorn
        policy = AlertPolicy(min_windows=args.min_windows,
                             poor_fraction=args.poor_fraction,
                             history_windows=args.history_windows,
                             cooldown_seconds=args.cooldown_seconds)
        app = create_app(args.model, alert_db=args.alert_db, policy=policy)
        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return 0
    if args.command == "alert-dispatch":
        import os
        from .delivery import dispatch_pending_alerts
        report = dispatch_pending_alerts(
            args.alert_db, endpoint=args.endpoint,
            token=os.environ.get(args.token_env),
            confirm_send=args.confirm_send, limit=args.limit)
        print(json.dumps(report.__dict__, indent=2))
        return 0
    if args.command == "doctor":
        import importlib.util
        for name, module in [("psws downloader", "requests"), ("digital-rf", "digital_rf"),
                             ("xgboost", "xgboost"), ("MultiROCKET/aeon", "aeon"),
                             ("InceptionTime/tsai", "tsai"), ("PyTorch autoencoder", "torch")]:
            print(f"{name:<24} {'available' if importlib.util.find_spec(module) else 'not installed'}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
