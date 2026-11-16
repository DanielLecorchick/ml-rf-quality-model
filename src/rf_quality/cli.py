"""Command-line interface: API ingestion -> Digital RF windows -> labels -> five-model benchmark."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .dataset import WindowDataset, assemble_windows
from .evaluation import benchmark
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

    apply = sub.add_parser("apply-labels", help="Merge reviewer labels back into NPZ")
    apply.add_argument("--data", type=Path, required=True)
    apply.add_argument("--labels", type=Path, required=True)
    apply.add_argument("--out", type=Path, default=Path("data/processed/psws_labeled.npz"))

    train = sub.add_parser("train", help="Train and evaluate models on held-out STATIONS")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--models", nargs="+", choices=[*AVAILABLE_MODELS, "all"], default=["all"])
    train.add_argument("--out", type=Path, default=Path("results/demo"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--rocket-kernels", type=int, default=2000)
    train.add_argument("--strict", action="store_true", help="Fail if optional model dependency missing")

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
            safe_unzip(result, dest)
            print(f"Extracted: {dest}")
        return 0
    if args.command == "build":
        from .ingestion.digital_rf_reader import find_observation_roots, iter_drf_windows
        roots = find_observation_roots(args.root, channel=args.channel)
        if not roots:
            raise SystemExit("No Digital RF channel directory found. Extract ZIPs first; inspect nested ZIPs")
        windows, stations, observations, ids = [], [], [], []
        expected_length = None
        for root in roots:
            if len(windows) >= args.max_windows:
                break
            logging.info("Reading Digital RF: %s", root)
            for window in iter_drf_windows(
                root, channel=args.channel, seconds=args.seconds,
                stride_seconds=args.stride_seconds, station_id=args.station,
                max_windows=args.max_windows - len(windows),
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
        ds = assemble_windows(windows, stations, observations, ids, origin="psws",
                              feature_workers=args.feature_workers)
        ds.save(args.out)
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
    if args.command == "apply-labels":
        ds = apply_labels(WindowDataset.load(args.data), args.labels)
        ds.save(args.out)
        print(f"Saved {int((ds.labels >= 0).sum())} labeled windows to {args.out}")
        return 0
    if args.command == "train":
        ds = WindowDataset.load(args.data)
        names = list(AVAILABLE_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
        reports = benchmark(ds, names, args.out, seed=args.seed, epochs=args.epochs,
                            rocket_kernels=args.rocket_kernels,
                            continue_on_missing=not args.strict)
        print(json.dumps(reports, indent=2))
        print(f"Saved results under {args.out}")
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
