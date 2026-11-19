"""Machine-readable provenance for complete experiment reruns."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_provenance(data_path: Path, output: Path, *, arguments: dict, grid_path: Path | None = None
                         ) -> Path:
    """Snapshot immutable inputs, environment and Git ref; avoid embedding raw observations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "xgboost", "torch", "aeon", "fastapi", "matplotlib", "digital-rf"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        revision = None
    metadata = {
        "dataset_path": str(Path(data_path).resolve()),
        "dataset_sha256": sha256_file(data_path),
        "grid_sha256": sha256_file(grid_path) if grid_path else None,
        "git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "arguments": {name: str(value) if isinstance(value, Path) else value
                      for name, value in arguments.items()},
    }
    target = output / "run_provenance.json"
    target.write_text(json.dumps(metadata, indent=2, allow_nan=False, default=str) + "\n")
    return target
