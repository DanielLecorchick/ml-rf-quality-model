"""Bounded recursive extraction of ZIPs containing observation ZIPs.

The PSWS API may return an outer archive of per-observation ZIPs. Never trust
archive entry paths, claimed sizes or symlinks, including nested archives.
"""
from __future__ import annotations

from pathlib import Path
import zipfile

from .psws_client import safe_unzip


def extract_observation_archives(
    source_zip: Path, target: Path, *, max_depth: int = 3,
    max_expanded_bytes: int = 5_000_000_000, max_archives: int = 100,
) -> list[Path]:
    """Safely expand outer ZIP and nested ZIPs; return all extracted destinations."""
    if max_depth < 0 or max_archives < 1 or max_expanded_bytes < 1:
        raise ValueError("Invalid extraction limits")
    source_zip = Path(source_zip)
    target = Path(target)
    destinations: list[Path] = []
    remaining = max_expanded_bytes
    queue = [(source_zip, target, 0)]
    seen = 0
    while queue:
        archive, dest, depth = queue.pop(0)
        seen += 1
        if seen > max_archives:
            raise ValueError("Nested ZIP count exceeds safety limit")
        with zipfile.ZipFile(archive) as zf:
            size = sum(i.file_size for i in zf.infolist())
        if size > remaining:
            raise ValueError("Cumulative uncompressed size exceeds safety limit")
        safe_unzip(archive, dest, max_total_bytes=remaining)
        remaining -= size
        destinations.append(dest)
        for nested in sorted(dest.rglob("*.zip")):
            # Do not traverse nested archive folders already expanded in this run.
            if nested == archive:
                continue
            if not zipfile.is_zipfile(nested):
                raise ValueError(f"Nested .zip file is not a valid archive: {nested}")
            if depth >= max_depth:
                raise ValueError("Nested ZIP depth exceeds safety limit")
            nested_dest = nested.with_suffix("")
            if nested_dest.exists():
                raise FileExistsError(f"Nested extraction target already exists: {nested_dest}")
            queue.append((nested, nested_dest, depth + 1))
    return destinations
