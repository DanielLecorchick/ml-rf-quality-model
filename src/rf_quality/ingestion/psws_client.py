"""Documented PSWS observation download API client.

Docs: https://pswsnetwork.eng.ua.edu/about/ (Download API Documentation)
No authentication; server documents a 100-request/day limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from urllib.parse import urlencode
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)
API_URL = "https://pswsnetwork.eng.ua.edu/observations/downloadapi/"


@dataclass(frozen=True)
class ObservationQuery:
    start_date: str
    end_date: str
    station_id: str | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    frequency: float | None = None
    instrument_id: int | None = None

    def params(self) -> dict[str, str | int | float]:
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
        except ValueError as exc:
            raise ValueError("Dates must be valid YYYY-MM-DD dates") from exc
        if end < start:
            raise ValueError("end_date must not precede start_date")
        bbox = (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        supplied = [v is not None for v in bbox]
        if self.station_id and any(supplied):
            raise ValueError("Choose station_id OR geographic bounds, not both")
        if not self.station_id and not all(supplied):
            raise ValueError("Supply station_id OR all four geographic bounds")
        p: dict[str, str | int | float] = {
            "start_date": self.start_date, "end_date": self.end_date,
        }
        if self.station_id:
            if not re.fullmatch(r"S\d{6,}", self.station_id):
                raise ValueError("station_id must be in the form S000028")
            p["station_id"] = self.station_id
        else:
            assert all(v is not None for v in bbox)
            lat_min, lat_max, lon_min, lon_max = bbox
            if not (-90 <= lat_min <= lat_max <= 90):
                raise ValueError("Invalid latitude bounds")
            if not (-180 <= lon_min <= lon_max <= 180):
                raise ValueError("Invalid longitude bounds")
            p.update(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        if self.frequency is not None:
            if not 0 <= self.frequency <= 99.999:
                raise ValueError("Frequency must be in MHz, 0–99.999")
            p["frequency"] = self.frequency
        if self.instrument_id is not None:
            if self.instrument_id < 0:
                raise ValueError("instrument_id must be nonnegative")
            p["instrument_id"] = self.instrument_id
        return p

    def identifier(self) -> str:
        p = self.params()
        place = p.get("station_id", "region")
        fingerprint = hashlib.sha256(urlencode(sorted(p.items())).encode()).hexdigest()[:12]
        return f"{place}_{self.start_date}_{self.end_date}_{fingerprint}"


def new_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "CS523-RF-Quality-Research/0.1"
    return session


def download_observations(
    query: ObservationQuery,
    output_dir: Path,
    *,
    session: requests.Session | None = None,
    max_bytes: int = 2_000_000_000,
    overwrite: bool = False,
) -> Path:
    """Stream the PSWS ZIP to disk, validate ZIP magic, and write provenance metadata."""
    params = query.params()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / f"{query.identifier()}.zip"
    if result.exists() and not overwrite:
        if not zipfile.is_zipfile(result):
            raise ValueError(f"Existing result is not a ZIP: {result}")
        LOG.info("Using cached download %s", result)
        return result
    temporary = result.with_suffix(".zip.part")
    client = session or new_session()
    LOG.info("Downloading %s?%s", API_URL, urlencode(params))
    try:
        with client.get(API_URL, params=params, stream=True, timeout=(15, 120)) as resp:
            resp.raise_for_status()
            count = 0
            digest = hashlib.sha256()
            with temporary.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    count += len(chunk)
                    if count > max_bytes:
                        raise ValueError(f"Response exceeded max_bytes={max_bytes}")
                    f.write(chunk)
                    digest.update(chunk)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("API response was not a valid ZIP (check response and filters)")
            temporary.replace(result)
            metadata = {
                "url": API_URL,
                "params": params,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest.hexdigest(),
                "bytes": count,
            }
            result.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
        if session is None:
            client.close()
    return result


def safe_unzip(archive: Path, destination: Path, *, max_total_bytes: int = 5_000_000_000) -> Path:
    """Extract verified ZIP entries while blocking traversal, symlinks, and oversized archives."""
    import shutil
    import stat

    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        if sum(m.file_size for m in members) > max_total_bytes:
            raise ValueError("Uncompressed archive exceeds allowed size")
        for m in members:
            path = (base / m.filename).resolve()
            if not path.is_relative_to(base) or path == base and not m.is_dir():
                raise ValueError(f"Unsafe ZIP entry: {m.filename}")
            if stat.S_ISLNK(m.external_attr >> 16):
                raise ValueError(f"ZIP symlink not allowed: {m.filename}")
        for m in members:
            out = (base / m.filename).resolve()
            if m.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    return destination
