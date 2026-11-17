"""Local-only station monitoring, backed by SQLite; no external notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
import uuid


@dataclass(frozen=True)
class AlertPolicy:
    min_windows: int = 5
    poor_fraction: float = 0.6
    history_windows: int = 20
    cooldown_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.min_windows < 1 or self.history_windows < self.min_windows:
            raise ValueError("Require 1 <= min_windows <= history_windows")
        if not 0 < self.poor_fraction <= 1 or self.cooldown_seconds < 0:
            raise ValueError("Invalid alert threshold or cooldown")


class LocalStationMonitor:
    """Idempotently store predictions and emit local alerts on sustained degradation."""

    def __init__(self, path: Path, policy: AlertPolicy | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or AlertPolicy()
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT NOT NULL, station_id TEXT NOT NULL, timestamp TEXT NOT NULL,
                    score REAL NOT NULL, poor INTEGER NOT NULL CHECK (poor IN (0,1)),
                    PRIMARY KEY (station_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_station_observations ON observations(station_id, timestamp);
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, poor_fraction REAL NOT NULL,
                    window_count INTEGER NOT NULL, reason TEXT NOT NULL
                );
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, station_id: str, score: float, poor: bool,
               window_id: str | None = None, now: datetime | None = None) -> dict:
        if not station_id or len(station_id) > 128:
            raise ValueError("Invalid station ID")
        if not (float("-inf") < float(score) < float("inf")):
            raise ValueError("Nonfinite station score")
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(timezone.utc)
        key = window_id or str(uuid.uuid4())
        with self._lock, self._connect() as con:
            exists = con.execute("SELECT 1 FROM observations WHERE station_id=? AND id=?", (station_id, key)).fetchone()
            if exists:
                summary = self._summary(con, station_id)
                return {**summary, "duplicate": True, "new_alert": None}
            con.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                        (key, station_id, now.isoformat(), float(score), int(poor)))
            summary = self._summary(con, station_id)
            p = self.policy
            new_alert = None
            if (summary["history_count"] >= p.min_windows
                    and summary["poor_fraction"] >= p.poor_fraction):
                recent = con.execute(
                    "SELECT created_at FROM alerts WHERE station_id=? ORDER BY id DESC LIMIT 1",
                    (station_id,)).fetchone()
                can_alert = (recent is None or
                             (now - datetime.fromisoformat(recent[0])).total_seconds() >= p.cooldown_seconds)
                if can_alert:
                    reason = (f"Poor-quality rate {summary['poor_fraction']:.1%} across "
                              f"{summary['history_count']} recent observations")
                    row = con.execute(
                        "INSERT INTO alerts(station_id,created_at,poor_fraction,window_count,reason) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (station_id, now.isoformat(), summary["poor_fraction"],
                         summary["history_count"], reason),
                    )
                    new_alert = {"id": row.lastrowid, "station_id": station_id,
                                 "created_at": now.isoformat(), "reason": reason,
                                 "poor_fraction": summary["poor_fraction"]}
            return {**summary, "duplicate": False, "new_alert": new_alert}

    def _summary(self, con: sqlite3.Connection, station_id: str) -> dict:
        rows = con.execute("SELECT poor FROM observations WHERE station_id=? "
                           "ORDER BY rowid DESC LIMIT ?",
                           (station_id, self.policy.history_windows)).fetchall()
        n = len(rows)
        return {"station_id": station_id, "history_count": n,
                "poor_count": sum(row[0] for row in rows),
                "poor_fraction": sum(row[0] for row in rows) / n if n else 0.0,
                "flagged": n >= self.policy.min_windows and
                           sum(row[0] for row in rows) / n >= self.policy.poor_fraction}

    def station(self, station_id: str) -> dict:
        with self._connect() as con:
            return self._summary(con, station_id)

    def alerts(self, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be 1..500")
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            return [dict(x) for x in con.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]
