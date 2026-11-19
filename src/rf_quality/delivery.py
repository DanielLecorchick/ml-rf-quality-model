"""Explicitly opt-in generic HTTP alert dispatch using a local SQLite outbox.

This is NOT a verified Vector API client. PSWS/Vector auth and payload contracts
have not been published in the project materials. Map this envelope to the real
contract only after documentation and permission to notify station operators.

No network requests happen unless the caller explicitly passes confirm_send=True.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class DeliverySummary:
    pending: int
    delivered: int
    failed: int
    dry_run: bool


def _init_outbox(con: sqlite3.Connection) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS alert_deliveries (
        alert_id INTEGER NOT NULL, destination TEXT NOT NULL, state TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT,
        delivered_at TEXT, PRIMARY KEY(alert_id, destination))""")


def dispatch_pending_alerts(
    database: Path, *, endpoint: str | None = None, token: str | None = None,
    confirm_send: bool = False, limit: int = 25, max_attempts: int = 3,
    session: requests.Session | None = None,
) -> DeliverySummary:
    """Send locally generated alerts to an explicitly configured HTTPS endpoint.

    The generic JSON envelope is a design placeholder, not a Vector specification.
    Existing delivered records are idempotently skipped. Failed sends are retried
    on subsequent explicit invocations (bounded by max_attempts).
    """
    if limit < 1 or limit > 500 or max_attempts < 1:
        raise ValueError("Invalid limit or max_attempts")
    db = Path(database)
    if not db.is_file():
        raise FileNotFoundError(f"Local station alerts database does not exist: {db}")
    if confirm_send:
        if not endpoint:
            raise ValueError("An explicit endpoint is required before external delivery")
        u = urlparse(endpoint)
        if u.scheme != "https" or not u.hostname or u.username or u.password or u.fragment:
            raise ValueError("External alert endpoint must be an HTTPS URL without embedded credentials")
        if not token:
            raise ValueError("A delivery token must be provided from an environment variable")
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        _init_outbox(con)
        rows = con.execute("""SELECT a.* FROM alerts a
            LEFT JOIN alert_deliveries d ON a.id=d.alert_id AND d.destination=?
            WHERE (d.state IS NULL OR d.state != 'delivered')
              AND COALESCE(d.attempts, 0) < ?
            ORDER BY a.id ASC LIMIT ?""", (endpoint or "unspecified", max_attempts, limit)).fetchall()
        if not confirm_send:
            return DeliverySummary(len(rows), 0, 0, True)
        client = session or requests.Session()
        delivered = failed = 0
        try:
            for row in rows:
                envelope = {
                    "event_type": "psws.station_quality_warning",
                    "event_id": f"local-alert-{row['id']}",
                    "station_id": row["station_id"],
                    "created_at_utc": row["created_at"],
                    "poor_fraction": float(row["poor_fraction"]),
                    "window_count": int(row["window_count"]),
                    "reason": row["reason"],
                }
                error = None
                succeeded = False
                try:
                    response = client.post(
                        endpoint, json=envelope,
                        headers={"Authorization": f"Bearer {token}",
                                 "Idempotency-Key": envelope["event_id"]},
                        timeout=(5, 20),
                    )
                    response.raise_for_status()
                    succeeded = True
                except requests.RequestException as exc:
                    # Do not persist URLs, auth headers, or raw response bodies.
                    error = type(exc).__name__
                con.execute("""INSERT INTO alert_deliveries
                    (alert_id,destination,state,attempts,last_error,delivered_at)
                    VALUES (?, ?, ?, 1, ?, CASE WHEN ? THEN datetime('now') ELSE NULL END)
                    ON CONFLICT(alert_id,destination) DO UPDATE SET
                    state=excluded.state, attempts=alert_deliveries.attempts+1,
                    last_error=excluded.last_error,
                    delivered_at=excluded.delivered_at""",
                    (row["id"], endpoint, "delivered" if succeeded else "failed", error, int(succeeded)))
                con.commit()  # per-alert checkpoint supports partial failures
                delivered += int(succeeded)
                failed += int(not succeeded)
        finally:
            if session is None:
                client.close()
        return DeliverySummary(len(rows), delivered, failed, False)
