"""Lokalna baza SQLite: wszystko, co już raz pobraliśmy z Garmina.

Dane z przeszłości są niezmienne (wczorajsze HRV, zakończony bieg), więc
pobieramy je raz. Dashboard czyta stąd, a do Garmina idą tylko braki.
"""
import json
import os
import sqlite3
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("BIEGAM_DATA_DIR", Path(__file__).parent / "data"))
DB_PATH = DATA_DIR / "biegam.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email       TEXT PRIMARY KEY,
    salt        BLOB NOT NULL,
    pw_digest   BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS activities (
    email        TEXT NOT NULL,
    activity_id  INTEGER NOT NULL,
    start_date   TEXT NOT NULL,          -- YYYY-MM-DD (czas lokalny)
    raw          TEXT NOT NULL,          -- JSON z listy aktywności
    PRIMARY KEY (email, activity_id)
);
CREATE INDEX IF NOT EXISTS ix_activities_date ON activities(email, start_date);
CREATE TABLE IF NOT EXISTS activity_details (
    email        TEXT NOT NULL,
    activity_id  INTEGER NOT NULL,
    raw          BLOB NOT NULL,          -- JSON skompresowany zlib
    fetched_at   REAL NOT NULL,
    PRIMARY KEY (email, activity_id)
);
CREATE TABLE IF NOT EXISTS daily (
    email           TEXT NOT NULL,
    date            TEXT NOT NULL,
    hrv             REAL,
    rhr             REAL,
    hrv_fetched_at  REAL,                -- NULL = jeszcze nie pytaliśmy
    rhr_fetched_at  REAL,
    PRIMARY KEY (email, date)
);
CREATE TABLE IF NOT EXISTS meta (
    scope  TEXT NOT NULL,                -- e-mail albo '*' dla całej aplikacji
    key    TEXT NOT NULL,
    value  TEXT,
    PRIMARY KEY (scope, key)
);
"""


def init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(DATA_DIR, 0o700)
    except OSError:
        pass
    with connect() as db:
        db.executescript(_SCHEMA)


@contextmanager
def connect():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with db:  # transakcja: commit albo rollback
            yield db
    finally:
        db.close()


# --- meta ---------------------------------------------------------------

def get_meta(scope: str, key: str, default=None):
    with connect() as db:
        row = db.execute("SELECT value FROM meta WHERE scope=? AND key=?", (scope, key)).fetchone()
    return json.loads(row["value"]) if row else default


def set_meta(scope: str, key: str, value) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO meta(scope, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(scope, key) DO UPDATE SET value=excluded.value",
            (scope, key, json.dumps(value)),
        )


# --- users --------------------------------------------------------------

def get_user(email: str):
    with connect() as db:
        return db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def upsert_user(email: str, salt: bytes, pw_digest: bytes) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO users(email, salt, pw_digest, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(email) DO UPDATE SET salt=excluded.salt, pw_digest=excluded.pw_digest",
            (email, salt, pw_digest, time.time()),
        )


# --- activities ---------------------------------------------------------

def save_activities(email: str, activities: list[dict]) -> int:
    rows = []
    for a in activities:
        act_id, start = a.get("activityId"), str(a.get("startTimeLocal") or "")
        if act_id is None or len(start) < 10:
            continue
        rows.append((email, int(act_id), start[:10], json.dumps(a)))
    with connect() as db:
        db.executemany(
            "INSERT INTO activities(email, activity_id, start_date, raw) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(email, activity_id) DO UPDATE SET start_date=excluded.start_date, raw=excluded.raw",
            rows,
        )
    return len(rows)


def load_activities(email: str, start: str, end: str) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT raw FROM activities WHERE email=? AND start_date BETWEEN ? AND ? ORDER BY start_date",
            (email, start, end),
        ).fetchall()
    return [json.loads(r["raw"]) for r in rows]


# --- activity details ---------------------------------------------------

def missing_details(email: str, activity_ids: list[int]) -> list[int]:
    if not activity_ids:
        return []
    with connect() as db:
        have = {
            r["activity_id"]
            for r in db.execute(
                f"SELECT activity_id FROM activity_details WHERE email=? AND activity_id IN ({','.join('?' * len(activity_ids))})",
                (email, *activity_ids),
            )
        }
    return [i for i in activity_ids if i not in have]


def save_details(email: str, activity_id: int, details: dict) -> None:
    blob = zlib.compress(json.dumps(details).encode(), 6)
    with connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO activity_details(email, activity_id, raw, fetched_at) VALUES (?, ?, ?, ?)",
            (email, int(activity_id), blob, time.time()),
        )


def load_details(email: str, activity_id: int) -> dict | None:
    with connect() as db:
        row = db.execute(
            "SELECT raw FROM activity_details WHERE email=? AND activity_id=?", (email, int(activity_id))
        ).fetchone()
    return json.loads(zlib.decompress(row["raw"])) if row else None


# --- daily (HRV, tętno spoczynkowe) -------------------------------------

def daily_fetch_times(email: str, start: str, end: str, field: str) -> dict[str, float | None]:
    """{data: kiedy pobrano `field`} dla dni, które są w bazie."""
    assert field in ("hrv", "rhr")
    with connect() as db:
        rows = db.execute(
            f"SELECT date, {field}_fetched_at AS t FROM daily WHERE email=? AND date BETWEEN ? AND ?",
            (email, start, end),
        ).fetchall()
    return {r["date"]: r["t"] for r in rows}


def save_daily(email: str, field: str, values: dict[str, float | None]) -> None:
    """Zapisuje wartości (także None = „Garmin nie ma danych”) z czasem pobrania."""
    assert field in ("hrv", "rhr")
    now = time.time()
    with connect() as db:
        db.executemany(
            f"INSERT INTO daily(email, date, {field}, {field}_fetched_at) VALUES (?, ?, ?, ?) "
            f"ON CONFLICT(email, date) DO UPDATE SET {field}=excluded.{field}, {field}_fetched_at=excluded.{field}_fetched_at",
            [(email, d, v, now) for d, v in values.items()],
        )


def load_daily(email: str, start: str, end: str) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT date, hrv, rhr FROM daily WHERE email=? AND date BETWEEN ? AND ? ORDER BY date",
            (email, start, end),
        ).fetchall()
    return [dict(r) for r in rows]
