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
CREATE TABLE IF NOT EXISTS activity_derived (
    email        TEXT NOT NULL,
    activity_id  INTEGER NOT NULL,
    data         TEXT NOT NULL,          -- JSON z metrics.derive_run (z wersją)
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


# Grupy danych dziennych: każda ma własny znacznik `<grupa>_fetched_at`,
# bo każdą pobieramy osobnym zapytaniem zakresowym.
DAILY_GROUPS = {
    "hrv": ("hrv",),
    "rhr": ("rhr",),
    "weight": ("weight", "body_fat", "muscle_mass"),
    "pred": ("pred_5k", "pred_10k", "pred_hm"),
}
DAILY_COLUMNS = tuple(c for cols in DAILY_GROUPS.values() for c in cols)


def init() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(DATA_DIR, 0o700)
    except OSError:
        pass
    with connect() as db:
        db.executescript(_SCHEMA)
        # Migracja: dokładamy kolumny grup dziennych, których jeszcze nie ma.
        have = {r["name"] for r in db.execute("PRAGMA table_info(daily)")}
        for group, cols in DAILY_GROUPS.items():
            for col in (*cols, f"{group}_fetched_at"):
                if col not in have:
                    db.execute(f"ALTER TABLE daily ADD COLUMN {col} REAL")


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


def load_derived(email: str, activity_ids: list[int]) -> dict[int, dict]:
    if not activity_ids:
        return {}
    with connect() as db:
        rows = db.execute(
            f"SELECT activity_id, data FROM activity_derived WHERE email=? AND activity_id IN ({','.join('?' * len(activity_ids))})",
            (email, *activity_ids),
        ).fetchall()
    return {r["activity_id"]: json.loads(r["data"]) for r in rows}


def save_derived(email: str, activity_id: int, data: dict) -> None:
    with connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO activity_derived(email, activity_id, data) VALUES (?, ?, ?)",
            (email, int(activity_id), json.dumps(data)),
        )


# --- daily (HRV, tętno spoczynkowe) -------------------------------------

def daily_fetch_times(email: str, start: str, end: str, group: str) -> dict[str, float | None]:
    """{data: kiedy pobrano grupę `group`} dla dni, które są w bazie."""
    assert group in DAILY_GROUPS
    with connect() as db:
        rows = db.execute(
            f"SELECT date, {group}_fetched_at AS t FROM daily WHERE email=? AND date BETWEEN ? AND ?",
            (email, start, end),
        ).fetchall()
    return {r["date"]: r["t"] for r in rows}


def save_daily(email: str, group: str, values: dict) -> None:
    """Zapisuje grupę dla podanych dni (także None = „Garmin nie ma danych”).

    `values`: {data: wartość} dla grup jednokolumnowych albo {data: {kolumna: wartość}}.
    """
    cols = DAILY_GROUPS[group]
    now = time.time()
    rows = []
    for d, v in values.items():
        v = v if isinstance(v, dict) else {cols[0]: v}
        rows.append((email, d, *(v.get(c) for c in cols), now))
    names = ", ".join(cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in (*cols, f"{group}_fetched_at"))
    with connect() as db:
        db.executemany(
            f"INSERT INTO daily(email, date, {names}, {group}_fetched_at) "
            f"VALUES (?, ?, {', '.join('?' * len(cols))}, ?) "
            f"ON CONFLICT(email, date) DO UPDATE SET {updates}",
            rows,
        )


def load_daily(email: str, start: str, end: str) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            f"SELECT date, {', '.join(DAILY_COLUMNS)} FROM daily WHERE email=? AND date BETWEEN ? AND ? ORDER BY date",
            (email, start, end),
        ).fetchall()
    return [dict(r) for r in rows]
