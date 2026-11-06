"""Wspólne fixtury: izolowana baza na test i fałszywy klient Garmin Connect.

Żaden test nie łączy się z Garminem. FakeAccount generuje deterministyczne dane
(biegi, HRV, tętno spoczynkowe, waga, prognozy, profil, szczegóły aktywności)
i zapisuje każde wywołanie, żeby testy mogły sprawdzać liczbę zapytań.
"""
import json
import math
import os
import random
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # Jinja2Templates(directory="templates")
os.environ.setdefault("BIEGAM_DATA_DIR", tempfile.mkdtemp(prefix="biegam_test_"))

import garminconnect  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import garmin_session as gs  # noqa: E402
import main  # noqa: E402
import store  # noqa: E402

TODAY = date.today()
PASSWORD = "dobre-haslo"
EMAIL = "biegacz@example.com"
HR_REST, HR_MAX = 48, 176


def true_vdot(d: date) -> float:
    """Forma w danych testowych: rośnie liniowo o ~3 VDOT przez 200 dni."""
    return 38 + 3 * (1 - (TODAY - d).days / 200)


def speed_for(vdot: float, hr: float) -> float:
    """Odwrotność modelu z metrics.vo2max_from_run: prędkość (m/s) dla VDOT i HR."""
    vo2 = 3.5 + (vdot - 3.5) * (hr - HR_REST) / (HR_MAX - HR_REST)
    a, b, c = 0.000104, 0.182258, -4.60 - vo2
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / 60


WARMUP_S, WALK_AT, WALK_S = 600, 1500, 60


def run_params(act_id: int, drift_pct: float = 3.0) -> dict:
    """Deterministyczny bieg: 10 min rozgrzewki, minuta marszu w 25. min, reszta równo.

    Zwraca strumienie 1/2 Hz (jak przerzedzone dane Garmina) i średnie z całego biegu.
    """
    d = date.fromordinal(act_id // 10)
    rnd = random.Random(act_id)
    hr = rnd.uniform(125, 150)
    v = speed_for(true_vdot(d) + rnd.gauss(0, 0.8), hr)
    dur = rnd.choice([1800, 2700, 3600, 5400])
    # czasem skok czujnika nadgarstkowego: rzadki i za każdym razem inny (200–233)
    max_hr = 200.0 + 11 * ((act_id // 490) % 4) if act_id % 490 == 0 else min(HR_MAX, hr + rnd.uniform(10, 30))
    rows = []
    for t in range(0, dur, 2):
        if t < WARMUP_S:
            sp, h = 0.75 * v, 100 + (hr - 100) * t / WARMUP_S
        elif WALK_AT <= t < WALK_AT + WALK_S:
            sp, h = 1.0, hr
        else:
            sp, h = v, hr * (1 + drift_pct / 100 * t / dur)
        rows.append([t, h, sp, 300.0 if 600 <= t < 900 else 220.0])
    return {"hr": hr, "speed": v, "dur": dur, "max_hr": max_hr, "rows": rows,
            "avg_hr": float(np.mean([r[1] for r in rows])), "avg_speed": float(np.mean([r[2] for r in rows]))}


def other_activities(d: date) -> list[dict]:
    """Treningi inne niż bieg (id = ordinal*10 + 5..8, żeby nie kolidowały z biegami).

    Dni (ordinal % 14): 1 i 8 – wędrówka (czasem tętno ponad limit 160), 3 i 10 – rowerek
    stacjonarny + siłownia, 6 – marsz, 13 – tylko typy pomijane (rower szosowy, stadion).
    """
    o, r = d.toordinal(), d.toordinal() % 14
    base = lambda k, t, hr, mx, dur: {
        "activityId": o * 10 + k, "activityType": {"typeKey": t}, "startTimeLocal": f"{d} 17:00:00",
        "averageHR": hr, "maxHR": mx, "duration": dur, "movingDuration": dur, "averageSpeed": 1.3,
        "aerobicTrainingEffect": 2.0, "anaerobicTrainingEffect": 0.0}
    if r in (1, 8):
        return [base(5, "hiking", 112.0, 165.0 if o % 28 == 1 else 135.0, 7200)]
    if r in (3, 10):
        return [base(6, "indoor_cycling", 118.0, 140.0, 2700), base(7, "strength_training", 100.0, 150.0, 1800)]
    if r == 6:
        return [base(5, "walking", 98.0, 115.0, 3600)]
    if r == 13:
        return [base(5, "cycling", 130.0, 170.0, 5400), base(6, "track_running", 150.0, 175.0, 2400)]
    return []


def _days(a: str, b: str):
    d = date.fromisoformat(a)
    while d <= date.fromisoformat(b):
        yield d
        d += timedelta(days=1)


class _Tokens:
    def dump(self, path):
        Path(path, "garmin_tokens.json").write_text(json.dumps({"di_token": "x"}))


class FakeAccount:
    def __init__(self):
        self.password = PASSWORD
        self.calls: list[tuple] = []
        self.sso: list[str] = []
        self.rate_limit = False
        self.hrv_format = "ok"
        self.drift_pct = 3.0          # dryf tętna w szczegółach biegu

    def kinds(self):
        return [c[0] for c in self.calls]


class FakeGarmin:
    def __init__(self, account: FakeAccount, email: str, password: str):
        self.acc, self.password, self.client = account, password, _Tokens()

    def login(self, tokenstore=None):
        if tokenstore and Path(tokenstore, "garmin_tokens.json").exists():
            self.acc.calls.append(("login_tokens",))
            return None, None
        self.acc.sso.append(self.password)
        if self.password != self.acc.password:
            raise garminconnect.GarminConnectAuthenticationError("bad credentials")
        return None, None

    def _call(self, *args):
        if self.acc.rate_limit:
            raise garminconnect.GarminConnectTooManyRequestsError("429")
        self.acc.calls.append(args)

    # --- aktywności ---
    def get_activities_by_date(self, a, b):
        self._call("acts", a, b)
        out = []
        for d in _days(a, b):
            out.extend(other_activities(d))
            if d.toordinal() % 7 not in (0, 2, 4, 5):
                continue
            for k in range(2 if d.toordinal() % 35 == 0 else 1):   # czasem dwa biegi dziennie
                p = run_params(d.toordinal() * 10 + k)
                out.append({
                    "activityId": d.toordinal() * 10 + k, "activityType": {"typeKey": "running"},
                    "startTimeLocal": f"{d} 0{7 + k}:00:00",
                    # podsumowanie = średnie z CAŁEGO biegu (z rozgrzewką i marszem), jak w Garminie
                    "averageHR": p["avg_hr"], "maxHR": p["max_hr"],
                    "duration": p["dur"] + 60, "movingDuration": p["dur"],
                    "averageSpeed": p["avg_speed"], "averageGradeAdjustedSpeed": p["avg_speed"],
                    "averagePower": 230, "vO2MaxValue": round(true_vdot(d) + 2),
                    "aerobicTrainingEffect": 3.0, "anaerobicTrainingEffect": 0.5,
                })
        return out

    def get_activity_details(self, act_id):
        self._call("details", act_id)
        p = run_params(int(act_id), self.acc.drift_pct)
        return {"metricDescriptors": [{"key": "sumDuration", "metricsIndex": 0}, {"key": "directHeartRate", "metricsIndex": 1},
                                      {"key": "directSpeed", "metricsIndex": 2}, {"key": "directPower", "metricsIndex": 3}],
                "activityDetailMetrics": [{"metrics": row} for row in p["rows"]]}

    # --- dane dzienne ---
    def get_rhr_daily(self, a, b):
        self._call("rhr", a, b)
        return [{"calendarDate": str(d), "value": 49 + (6 if (TODAY - d).days < 4 else 0)} for d in _days(a, b)]

    def get_hrv_data_range(self, a, b):
        self._call("hrv_range", a, b)
        if self.acc.hrv_format != "ok":
            return {"unexpected": True}
        return {"hrvSummaries": [{"calendarDate": str(d), "lastNightAvg": 40 + d.toordinal() % 6} for d in _days(a, b)]}

    def get_hrv_data(self, d):
        self._call("hrv_day", d)
        return {"hrvSummary": {"lastNightAvg": 42}}

    def get_body_composition(self, a, b):
        self._call("weight", a, b)
        rows = [{"calendarDate": str(d), "weight": (84.0 + (TODAY - d).days * 0.02) * 1000,
                 "bodyFat": 24.0, "muscleMass": 35000} for d in _days(a, b) if d.toordinal() % 2 == 0]
        return {"dateWeightList": rows}

    def get_race_predictions(self, a, b, _type):
        self._call("pred", a, b)
        return [{"calendarDate": str(d), "time5K": 1500, "time10K": 3100, "timeHalfMarathon": 7000} for d in _days(a, b)]

    def get_user_profile(self):
        self._call("profile")
        return {"userData": {"birthDate": "1966-03-15", "height": 178.0, "gender": "MALE", "vo2MaxRunning": 42.0}}


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.sqlite3")
    store.init()
    gs._sessions.clear()
    gs._login_times.clear()
    monkeypatch.setattr(gs, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(gs, "JITTER_S", 0)
    yield


@pytest.fixture
def garmin(monkeypatch) -> FakeAccount:
    acc = FakeAccount()
    monkeypatch.setattr(gs, "Garmin", lambda email, password: FakeGarmin(acc, email, password))
    return acc


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def settle(analyze, **kw):
    """Powtarza analizę, aż szczegóły biegów dociągną się do końca (limit na analizę)."""
    for _ in range(8):
        r = analyze(**kw)
        if "details" not in [c[0] for c in analyze.account.calls]:
            return r
    raise AssertionError("szczegóły biegów nie dociągnęły się w 8 analizach")


@pytest.fixture
def analyze(client, garmin):
    """analyze(days=60, end_offset=0, password=..., **extra) -> Response; czyści log wywołań."""
    def run(days=60, end_offset=0, password=PASSWORD, **extra):
        end = TODAY - timedelta(days=end_offset)
        garmin.calls.clear()
        data = {"email": f"  {EMAIL.upper()} ", "password": password,
                "start_date": str(end - timedelta(days=days)), "end_date": str(end), **extra}
        return client.post("/api/analyze", data=data)
    run.account = garmin
    return run
