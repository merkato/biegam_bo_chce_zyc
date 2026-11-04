"""Czyste funkcje z metrics.py – bez sieci i bazy."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

import metrics as M


# --- VDOT: zgodność z tablicami Danielsa (Daniels' Running Formula) ---------

@pytest.mark.parametrize("vdot, dist, expected", [
    (40, M.KM10, "50:03"), (50, M.KM10, "41:21"), (60, M.KM10, "35:22"),
    (40, M.HM, "1:50:59"), (50, M.HM, "1:31:35"),
    (50, 5000, "19:57"),
])
def test_race_time_matches_daniels_tables(vdot, dist, expected):
    def secs(s):
        parts = [int(p) for p in s.split(":")]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    got = M.fmt_time(M.race_time_from_vdot(vdot, dist))
    assert abs(secs(got) - secs(expected)) <= 10, (got, expected)


def test_vdot_roundtrip():
    t = M.race_time_from_vdot(45, M.KM10)
    assert M.vdot_from_race(M.KM10, t) == pytest.approx(45, abs=0.01)


def test_race_time_rejects_nonsense():
    assert M.race_time_from_vdot(None, M.KM10) is None
    assert M.race_time_from_vdot(float("nan"), M.KM10) is None
    assert M.race_time_from_vdot(5, M.KM10) is None


def test_vo2max_from_run_easy_run():
    # 6:00/km przy 135 bpm (HRR 68%): ok. 40 ml/kg/min
    assert M.vo2max_from_run(1000 / 360, 135, 48, 176) == pytest.approx(40.5, abs=0.3)


@pytest.mark.parametrize("hr", [60, 175.9])   # poza 50–97% HRR -> brak szacunku
def test_vo2max_from_run_outside_valid_range(hr):
    assert math.isnan(M.vo2max_from_run(3.0, hr, 48, 176))


def test_speed_at_hr_is_linear_in_hrr():
    assert M.speed_at_hr(3.0, 128, 48, 88) == pytest.approx(1.5)
    assert math.isnan(M.speed_at_hr(3.0, 50, 48, 130))


# --- obciążenie -------------------------------------------------------------

def test_training_load_time_constants():
    trimp = pd.Series([100.0] * 300)
    load = M.training_load(trimp)
    # model startuje od zera; po 42 dniach stałego obciążenia CTL ≈ 1 - e^-1 ≈ 63%
    assert load["CTL"].iloc[0] == pytest.approx(100 / 42)
    assert load["CTL"].iloc[41] / 100 == pytest.approx(1 - (1 - 1 / 42) ** 42, rel=1e-6)
    assert load["ATL"].iloc[-1] == pytest.approx(100, rel=1e-3)
    assert math.isnan(load["TSB"].iloc[0])     # TSB = wczorajsze CTL - ATL
    assert load["TSB"].iloc[10] == pytest.approx(load["CTL"].iloc[9] - load["ATL"].iloc[9])


# --- strumienie, MMP, decoupling --------------------------------------------

def _details(rows, keys):
    return {"metricDescriptors": [{"key": k, "metricsIndex": i} for i, k in enumerate(keys)],
            "activityDetailMetrics": [{"metrics": r} for r in rows]}


def test_mmp_uses_seconds_not_samples():
    # próbka co 2 s; 300 W przez 5 min (t=600..899), poza tym 220 W, bieg 1 h
    rows = [[t, 300.0 if 600 <= t < 900 else 220.0] for t in range(0, 3600, 2)]
    vals = M.mmp_curve([_details(rows, ["sumDuration", "directPower"])])
    assert vals[3] == pytest.approx(300.0)          # 5 min
    assert vals[4] == pytest.approx(240.0, abs=0.5)  # 20 min = (5x300 + 15x220) / 20
    assert vals[5] is None                           # bieg krótszy niż 1 h (ostatnia próbka t=3598)


def test_power_pause_counts_as_zero():
    rows = [[t, 400.0] for t in range(0, 60)] + [[t, 400.0] for t in range(300, 360)]
    s = M.power_series_1hz(_details(rows, ["sumDuration", "directPower"]))
    assert s.loc[200] == 0.0 and s.loc[70] == 400.0  # luka > 15 s = pauza


def _drift_run(drift_pct, minutes=60, walk_every=None):
    rows = []
    for t in range(0, minutes * 60, 2):
        hr = 135 * (1 + drift_pct / 100 * t / (minutes * 60))
        speed = 0.8 if walk_every and (t // 60) % walk_every == 0 else 2.8
        rows.append([t, hr, speed])
    return _details(rows, ["sumDuration", "directHeartRate", "directSpeed"])


def test_decoupling_zero_for_steady_run():
    assert M.aerobic_decoupling(_drift_run(0)) == pytest.approx(0, abs=0.1)


def test_decoupling_grows_with_cardiac_drift():
    low, high = M.aerobic_decoupling(_drift_run(4)), M.aerobic_decoupling(_drift_run(15))
    assert 0 < low < 5 < high


def test_decoupling_ignores_walking_breaks():
    with_walks = M.aerobic_decoupling(_drift_run(0, walk_every=5))
    assert with_walks == pytest.approx(0, abs=0.1)


def test_decoupling_requires_long_run():
    assert M.aerobic_decoupling(_drift_run(5, minutes=30)) is None
    assert M.aerobic_decoupling({}) is None


# --- zdrowie ----------------------------------------------------------------

def test_age_and_bmi_band():
    assert M.age_on("1960-11-05", date(2026, 11, 4)) == 65
    assert M.age_on("1960-11-04", date(2026, 11, 4)) == 66
    assert M.age_on(None, date(2026, 1, 1)) is None
    assert M.bmi_band(64)[:2] == (18.5, 24.9)
    assert M.bmi_band(65)[:2] == (23.0, 27.0)


@pytest.mark.parametrize("waist, gender, risk", [
    (93, "MALE", "ok"), (94, "MALE", "podwyższone"), (102, "MALE", "wysokie"),
    (79, "FEMALE", "ok"), (85, "FEMALE", "podwyższone"), (88, "FEMALE", "wysokie"),
])
def test_waist_thresholds(waist, gender, risk):
    assert M.waist_assessment(waist, 178, gender)["risk"] == risk


def test_slope_per_week():
    idx = pd.date_range("2026-01-01", periods=29)
    s = pd.Series(80 - np.arange(29) / 7 * 0.5, index=idx)   # -0,5 kg/tydz.
    assert M.slope_per_week(s) == pytest.approx(-0.5)


def test_formatting():
    assert M.fmt_pace(5.5) == "5:30"
    assert M.fmt_pace(4.9999) == "5:00"
    assert M.fmt_time(91.5) == "1:31:30"
    assert M.fmt_pace(None) is None and M.fmt_time(float("inf")) is None


# --- VDOT z równych odcinków i HR max ---------------------------------------

def _real_run(true_vdot, hr=135, minutes=50, warmup=True, walk=True, intervals=False, rest=48, hmax=176):
    """Bieg jak z zegarka: rozgrzewka, marsz w środku; prędkość dobrana do VDOT."""
    vo2 = 3.5 + (true_vdot - 3.5) * (hr - rest) / (hmax - rest)
    a, b, c = 0.000104, 0.182258, -4.60 - vo2
    v = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a) / 60
    rows = []
    for t in range(0, minutes * 60, 2):
        if warmup and t < 600:
            sp, h = 0.75 * v, 100 + (hr - 100) * t / 600
        elif walk and 1500 <= t < 1560:
            sp, h = 1.0, hr
        elif intervals:
            fast = (t // 60) % 2 == 0
            sp, h = (v * 1.3, hr + 15) if fast else (v * 0.7, hr - 10)
        else:
            sp, h = v, hr
        rows.append([t, h, sp])
    det = _details(rows, ["sumDuration", "directHeartRate", "directSpeed"])
    avg_sp = np.mean([r[2] for r in rows]); avg_hr = np.mean([r[1] for r in rows])
    return det, avg_sp, avg_hr


def test_steady_segments_recover_true_vdot_where_average_fails():
    det, avg_sp, avg_hr = _real_run(40)
    est, blocks = M.steady_vo2max(det, 48, 176)
    assert est == pytest.approx(40, abs=0.3) and blocks >= 5
    # średnia z całego biegu (rozgrzewka + marsz) wyraźnie zaniża
    assert M.vo2max_from_run(avg_sp, avg_hr, 48, 176) < 39


def test_steady_segments_reject_intervals_and_short_runs():
    assert M.steady_vo2max(_real_run(40, intervals=True)[0], 48, 176) is None
    assert M.steady_vo2max(_real_run(40, minutes=12)[0], 48, 176) is None
    assert M.steady_vo2max({}, 48, 176) is None


def test_robust_max_hr_ignores_sensor_spike():
    values = [150, 160, 171, 172, 174, 175, 176, 205, 215, 228]
    assert M.robust_max_hr(values) == 176          # 176 potwierdzone przez 175 i 174; skoki się nie potwierdzają
    assert M.robust_max_hr([170, 215]) == 215      # za mało biegów, by odrzucić skok
    assert M.robust_max_hr([None, float("nan")]) is None
