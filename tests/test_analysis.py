"""Wynik /api/analyze: kafelki, serie i odporność na brak danych."""
import json

import pytest

from datetime import timedelta

import store
from conftest import EMAIL, TODAY


def test_kpis_from_synthetic_data(analyze):
    r = analyze(days=90, hr_cap=160, waist_cm=98)
    assert r.status_code == 200, r.text
    j = r.json()
    json.dumps(j, allow_nan=False)                     # brak NaN/inf w JSON
    k = j["kpi"]
    assert 38 < k["vdot"] < 43                        # dane generowane z VDOT 38–41
    assert k["own_10k"] and k["own_hm"] and k["garmin_10k"] == "51:40"
    assert k["rhr_alert"] is True                     # ostatnie 4 dni +6 bpm
    assert k["weight_rate"] < 0 and k["bmi"] and k["target_kg"]
    assert k["waist"]["risk"] == "podwyższone"
    assert k["hr_cap"]["cap"] == 160
    assert k["decoupling"] is not None and 0 < k["decoupling"] < 5
    assert j["debug"]["hr_rest"] == 49                # mediana 30 dni, odporna na skok
    assert j["debug"]["hr_max"] == 176                # skoki czujnika (215) pominięte
    assert k["vdot_seg"] > 0 and k["vdot_seg"] <= k["vdot_runs"]


def test_series_are_aligned(analyze):
    j = analyze(days=90).json()
    n = len(j["dates"])
    assert len(set(j["dates"])) == n                   # dwa biegi jednego dnia = jeden wiersz
    for key in ("ctl", "atl", "tsb", "ef", "pace", "hrv_7d", "hrv_upper"):
        assert len(j[key]) == n, key
    f = j["fitness"]
    assert len(f["run_dates"]) == len(f["vdot_run"]) == len(f["pace_ref_run"])
    assert len(f["trend_dates"]) == len(f["vdot_trend"]) == len(f["own_10k"])


def test_ctl_has_warmup(analyze):
    assert analyze(days=60).json()["ctl"][0] > 5      # bez rozbiegu startowałoby od 0


def test_mmp_from_details(analyze):
    assert analyze().json()["mmp_vals"][3] == 300.0


def test_details_filled_gradually_then_cached(analyze, garmin):
    """Szczegóły biegów dociągają się w kilku analizach (limit na analizę), potem 0 zapytań."""
    counts, seg = [], []
    for _ in range(6):
        r = analyze(days=90)
        counts.append(garmin.kinds().count("details"))
        seg.append(r.json()["kpi"]["vdot_seg"])
        if counts[-1] == 0:
            break
    assert counts[0] > 0 and counts[-1] == 0, counts
    assert all(c <= 30 + 5 + 20 for c in counts)     # odcinki + MMP + dryf
    assert seg[-1] == r.json()["kpi"]["vdot_runs"]    # w końcu wszystkie biegi z odcinków
    assert seg == sorted(seg)


def test_segment_vdot_close_to_truth(analyze):
    """Dane testowe mają VDOT ~41 dziś; średnie z całego biegu zaniżają, odcinki nie."""
    for _ in range(4):
        k = analyze(days=60).json()["kpi"]
    assert k["vdot_seg"] == k["vdot_runs"]
    assert k["vdot"] == pytest.approx(40.6, abs=1.0)


def test_missing_health_data_and_profile(analyze):
    analyze()
    store.set_meta(EMAIL, "profile", {})
    with store.connect() as db:
        db.execute("UPDATE daily SET weight=NULL, rhr=NULL, hrv=NULL, pred_10k=NULL, pred_hm=NULL")
        db.execute("DELETE FROM activity_details")
    r = analyze()
    assert r.status_code == 200, r.text
    k = r.json()["kpi"]
    assert k["bmi"] is None and k["weight"] is None and k["garmin_10k"] is None


def test_no_runs_in_range_is_404(analyze):
    # dane testowe nie mają biegów w dni z ordinal % 7 == 1
    offset = next(o for o in range(1, 8) if (TODAY - timedelta(days=o)).toordinal() % 7 == 1)
    r = analyze(days=0, end_offset=offset)
    assert r.status_code == 404, r.text


def test_low_vdot_does_not_crash(analyze):
    # bardzo stare dane testowe dają VDOT < 15 (poza tablicami) – bez prognozy, ale bez błędu
    r = analyze(days=30, end_offset=2000)
    assert r.status_code == 200, r.text
    assert r.json()["kpi"]["own_10k"] is None


def test_derived_results_recomputed_from_db_after_version_bump(analyze, garmin, monkeypatch):
    import metrics
    from conftest import settle
    settle(analyze, days=30)
    with store.connect() as db:
        n_before = db.execute("SELECT COUNT(*) FROM activity_derived").fetchone()[0]
    assert n_before > 0
    monkeypatch.setattr(metrics, "DERIVED_VERSION", metrics.DERIVED_VERSION + 1)
    r = analyze(days=30)
    assert r.status_code == 200 and "details" not in garmin.kinds()   # z bazy, bez Garmina
    with store.connect() as db:
        versions = {json.loads(d)["v"] for (d,) in db.execute("SELECT data FROM activity_derived")}
    assert metrics.DERIVED_VERSION in versions
