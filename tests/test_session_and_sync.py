"""Logowanie, tokeny, ochrona przed banem i synchronizacja przyrostowa."""
import time
from datetime import timedelta

import garmin_session as gs
import store
from conftest import EMAIL, PASSWORD, TODAY, settle


def test_first_analysis_uses_range_requests(analyze, garmin):
    r = analyze()
    assert r.status_code == 200, r.text
    k = garmin.kinds()
    assert garmin.sso == [PASSWORD]
    assert k.count("acts") == 1 and k.count("rhr") == 1
    assert k.count("weight") == 1 and k.count("pred") == 1 and k.count("profile") == 1
    assert 3 <= k.count("hrv_range") <= 5           # paczki po 28 dni
    assert "hrv_day" not in k
    assert r.json()["sync"]["calls"] == len(garmin.calls)


def test_repeat_analysis_makes_no_requests(analyze, garmin):
    settle(analyze)
    r = analyze()
    assert r.status_code == 200 and garmin.calls == [] and r.json()["sync"]["calls"] == 0


def test_restart_resumes_from_tokens_without_sso(analyze, garmin):
    settle(analyze)
    gs._sessions.clear()                             # restart procesu
    assert analyze().status_code == 200
    assert garmin.kinds() == ["login_tokens"] and garmin.sso == [PASSWORD]


def test_wrong_password_never_unlocks_saved_tokens(analyze, garmin):
    analyze()
    r = analyze(password="cudze")
    assert r.status_code == 401
    assert garmin.sso[-1] == "cudze" and "login_tokens" not in garmin.kinds()
    assert (gs._token_dir(EMAIL) / "garmin_tokens.json").exists()


def test_password_is_not_stored(analyze):
    analyze()
    with store.connect() as db:
        dump = "\n".join(db.iterdump())
    assert PASSWORD not in dump


def test_login_rate_limit(analyze, garmin):
    for _ in range(gs.LOGIN_LIMIT):
        assert analyze(password="zle").status_code == 401
    r = analyze(password="zle")
    assert r.status_code == 429 and "logowania" in r.json()["detail"]
    assert len(garmin.sso) == gs.LOGIN_LIMIT


def test_rate_limit_starts_persistent_cooldown(analyze, garmin):
    garmin.rate_limit = True
    assert analyze().status_code == 429
    garmin.rate_limit = False
    assert store.get_meta("*", "cooldown_until", 0) > time.time()
    gs._sessions.clear()
    r = analyze()
    assert r.status_code == 429 and "min" in r.json()["detail"] and garmin.calls == []


def test_longer_range_fetches_only_missing_head(analyze, garmin):
    analyze(days=60)
    analyze(days=90)
    acts = [c for c in garmin.calls if c[0] == "acts"]
    warm = 84   # rozbieg CTL
    assert acts == [("acts", str(TODAY - timedelta(days=90 + warm)), str(TODAY - timedelta(days=61 + warm)))]


def test_recent_days_refreshed_after_interval(analyze, garmin):
    settle(analyze)
    store.set_meta(EMAIL, "act_tail_at", time.time() - 3600)
    with store.connect() as db:
        db.execute("UPDATE daily SET hrv_fetched_at = hrv_fetched_at - 3600, rhr_fetched_at = rhr_fetched_at - 3600")
    analyze()
    recent = (str(TODAY - timedelta(days=3)), str(TODAY))
    assert ("acts", *recent) in garmin.calls and ("hrv_range", *recent) in garmin.calls
    assert ("rhr", *recent) in garmin.calls
    assert "weight" not in garmin.kinds()            # waga pobrana niedawno


def test_unknown_hrv_format_falls_back_to_daily(analyze, garmin):
    garmin.hrv_format = "weird"
    assert analyze(days=10).status_code == 200
    assert "hrv_day" in garmin.kinds()


def test_request_budget_returns_partial_result(analyze, garmin, monkeypatch):
    monkeypatch.setattr(gs.Gateway.__init__, "__defaults__", (3,))
    r = analyze()
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.json()["sync"]["partial"] and r.json()["sync"]["calls"] == 3


def test_invalid_input(analyze):
    assert analyze(days=-5).status_code == 422
