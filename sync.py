"""Synchronizacja przyrostowa: pobiera z Garmina tylko to, czego nie ma w bazie.

- aktywności: baza pamięta przedział [act_from, act_to], który już pobraliśmy;
  dociągamy tylko brakujący początek i świeży koniec,
- HRV i tętno spoczynkowe: zapytania zakresowe (HRV po 28 dni, RHR do roku),
  każdy dzień raz; ostatnie RECENT_DAYS dni odświeżamy, bo mogą się zmienić,
- waga/skład ciała i prognozy biegowe Garmina: zakresowo (do roku na raz),
- profil (data urodzenia, wzrost, płeć): raz na PROFILE_REFRESH_S,
- szczegóły aktywności: raz na zawsze.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import store
from garmin_session import BudgetExceeded, FATAL, Gateway

log = logging.getLogger("biegam.sync")

RECENT_DAYS = 3
REFRESH_S = 15 * 60
HRV_CHUNK_DAYS = 28
RHR_CHUNK_DAYS = 365
YEAR_CHUNK_DAYS = 365
PROFILE_REFRESH_S = 7 * 24 * 3600
DAY = timedelta(days=1)


@dataclass
class SyncReport:
    calls: int = 0
    partial: bool = False
    notes: list[str] = field(default_factory=list)


def _d(s) -> date | None:
    return date.fromisoformat(s) if s else None


def sync_range(gw: Gateway, email: str, start: date, end: date, daily_start: date, report: SyncReport) -> None:
    """Uzupełnia bazę dla analizy [start, end] (+ dane dzienne od daily_start)."""
    end = min(end, date.today())
    try:
        _sync_activities(gw, email, start, end)
        _sync_daily(gw, email, "rhr", daily_start, end, RHR_CHUNK_DAYS, _fetch_rhr)
        _sync_daily(gw, email, "hrv", daily_start, end, HRV_CHUNK_DAYS, _fetch_hrv)
        _sync_daily(gw, email, "weight", daily_start, end, YEAR_CHUNK_DAYS, _fetch_weight)
        _sync_daily(gw, email, "pred", daily_start, end, YEAR_CHUNK_DAYS, _fetch_pred)
        _sync_profile(gw, email)
    except BudgetExceeded:
        report.partial = True
        report.notes.append("Limit zapytań na jedną analizę wyczerpany – reszta danych dociągnie się przy następnej.")
    finally:
        report.calls = gw.calls


def sync_details(gw: Gateway, email: str, activity_ids: list[int], report: SyncReport, max_new: int | None = None) -> None:
    """Pobiera brakujące szczegóły w podanej kolejności (najwyżej `max_new` nowych)."""
    missing = store.missing_details(email, activity_ids)
    if max_new is not None:
        missing = missing[:max_new]
    try:
        for act_id in missing:
            try:
                store.save_details(email, act_id, gw.call("get_activity_details", act_id) or {})
            except FATAL:
                raise
            except BudgetExceeded:
                raise
            except Exception:
                log.warning("Pominięto szczegóły aktywności %s", act_id, exc_info=True)
    except BudgetExceeded:
        report.partial = True
        report.notes.append("Część szczegółów aktywności dociągnie się przy następnej analizie.")
    finally:
        report.calls = gw.calls


# --- aktywności -----------------------------------------------------------

def _sync_activities(gw: Gateway, email: str, start: date, end: date) -> None:
    today = date.today()
    cov_from, cov_to = _d(store.get_meta(email, "act_from")), _d(store.get_meta(email, "act_to"))
    stale = time.time() - store.get_meta(email, "act_tail_at", 0) > REFRESH_S

    ranges: list[tuple[date, date]] = []
    if cov_from is None:
        ranges.append((start, end))
    else:
        if start < cov_from:
            ranges.append((start, cov_from - DAY))
        # Koniec dociągamy od (cov_to - RECENT_DAYS), żeby przedział w bazie był ciągły.
        if end > cov_to or (end >= today - timedelta(days=RECENT_DAYS) and stale):
            ranges.append((max(cov_to - timedelta(days=RECENT_DAYS), cov_from), end))

    for a, b in ranges:
        acts = gw.call("get_activities_by_date", a.isoformat(), b.isoformat()) or []
        store.save_activities(email, acts)
        cov_from = min(cov_from or a, a)
        cov_to = max(cov_to or b, b)
        store.set_meta(email, "act_from", cov_from.isoformat())
        store.set_meta(email, "act_to", cov_to.isoformat())
        if b >= today - timedelta(days=RECENT_DAYS):
            store.set_meta(email, "act_tail_at", time.time())
        log.info("Aktywności %s..%s: %d", a, b, len(acts))


# --- dane dzienne ---------------------------------------------------------

def _missing_spans(email: str, fld: str, start: date, end: date) -> list[tuple[date, date]]:
    fetched = store.daily_fetch_times(email, start.isoformat(), end.isoformat(), fld)
    recent_from = date.today() - timedelta(days=RECENT_DAYS)
    now = time.time()
    spans, cur = [], None
    d = start
    while d <= end:
        t = fetched.get(d.isoformat())
        need = t is None or (d >= recent_from and now - t > REFRESH_S)
        if need:
            cur = (cur[0], d) if cur else (d, d)
        elif cur:
            spans.append(cur)
            cur = None
        d += DAY
    if cur:
        spans.append(cur)
    return spans


def _chunks(a: date, b: date, size: int):
    while a <= b:
        c = min(b, a + timedelta(days=size - 1))
        yield a, c
        a = c + DAY


def _sync_daily(gw, email, fld, start, end, chunk, fetcher) -> None:
    for a, b in _missing_spans(email, fld, start, end):
        for ca, cb in _chunks(a, b, chunk):
            try:
                values = fetcher(gw, ca, cb)
            except (*FATAL, BudgetExceeded):
                raise
            except Exception:
                # Nie zapisujemy nic – spróbujemy przy następnej analizie.
                log.warning("Nie udało się pobrać %s %s..%s", fld, ca, cb, exc_info=True)
                continue
            # Dni, dla których Garmin nic nie zwrócił, też zapisujemy (None),
            # żeby nie pytać o nie w kółko.
            d, all_days = ca, {}
            while d <= cb:
                all_days[d.isoformat()] = values.get(d.isoformat())
                d += DAY
            store.save_daily(email, fld, all_days)
            log.info("%s %s..%s: %d dni z danymi", fld.upper(), ca, cb, len(values))


def _fetch_rhr(gw: Gateway, a: date, b: date) -> dict[str, float]:
    rows = gw.call("get_rhr_daily", a.isoformat(), b.isoformat()) or []
    return {r["calendarDate"]: r["value"] for r in rows if r.get("calendarDate") and r.get("value")}


def _fetch_hrv(gw: Gateway, a: date, b: date) -> dict[str, float]:
    try:
        res = gw.call("get_hrv_data_range", a.isoformat(), b.isoformat())
    except FATAL:
        raise
    except BudgetExceeded:
        raise
    except Exception:
        log.warning("Zakresowe HRV nie zadziałało dla %s..%s", a, b, exc_info=True)
        res = None
    summaries = res.get("hrvSummaries") if isinstance(res, dict) else None
    if isinstance(summaries, list):
        return {
            s["calendarDate"]: s["lastNightAvg"]
            for s in summaries
            if isinstance(s, dict) and s.get("calendarDate") and s.get("lastNightAvg")
        }
    # Nieznany format odpowiedzi: awaryjnie dzień po dniu (w ramach budżetu).
    log.warning("Nieoczekiwana odpowiedź zakresowego HRV – pobieram dzień po dniu")
    out, d = {}, a
    while d <= b:
        try:
            r = gw.call("get_hrv_data", d.isoformat())
            if isinstance(r, dict):
                avg = (r.get("hrvSummary") or {}).get("lastNightAvg") or r.get("lastNightAvg")
                if avg:
                    out[d.isoformat()] = avg
        except (*FATAL, BudgetExceeded):
            raise
        except Exception:
            log.debug("Brak HRV dla %s", d, exc_info=True)
        d += DAY
    return out


def _fetch_weight(gw: Gateway, a: date, b: date) -> dict[str, dict]:
    res = gw.call("get_body_composition", a.isoformat(), b.isoformat()) or {}
    rows = res.get("dateWeightList") if isinstance(res, dict) else res
    out: dict[str, dict] = {}
    for r in rows or []:
        d, w = r.get("calendarDate"), r.get("weight")
        if not d or not w:
            continue
        mm = r.get("muscleMass")
        # Garmin podaje masę w gramach; kilka ważeń dziennie -> zostaje ostatnie.
        out[d] = {
            "weight": w / 1000 if w > 1000 else w,
            "body_fat": r.get("bodyFat"),
            "muscle_mass": (mm / 1000 if mm and mm > 1000 else mm),
        }
    return out


def _fetch_pred(gw: Gateway, a: date, b: date) -> dict[str, dict]:
    res = gw.call("get_race_predictions", a.isoformat(), b.isoformat(), "daily")
    rows = res if isinstance(res, list) else (res or {}).get("racePredictions") or []
    out = {}
    for r in rows:
        d = r.get("calendarDate")
        if d and r.get("time10K"):
            out[d] = {"pred_5k": r.get("time5K"), "pred_10k": r.get("time10K"), "pred_hm": r.get("timeHalfMarathon")}
    return out


def _sync_profile(gw: Gateway, email: str) -> None:
    if time.time() - store.get_meta(email, "profile_at", 0) < PROFILE_REFRESH_S:
        return
    try:
        res = gw.call("get_user_profile") or {}
    except (*FATAL, BudgetExceeded):
        raise
    except Exception:
        log.warning("Nie udało się pobrać profilu", exc_info=True)
        return
    u = res.get("userData") or {}
    store.set_meta(email, "profile", {
        "birth_date": u.get("birthDate"),
        "height_cm": u.get("height"),
        "gender": u.get("gender"),
        "vo2max_running": u.get("vo2MaxRunning"),
        "lthr": u.get("lactateThresholdHeartRate"),
    })
    store.set_meta(email, "profile_at", time.time())
