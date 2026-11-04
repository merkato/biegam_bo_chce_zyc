"""Wskaźniki formy i zdrowia – czyste funkcje, bez sieci i bez bazy.

Źródła wzorów:
- Daniels & Gilbert (1979): koszt tlenowy biegu i % VO2max utrzymywany przez czas t
  (tablice VDOT),
- Swain (1997): %HRR ≈ %VO2R, gdzie VO2R = VO2 - 3.5 ml/kg/min,
- Banister: TRIMP i model obciążenia CTL/ATL (stałe czasowe 42 i 7 dni),
- WHO: BMI 18.5–24.9; dla osób 65+ przedział najniższej śmiertelności w metaanalizach
  (Winter i in., AJCN 2014) wypada ok. 23–27,
- obwód talii (IDF/WHO): M ≥94 / ≥102 cm, K ≥80 / ≥88 cm; talia/wzrost < 0.5.
"""
import math
from datetime import date

import numpy as np
import pandas as pd

KM10, HM = 10_000, 21_097.5


# --- Daniels / VDOT ---------------------------------------------------------

def vo2_at_speed(v_m_min: float) -> float:
    """Koszt tlenowy biegu (ml/kg/min) przy prędkości w m/min."""
    return -4.60 + 0.182258 * v_m_min + 0.000104 * v_m_min ** 2


def pct_vo2max_for(t_min: float) -> float:
    """Ułamek VO2max możliwy do utrzymania przez t minut wysiłku maksymalnego."""
    return 0.8 + 0.1894393 * math.exp(-0.012778 * t_min) + 0.2989558 * math.exp(-0.1932605 * t_min)


def vdot_from_race(dist_m: float, t_min: float) -> float:
    return vo2_at_speed(dist_m / t_min) / pct_vo2max_for(t_min)


def race_time_from_vdot(vdot: float, dist_m: float) -> float | None:
    """Czas (min) na dystansie przy danym VDOT – bisekcja (VDOT maleje z czasem)."""
    if not vdot or not math.isfinite(vdot) or vdot < 15:
        return None
    lo, hi = dist_m / 1000 * 2.0, dist_m / 1000 * 20.0   # 2:00 .. 20:00 min/km
    for _ in range(60):
        mid = (lo + hi) / 2
        if vdot_from_race(dist_m, mid) > vdot:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hrr_fraction(hr, hr_rest, hr_max):
    return (hr - hr_rest) / (hr_max - hr_rest)


def vo2max_from_run(speed_mps, hr_avg, hr_rest, hr_max) -> float:
    """Szacunek VO2max z biegu submaksymalnego: VO2 przy danym tempie / %HRR.

    Wiarygodny dla równych biegów (bez interwałów), przy %HRR 50–97%.
    """
    if not speed_mps or not hr_avg or hr_max <= hr_rest:
        return np.nan
    f = hrr_fraction(hr_avg, hr_rest, hr_max)
    if not 0.5 <= f <= 0.97:
        return np.nan
    return 3.5 + (vo2_at_speed(speed_mps * 60) - 3.5) / f


def speed_at_hr(speed_mps, hr_avg, hr_rest, hr_ref) -> float:
    """Prędkość przy tętnie odniesienia: zakładamy, że prędkość rośnie liniowo z HR - HRrest."""
    if not speed_mps or not hr_avg or hr_avg - hr_rest < 10:
        return np.nan
    return speed_mps * (hr_ref - hr_rest) / (hr_avg - hr_rest)


def rolling_median_by_time(dates: pd.Series, values: pd.Series, days: int = 42, min_n: int = 3) -> pd.Series:
    """Dla każdego punktu: mediana wartości z ostatnich `days` dni (wymaga min_n punktów)."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates)).sort_index().dropna()
    return s.rolling(f"{days}D", min_periods=min_n).median()


def ewm_by_time(dates, values, halflife_days: float) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates)).sort_index().dropna()
    if s.empty:
        return s
    return s.ewm(halflife=pd.Timedelta(days=halflife_days), times=s.index).mean()


def slope_per_week(series: pd.Series, last_days: int = 28) -> float | None:
    """Nachylenie (jednostek / tydzień) z ostatnich `last_days` dni serii czasowej."""
    s = series.dropna()
    if s.empty:
        return None
    s = s[s.index >= s.index.max() - pd.Timedelta(days=last_days)]
    x = (s.index - s.index.min()).days.values.astype(float)
    if len(s) < 3 or x.max() < 7:
        return None
    return float(np.polyfit(x, s.values, 1)[0] * 7)


# --- obciążenie -------------------------------------------------------------

def training_load(daily_trimp: pd.Series) -> pd.DataFrame:
    """CTL/ATL (Banister, stałe 42/7 dni), TSB (forma na dziś = wczorajsze CTL - ATL), ramp."""
    df = pd.DataFrame({"trimp": daily_trimp})
    df["CTL"] = df["trimp"].ewm(alpha=1 / 42, adjust=False).mean()
    df["ATL"] = df["trimp"].ewm(alpha=1 / 7, adjust=False).mean()
    df["TSB"] = (df["CTL"] - df["ATL"]).shift(1)
    df["ramp"] = df["CTL"] - df["CTL"].shift(7)
    return df


# --- moc maksymalna (MMP) ---------------------------------------------------

def power_series_1hz(details: dict) -> pd.Series | None:
    """Strumień mocy z get_activity_details przeliczony na 1 Hz (dane Garmina są przerzedzone)."""
    desc = {d.get("key"): d.get("metricsIndex") for d in details.get("metricDescriptors", [])}
    p_idx = desc.get("directPower", desc.get("power"))
    t_key = next((k for k in ("sumDuration", "sumElapsedDuration", "directTimestamp") if k in desc), None)
    if p_idx is None or t_key is None:
        return None
    t_idx = desc[t_key]
    t, p = [], []
    for m in details.get("activityDetailMetrics", []):
        vals = m.get("metrics", [])
        if len(vals) > max(p_idx, t_idx) and vals[t_idx] is not None:
            t.append(vals[t_idx] / 1000 if t_key == "directTimestamp" else vals[t_idx])
            p.append(vals[p_idx] if vals[p_idx] is not None else 0.0)
    if len(t) < 2:
        return None
    s = pd.Series(p, index=np.round(np.asarray(t) - t[0]).astype(int))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # do 1 Hz: krótkie luki (przerzedzenie) wypełniamy ostatnią wartością,
    # dłuższe (pauza) zerem, żeby nie zawyżać mocy.
    return s.reindex(range(0, int(s.index.max()) + 1)).ffill(limit=15).fillna(0.0)


def mmp_curve(details_list: list[dict], windows=(10, 30, 60, 300, 1200, 3600)) -> list[float | None]:
    """Najwyższa średnia moc dla każdego okna; None, gdy żaden bieg nie był tak długi."""
    best = {w: 0.0 for w in windows}
    for det in details_list:
        s = power_series_1hz(det) if det else None
        if s is None:
            continue
        for w in windows:
            if len(s) >= w:
                best[w] = max(best[w], float(s.rolling(w).mean().max()))
    return [round(best[w], 1) if best[w] > 0 else None for w in windows]


# --- zdrowie ----------------------------------------------------------------

def age_on(birth_date: str | None, on: date) -> int | None:
    try:
        b = date.fromisoformat(str(birth_date)[:10])
    except (TypeError, ValueError):
        return None
    return on.year - b.year - ((on.month, on.day) < (b.month, b.day))


def bmi_band(age: int | None) -> tuple[float, float, str]:
    if age is not None and age >= 65:
        return 23.0, 27.0, "65+: najniższe ryzyko w badaniach osób starszych ok. 23–27"
    return 18.5, 24.9, "WHO: prawidłowe BMI 18,5–24,9"


def waist_assessment(waist_cm, height_cm, gender) -> dict | None:
    if not waist_cm:
        return None
    male = str(gender or "").upper().startswith("M")
    hi1, hi2 = (94, 102) if male else (80, 88)
    level = "ok" if waist_cm < hi1 else ("podwyższone" if waist_cm < hi2 else "wysokie")
    whtr = round(waist_cm / height_cm, 2) if height_cm else None
    return {"waist_cm": waist_cm, "whtr": whtr, "risk": level,
            "thresholds": f"{'M' if male else 'K'}: <{hi1} cm ok, {hi1}–{hi2 - 1} podwyższone, ≥{hi2} wysokie; talia/wzrost < 0,5"}


def fmt_time(minutes: float | None) -> str | None:
    if minutes is None or not math.isfinite(minutes):
        return None
    s = int(round(minutes * 60))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_pace(min_per_km: float | None) -> str | None:
    if min_per_km is None or not math.isfinite(min_per_km) or min_per_km <= 0:
        return None
    m = int(min_per_km)
    s = int(round((min_per_km - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"
