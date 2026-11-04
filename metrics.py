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
    # Model startuje od zera: ewm(adjust=False) przyjąłby pierwszy dzień (często
    # trening) jako stan początkowy, więc dokładamy zerowy dzień przed serią.
    padded = pd.concat([pd.Series([0.0]), df["trimp"].reset_index(drop=True)], ignore_index=True)
    df["CTL"] = padded.ewm(alpha=1 / 42, adjust=False).mean().iloc[1:].values
    df["ATL"] = padded.ewm(alpha=1 / 7, adjust=False).mean().iloc[1:].values
    df["TSB"] = (df["CTL"] - df["ATL"]).shift(1)
    df["ramp"] = df["CTL"] - df["CTL"].shift(7)
    return df


# --- strumienie ze szczegółów aktywności -----------------------------------

_TIME_KEYS = ("sumDuration", "sumElapsedDuration", "directTimestamp")


def streams_1hz(details: dict, wanted: dict[str, tuple[str, ...]], max_gap_s: int = 15) -> pd.DataFrame | None:
    """Wybrane strumienie z get_activity_details przeliczone na 1 Hz.

    Garmin zwraca dane przerzedzone (~2000 punktów na bieg), więc odtwarzamy
    oś czasu w sekundach: krótkie luki wypełniamy ostatnią wartością, dłuższe
    (pauza) zostają puste (NaN). `wanted`: {nazwa: (klucze Garmina w kolejności)}.
    """
    desc = {d.get("key"): d.get("metricsIndex") for d in (details or {}).get("metricDescriptors", [])}
    t_key = next((k for k in _TIME_KEYS if k in desc), None)
    cols = {name: next((desc[k] for k in keys if k in desc), None) for name, keys in wanted.items()}
    if t_key is None or all(i is None for i in cols.values()):
        return None
    t_idx = desc[t_key]
    rows_t, rows_v = [], []
    for m in details.get("activityDetailMetrics", []):
        vals = m.get("metrics", [])
        if len(vals) <= t_idx or vals[t_idx] is None:
            continue
        rows_t.append(vals[t_idx] / 1000 if t_key == "directTimestamp" else vals[t_idx])
        rows_v.append([vals[i] if i is not None and i < len(vals) else None for i in cols.values()])
    if len(rows_t) < 2:
        return None
    df = pd.DataFrame(rows_v, columns=list(cols), index=np.round(np.asarray(rows_t, dtype=float) - rows_t[0]).astype(int), dtype=float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.reindex(range(0, int(df.index.max()) + 1)).ffill(limit=max_gap_s)


# --- moc maksymalna (MMP) ---------------------------------------------------

def power_series_1hz(details: dict) -> pd.Series | None:
    """Moc 1 Hz; pauzy liczone jako 0 W, żeby nie zawyżać średnich."""
    df = streams_1hz(details, {"power": ("directPower", "power")})
    if df is None or df["power"].isna().all():
        return None
    return df["power"].fillna(0.0)


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


# --- aerobic decoupling (Pa:HR) -------------------------------------------

DECOUPLING_MIN_S = 40 * 60      # bieg co najmniej 40 min...
DECOUPLING_WARMUP_S = 10 * 60   # ...pierwsze 10 min (rozgrzewka) pomijamy
RUN_MIN_SPEED = 1.4             # m/s (~12 min/km): wolniej = marsz/postój, pomijamy


def aerobic_decoupling(details: dict) -> float | None:
    """Dryf sercowy w %: o ile spada stosunek tempo/tętno w drugiej połowie biegu.

    (EF1 - EF2) / EF1 * 100, gdzie EF = prędkość / HR, liczone po rozgrzewce,
    tylko z odcinków biegowych. < 5% = solidna baza tlenowa dla tego czasu
    wysiłku, 5–10% = do pracy, > 10% = bieg był za długi/za mocny jak na formę
    (albo upał, odwodnienie, podbiegi pod koniec).
    """
    df = streams_1hz(details, {"hr": ("directHeartRate",), "speed": ("directGradeAdjustedSpeed", "directSpeed")})
    if df is None or df["hr"].isna().all() or df["speed"].isna().all():
        return None
    if len(df) < DECOUPLING_MIN_S:
        return None
    df = df.iloc[DECOUPLING_WARMUP_S:]
    run = df[(df["speed"] >= RUN_MIN_SPEED) & (df["speed"] < 10) & (df["hr"] > 60)]
    if len(run) < (DECOUPLING_MIN_S - DECOUPLING_WARMUP_S) * 0.8:
        return None
    half = len(run) // 2
    ef1 = run["speed"].iloc[:half].mean() / run["hr"].iloc[:half].mean()
    ef2 = run["speed"].iloc[half:].mean() / run["hr"].iloc[half:].mean()
    return round(float((ef1 - ef2) / ef1 * 100), 1)


# --- VDOT z równych odcinków biegu -----------------------------------------

STEADY_BLOCK_S = 5 * 60   # blok ciągłego biegu
STEADY_SPEED_CV = 0.10    # maks. zmienność tempa w bloku (interwały, podbiegi odpadają)
STEADY_HR_SHIFT = 6       # maks. zmiana HR (bpm) między 1. a ostatnią minutą bloku
STEADY_MIN_BLOCKS = 2     # co najmniej 10 min równego biegu
# Forma (VDOT, tempo przy HR) tylko z bloków zaczynających się przed tą chwilą
# biegu: 10 min rozgrzewki + 30 min. Dalej zmęczenie (dryf) zaniżałoby wynik –
# to pokazuje osobno wskaźnik dryfu tętna.
FORM_WINDOW_END_S = DECOUPLING_WARMUP_S + 30 * 60


def steady_blocks(details: dict) -> list[tuple[float, float, int]]:
    """Równe odcinki biegu: [(średnia prędkość m/s, średnie HR, start bloku w s)] dla 5-min bloków.

    Pomija rozgrzewkę, marsz i postoje; blok musi mieć stałe tempo i
    ustabilizowane tętno. Nie zależy od HR spoczynkowego/max, więc wynik
    można zapisać w bazie raz na bieg.
    """
    df = streams_1hz(details, {"hr": ("directHeartRate",), "speed": ("directGradeAdjustedSpeed", "directSpeed")})
    if df is None or df["hr"].isna().all() or df["speed"].isna().all():
        return []
    df = df.iloc[min(DECOUPLING_WARMUP_S, len(df) // 4):]
    run = df[(df["speed"] >= RUN_MIN_SPEED) & (df["speed"] < 10) & (df["hr"] > 60)]
    if len(run) < STEADY_BLOCK_S:
        return []
    sp_all, hr_all = run["speed"].to_numpy(), run["hr"].to_numpy()
    # ciągłe fragmenty biegu: przerwa > 5 s (marsz, postój) rozcina fragment
    pieces = np.split(np.arange(len(run)), np.where(np.diff(run.index.values) > 5)[0] + 1)
    out = []
    for piece in pieces:
        for i in range(0, len(piece) - STEADY_BLOCK_S + 1, STEADY_BLOCK_S):
            sl = piece[i:i + STEADY_BLOCK_S]
            sp, hr = sp_all[sl], hr_all[sl]
            if sp.std(ddof=1) / sp.mean() > STEADY_SPEED_CV:
                continue
            if abs(hr[-60:].mean() - hr[:60].mean()) > STEADY_HR_SHIFT:
                continue
            out.append((round(float(sp.mean()), 4), round(float(hr.mean()), 2), int(run.index.values[sl[0]])))
    return out


def form_blocks(blocks, until_s: int | None = FORM_WINDOW_END_S) -> list[tuple[float, float]]:
    """Bloki do oceny formy: tylko te zaczynające się przed `until_s` (None = wszystkie)."""
    out = []
    for b in blocks or []:
        if until_s is None or len(b) < 3 or b[2] < until_s:
            out.append((b[0], b[1]))
    return out


def vo2max_from_blocks(blocks, hr_rest: float, hr_max: float, until_s: int | None = FORM_WINDOW_END_S) -> tuple[float, int] | None:
    """Mediana szacunków VO2max z równych bloków (z okna formy); None przy < STEADY_MIN_BLOCKS."""
    ests = [e for e in (vo2max_from_run(sp, hr, hr_rest, hr_max) for sp, hr in form_blocks(blocks, until_s)) if e == e]
    if len(ests) < STEADY_MIN_BLOCKS:
        return None
    return float(np.median(ests)), len(ests)


def speed_at_hr_from_blocks(blocks, hr_rest: float, hr_max: float, hr_ref: float,
                            until_s: int | None = FORM_WINDOW_END_S) -> float | None:
    """Prędkość (m/s) przy tętnie odniesienia – mediana z równych bloków okna formy."""
    vals = []
    for sp, hr in form_blocks(blocks, until_s):
        if 0.4 <= hrr_fraction(hr, hr_rest, hr_max) <= 0.97:
            v = speed_at_hr(sp, hr, hr_rest, hr_ref)
            if v == v:
                vals.append(v)
    return float(np.median(vals)) if len(vals) >= STEADY_MIN_BLOCKS else None


def steady_vo2max(details: dict, hr_rest: float, hr_max: float, until_s: int | None = FORM_WINDOW_END_S) -> tuple[float, int] | None:
    """Szacunek VO2max tylko z równych, biegowych odcinków jednego biegu."""
    return vo2max_from_blocks(steady_blocks(details), hr_rest, hr_max, until_s)


# --- wyniki z jednego biegu, liczone raz i zapisywane w bazie ---------------

DERIVED_VERSION = 2   # podbij po zmianie wzorów – wyniki przeliczą się same (v2: czas bloku)
MMP_WINDOWS = (10, 30, 60, 300, 1200, 3600)


def derive_run(details: dict) -> dict:
    """Wszystko, czego dashboard potrzebuje ze szczegółów biegu (mały JSON)."""
    p = power_series_1hz(details) if details else None
    mmp = {str(w): round(float(p.rolling(w).mean().max()), 1) for w in MMP_WINDOWS if p is not None and len(p) >= w}
    return {"v": DERIVED_VERSION, "blocks": steady_blocks(details) if details else [],
            "decoupling": aerobic_decoupling(details) if details else None, "mmp": mmp}


def mmp_from_derived(derived_list) -> list[float | None]:
    best = {w: 0.0 for w in MMP_WINDOWS}
    for d in derived_list:
        for w in MMP_WINDOWS:
            best[w] = max(best[w], (d or {}).get("mmp", {}).get(str(w)) or 0.0)
    return [round(best[w], 1) if best[w] > 0 else None for w in MMP_WINDOWS]


def robust_max_hr(values, tolerance: int = 3, confirmations: int = 3) -> int | None:
    """Tętno maksymalne potwierdzone przez kilka biegów.

    Najwyższa wartość, w promieniu ± `tolerance` bpm od której jest co najmniej
    `confirmations` biegów (łącznie z nią) – pojedyncze skoki czujnika nadgarstkowego
    (zwykle różne wartości) jej nie zawyżą.
    Im więcej biegów w historii, tym pewniejsza wartość (kalibruje się sama).
    """
    v = sorted((float(x) for x in values if x is not None and x == x and 100 < float(x) < 230), reverse=True)
    if not v:
        return None
    if len(v) < confirmations:
        return int(round(v[0]))
    for x in v:
        if sum(1 for y in v if abs(y - x) <= tolerance) >= confirmations:
            return int(round(x))
    return int(round(v[0]))


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
