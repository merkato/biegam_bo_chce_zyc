"""Obliczenia dashboardu na danych z lokalnej bazy (bez sieci).

Zwraca jeden słownik JSON dla frontendu: kafelki KPI (`kpi`), serie dzienne
(obciążenie, HRV, EF), serie per bieg (`fitness`) i zdrowie (`health`).
"""
import math
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

import metrics as M


DECOUPLING_MAX_RUNS = 20   # tyle ostatnich długich biegów z zakresu analizujemy
SEG_DETAILS_PER_ANALYSIS = 30   # nowych szczegółów biegów na jedną analizę (ochrona przed banem)


class NoData(Exception):
    pass


def calculate_trimp(hr_avg, duration_sec, hr_rest, hr_max):
    if pd.isna(hr_avg) or pd.isna(duration_sec) or duration_sec <= 0: return 0
    hr_ratio = max(0, min((hr_avg - hr_rest) / (hr_max - hr_rest), 1))
    return duration_sec / 60.0 * hr_ratio * 0.64 * math.exp(1.92 * hr_ratio)


def get_trend_and_forecast(df, col_name, span=21, forecast_days=21):
    if col_name not in df.columns: return [], [], [], []
    df_clean = df[['date', col_name]].dropna().sort_values('date')
    if df_clean.empty: return [], [], [], []
    min_d, max_d = df_clean['date'].min(), df_clean['date'].max()
    df_cont = df_clean.set_index('date').reindex(pd.date_range(start=min_d, end=max_d))
    df_cont['smoothed'] = df_cont[col_name].interpolate(method='time').ewm(span=span, adjust=False).mean()
    recent = df_cont['smoothed'].dropna().tail(span)
    if len(recent) > 1:
        slope = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        f_dates = pd.date_range(start=max_d + pd.Timedelta(days=1), periods=forecast_days)
        f_vals = [recent.iloc[-1] + slope * i for i in range(1, forecast_days + 1)]
    else:
        f_dates, f_vals = pd.DatetimeIndex([]), []
    return (df_cont.index.strftime('%Y-%m-%d').tolist(), _clean(df_cont['smoothed']),
            f_dates.strftime('%Y-%m-%d').tolist() if len(f_dates) else [], f_vals)


def resting_hr_from_daily(daily_rows: list[dict], default: int = 50) -> int:
    """Mediana tętna spoczynkowego z ostatnich 30 dni z danymi.

    Stabilna norma do TRIMP/VDOT – kilka dni choroby nie przestawia obliczeń
    (chwilowe skoki pokazuje osobno kafelek tętna spoczynkowego).
    """
    vals = [r["rhr"] for r in daily_rows if r.get("rhr")][-30:]
    return int(round(float(np.median(vals)))) if vals else default


def _clean(values) -> list:
    """Lista do JSON: NaN/inf -> None, numpy -> float."""
    out = []
    for v in (values.tolist() if hasattr(values, "tolist") else values):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        else:
            out.append(round(v, 3) if isinstance(v, float) else v)
    return out


def _ds(idx) -> list[str]:
    return pd.DatetimeIndex(idx).strftime('%Y-%m-%d').tolist()


def _col(df, *names):
    return next((c for c in names if c in df.columns), None)


def _last(s: pd.Series):
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _value_days_ago(s: pd.Series, days: int):
    s = s.dropna()
    if s.empty:
        return None
    old = s[s.index <= s.index.max() - pd.Timedelta(days=days)]
    return float(old.iloc[-1]) if len(old) else None


def analyze(
    activities: list[dict],
    daily_rows: list[dict],
    profile: dict,
    start: date,
    end: date,
    daily_start: date,
    user_hr_rest: int | None,
    user_hr_max: int | None,
    hr_cap: int | None,
    waist_cm: float | None,
    get_derived: Callable[..., list[dict | None]],
    max_hr_history: list | None = None,
) -> dict:
    """`activities` obejmują też okres rozruchowy przed `start` (dla CTL).

    `get_derived(ids, max_new=None)` zwraca wyniki z biegów (metrics.derive_run;
    None, gdy szczegóły jeszcze nie pobrane); `max_hr_history` to HR max biegów
    z ostatniego roku w bazie.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    today_ts = pd.Timestamp.today().normalize()

    df = pd.DataFrame(activities)
    if df.empty or 'activityType' not in df.columns:
        raise NoData("Brak aktywności we wskazanym zakresie.")
    df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
    runs = df[df['type_key'].str.contains('running', na=False)].copy()
    runs['date'] = pd.to_datetime(runs['startTimeLocal']).dt.normalize()
    if runs.empty or not (runs['date'] >= start_ts).any():
        raise NoData("Brak biegów w wybranym zakresie dat.")
    runs = runs.sort_values('startTimeLocal')

    # --- fizjologia ---
    hr_rest = user_hr_rest if user_hr_rest is not None else resting_hr_from_daily(daily_rows)
    max_hr_col = _col(runs, 'maxHR', 'maxHeartRateInBeatsPerMinute')
    run_max = pd.to_numeric(runs[max_hr_col], errors='coerce').tolist() if max_hr_col else []
    hr_max_auto = M.robust_max_hr(list(max_hr_history or []) + run_max)
    hr_max = user_hr_max if user_hr_max is not None else (hr_max_auto or 185)
    hr_max = max(hr_max, hr_rest + 20)
    hr_ref = hr_cap or int(round(hr_rest + 0.65 * (hr_max - hr_rest)))

    pow_col = _col(runs, 'averagePower', 'avgPower')
    hr_col = _col(runs, 'averageHR', 'averageHeartRateInBeatsPerMinute', 'averageBpm')
    spd_col = _col(runs, 'averageGradeAdjustedSpeed', 'avgGradeAdjustedSpeed', 'averageSpeed')
    dur_col = _col(runs, 'movingDuration', 'duration')
    vo_col = _col(runs, 'averageVerticalOscillation', 'avgVerticalOscillation')
    gct_col = _col(runs, 'averageGroundContactTime', 'avgGroundContactTime')
    ae_col, an_col = _col(runs, 'aerobicTrainingEffect'), _col(runs, 'anaerobicTrainingEffect')
    vo2_col = _col(runs, 'vO2MaxValue', 'vO2MaxPreciseValue')

    num = lambda c: pd.to_numeric(runs[c], errors='coerce') if c else pd.Series(np.nan, index=runs.index)
    for c in (vo_col, gct_col, ae_col, an_col):
        if c: runs[c] = num(c)
    runs['speed'] = num(spd_col).where(lambda s: s > 0)
    runs['dur'] = num(dur_col)
    runs['hr'] = num(hr_col)
    runs['pow'] = num(pow_col)
    runs['EF'] = runs['pow'] / runs['hr']
    runs['trimp'] = [calculate_trimp(h, d, hr_rest, hr_max) for h, d in zip(runs['hr'], runs['dur'])]

    # wskaźniki per bieg (tylko równe biegi >= 20 min)
    steady = runs['dur'] >= 20 * 60
    runs['vdot_run'] = [M.vo2max_from_run(s, h, hr_rest, hr_max) if ok else np.nan
                        for s, h, ok in zip(runs['speed'], runs['hr'], steady)]
    runs['vdot_src'] = np.where(runs['vdot_run'].notna(), 'avg', None)

    # Lepszy szacunek z równych odcinków (szczegóły biegu). Pobieramy je stopniowo,
    # od najnowszych, najwyżej SEG_DETAILS_PER_ANALYSIS nowych na analizę – reszta
    # biegów do czasu pobrania korzysta ze średniej z całego biegu.
    seg_cand = runs[steady & (runs['date'] >= start_ts - pd.Timedelta(days=42))]
    if 'activityId' in runs:
        seg_cand = seg_cand.dropna(subset=['activityId']).sort_values('startTimeLocal', ascending=False)
        seg_ids = [int(i) for i in seg_cand['activityId']]
        for idx, der in zip(seg_cand.index, get_derived(seg_ids, max_new=SEG_DETAILS_PER_ANALYSIS)):
            res = M.vo2max_from_blocks(der["blocks"], hr_rest, hr_max) if der else None
            if res:
                runs.at[idx, 'vdot_run'], runs.at[idx, 'vdot_src'] = res[0], 'seg'

    hrr = M.hrr_fraction(runs['hr'], hr_rest, hr_max)
    ok_ref = (runs['dur'] >= 15 * 60) & hrr.between(0.4, 0.97)
    runs['spd_ref'] = [M.speed_at_hr(s, h, hr_rest, hr_ref) if ok else np.nan
                       for s, h, ok in zip(runs['speed'], runs['hr'], ok_ref)]

    # --- obciążenie: od pierwszego biegu (rozruch) do +21 dni ---
    runs['sd'] = runs['speed'] * runs['dur']
    by_day = runs.groupby('date')
    daily = pd.DataFrame({
        'trimp': by_day['trimp'].sum(),
        # tempo dnia ważone czasem biegu (a nie średnia temp)
        'speed': by_day['sd'].sum(min_count=1) / runs.dropna(subset=['sd']).groupby('date')['dur'].sum(),
        'hr': by_day['hr'].mean(), 'pow': by_day['pow'].mean(), 'EF': by_day['EF'].mean(),
        'vo': by_day[vo_col].mean() if vo_col else np.nan,
        'gct': by_day[gct_col].mean() if gct_col else np.nan,
        'ae': by_day[ae_col].sum() if ae_col else np.nan,
        'an': by_day[an_col].sum() if an_col else np.nan,
    })
    daily['pace'] = 1000 / daily['speed'] / 60
    forecast_end = max(today_ts, end_ts) + pd.Timedelta(days=21)
    full = pd.date_range(min(runs['date'].min(), start_ts), forecast_end)
    load = M.training_load(daily['trimp'].reindex(full, fill_value=0.0))
    view = pd.date_range(start_ts, forecast_end)
    load = load.reindex(view)
    daily_v = daily.reindex(view)

    # --- HRV ---
    dr = pd.DataFrame(daily_rows)
    if not dr.empty:
        dr['date'] = pd.to_datetime(dr['date'])
        dr = dr.set_index('date').sort_index()
    hrv_idx = pd.date_range(pd.Timestamp(daily_start), end_ts)
    hrv = pd.DataFrame(index=hrv_idx)
    if not dr.empty and dr.get('hrv') is not None and dr['hrv'].notna().any():
        hrv['hrv'] = pd.to_numeric(dr['hrv'], errors='coerce').reindex(hrv_idx).interpolate(method='time')
        hrv['hrv_7d'] = hrv['hrv'].ewm(span=7, adjust=False).mean()
        avg30 = hrv['hrv'].rolling(30, min_periods=1).mean()
        std30 = hrv['hrv'].rolling(30, min_periods=1).std().fillna(0)
        hrv['hrv_upper'], hrv['hrv_lower'] = avg30 + std30, avg30 - std30
    hrv = hrv.reindex(view)
    for c in ('hrv', 'hrv_7d', 'hrv_upper', 'hrv_lower'):
        if c not in hrv.columns: hrv[c] = np.nan

    # --- heatmap (tylko historia w zakresie) ---
    hist = load[load.index <= min(today_ts, end_ts)]
    if len(hist):
        hm_df = pd.DataFrame({'trimp': hist['trimp'].values}, index=hist.index)
        hm_df['week'] = hm_df.index - pd.to_timedelta(hm_df.index.dayofweek, unit='d')
        hm_df['dow'] = hm_df.index.dayofweek
        weeks = pd.date_range(hm_df['week'].min(), hm_df['week'].max(), freq='7D')
        hm = hm_df.pivot(index='dow', columns='week', values='trimp').reindex(index=range(7), columns=weeks).fillna(0)
        heatmap_z, heatmap_x = hm.values.tolist(), _ds(hm.columns)
    else:
        heatmap_z, heatmap_x = [], []

    trends = {}
    trend_src = daily.reset_index(names='date')
    trend_src = trend_src[trend_src['date'] >= start_ts]
    for metric, c in (('ef', 'EF'), ('pow', 'pow')):
        h_d, h_v, f_d, f_v = get_trend_and_forecast(trend_src, c)
        trends[metric] = {'hist_dates': h_d, 'hist_vals': h_v, 'fut_dates': f_d, 'fut_vals': f_v}

    # --- MMP z 5 biegów o najwyższym TRIMP w zakresie ---
    in_range = runs[runs['date'] >= start_ts]
    top_ids = [int(i) for i in in_range.nlargest(5, 'trimp')['activityId'].dropna()] if 'activityId' in runs else []
    mmp_vals = M.mmp_from_derived(get_derived(top_ids))

    # --- aerobic decoupling z długich biegów (szczegóły pobierane raz, potem z bazy) ---
    long_runs = in_range[in_range['dur'] >= M.DECOUPLING_MIN_S].tail(DECOUPLING_MAX_RUNS) if 'activityId' in runs else in_range.iloc[:0]
    long_ids = [int(i) for i in long_runs['activityId'].dropna()]
    dec = []
    for (_, row), der in zip(long_runs.iterrows(), get_derived(long_ids)):
        pct = der.get("decoupling") if der else None
        if pct is not None:
            dec.append({"date": row['date'].strftime('%Y-%m-%d'), "pct": pct, "min": int(row['dur'] // 60)})
    dec_recent = [d["pct"] for d in dec[-4:]]

    # --- forma: VDOT, VO2max Garmina, tempo przy HR, prognozy ---
    recent_src = runs.loc[runs['date'] > runs['date'].max() - pd.Timedelta(days=42), 'vdot_src']
    vdot_trend = M.rolling_median_by_time(runs['date'], runs['vdot_run'], days=42, min_n=3)
    spd_ref_trend = M.ewm_by_time(runs['date'], runs['spd_ref'], halflife_days=14)
    pace_ref_trend = 1000 / spd_ref_trend / 60
    vo2_garmin = M.ewm_by_time(runs['date'], num(vo2_col), 1) if vo2_col else pd.Series(dtype=float)
    vdot_now = _last(vdot_trend)
    t10_now = M.race_time_from_vdot(vdot_now, M.KM10) if vdot_now else None
    thm_now = M.race_time_from_vdot(vdot_now, M.HM) if vdot_now else None
    own_10k = [M.race_time_from_vdot(v, M.KM10) for v in vdot_trend.values]
    own_hm = [M.race_time_from_vdot(v, M.HM) for v in vdot_trend.values]

    pred = pd.DataFrame(index=pd.DatetimeIndex([]))
    if not dr.empty and 'pred_10k' in dr.columns:
        pred = dr[['pred_10k', 'pred_hm']].apply(pd.to_numeric, errors='coerce').dropna(how='all') / 60
    g10, ghm = (_last(pred['pred_10k']), _last(pred['pred_hm'])) if len(pred) else (None, None)

    r_view = runs[runs['date'] >= start_ts]
    fitness = {
        "run_dates": _ds(r_view['date']),
        "vdot_run": _clean(r_view['vdot_run']),
        "vdot_src": [v if isinstance(v, str) else None for v in r_view['vdot_src']],
        "vo2_garmin_run": _clean(num(vo2_col)[r_view.index]) if vo2_col else [],
        "pace_ref_run": _clean(1000 / r_view['spd_ref'] / 60),
        "max_hr_run": _clean(pd.to_numeric(r_view[max_hr_col], errors='coerce')) if max_hr_col else [],
        "avg_hr_run": _clean(r_view['hr']),
        "trend_dates": _ds(vdot_trend.index[vdot_trend.index >= start_ts]),
        "vdot_trend": _clean(vdot_trend[vdot_trend.index >= start_ts]),
        "own_10k": _clean([v for v, d in zip(own_10k, vdot_trend.index) if d >= start_ts]),
        "own_hm": _clean([v for v, d in zip(own_hm, vdot_trend.index) if d >= start_ts]),
        "ref_dates": _ds(pace_ref_trend.index[pace_ref_trend.index >= start_ts]),
        "pace_ref_trend": _clean(pace_ref_trend[pace_ref_trend.index >= start_ts]),
        "pred_dates": _ds(pred.index[pred.index >= start_ts]) if len(pred) else [],
        "garmin_10k": _clean(pred['pred_10k'][pred.index >= start_ts]) if len(pred) else [],
        "garmin_hm": _clean(pred['pred_hm'][pred.index >= start_ts]) if len(pred) else [],
        "hr_ref": hr_ref,
        "decoupling": dec,
    }

    # --- zdrowie: tętno spoczynkowe, waga, BMI, talia ---
    health, kpi_health = _health(dr, profile, start_ts, end, waist_cm)

    # --- limit tętna ---
    cap = None
    if hr_cap and max_hr_col:
        mx = pd.to_numeric(r_view[max_hr_col], errors='coerce')
        over = r_view[mx > hr_cap]
        cap = {"cap": hr_cap, "runs": int(mx.notna().sum()), "over": int(len(over)),
               "last_over": over['date'].max().strftime('%Y-%m-%d') if len(over) else None,
               "max_seen": int(mx.max()) if mx.notna().any() else None}

    ctl_now = _last(load['CTL'][load.index <= today_ts])
    kpi = {
        "vdot": round(vdot_now, 1) if vdot_now else None,
        "vdot_seg": int((recent_src == 'seg').sum()),
        "vdot_runs": int(recent_src.notna().sum()),
        "vdot_delta_28d": _delta(vdot_trend, 28),
        "vo2_garmin": round(_last(vo2_garmin), 1) if len(vo2_garmin) else profile.get("vo2max_running"),
        "own_10k": M.fmt_time(t10_now),
        "own_hm": M.fmt_time(thm_now),
        "garmin_10k": M.fmt_time(g10), "garmin_hm": M.fmt_time(ghm),
        "pace_10k": M.fmt_pace(t10_now / 10) if t10_now else None,
        "pace_hm": M.fmt_pace(thm_now / 21.0975) if thm_now else None,
        "hr_ref": hr_ref,
        "pace_ref": M.fmt_pace(_last(pace_ref_trend)),
        "pace_ref_delta_s_28d": _pace_delta_s(pace_ref_trend, 28),
        "ctl": round(ctl_now, 1) if ctl_now is not None else None,
        "tsb": _round(_last(load['TSB'][load.index <= today_ts])),
        "ramp": _round(_last(load['ramp'][load.index <= today_ts])),
        "hr_cap": cap,
        "decoupling": round(float(np.median(dec_recent)), 1) if dec_recent else None,
        "decoupling_n": len(dec_recent),
        **kpi_health,
    }

    return {
        "debug": {"hr_rest": hr_rest, "hr_max": hr_max},
        "kpi": kpi,
        "dates": _ds(view),
        "ctl": _clean(load['CTL']), "atl": _clean(load['ATL']), "tsb": _clean(load['TSB']),
        "ef": _clean(daily_v['EF']), "pow": _clean(daily_v['pow']), "hr": _clean(daily_v['hr']),
        "vo": _clean(daily_v['vo']), "gct": _clean(daily_v['gct']),
        "ae": _clean(daily_v['ae']), "an": _clean(daily_v['an']),
        "pace": _clean(daily_v['pace']),
        "pace_str": [M.fmt_pace(p) if p == p else None for p in daily_v['pace']],
        "hrv": _clean(hrv['hrv']), "hrv_7d": _clean(hrv['hrv_7d']),
        "hrv_upper": _clean(hrv['hrv_upper']), "hrv_lower": _clean(hrv['hrv_lower']),
        "trends": trends,
        "hm_z": heatmap_z, "hm_x": heatmap_x, "hm_y": ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Ndz'],
        "mmp_labels": ['10s', '30s', '1 min', '5 min', '20 min', '1 godz.'], "mmp_vals": mmp_vals,
        "fitness": fitness,
        "health": health,
    }


def _round(v, n=1):
    return round(v, n) if v is not None else None


def _delta(s: pd.Series, days: int):
    now, old = _last(s), _value_days_ago(s, days)
    return round(now - old, 1) if now is not None and old is not None else None


def _pace_delta_s(pace: pd.Series, days: int):
    """Zmiana tempa w s/km (ujemna = szybciej)."""
    now, old = _last(pace), _value_days_ago(pace, days)
    return int(round((now - old) * 60)) if now is not None and old is not None else None


def _health(dr: pd.DataFrame, profile: dict, start_ts, end: date, waist_cm):
    health = {"rhr_dates": [], "rhr": [], "rhr_7d": [], "rhr_60d": [],
              "w_dates": [], "weight": [], "weight_trend_dates": [], "weight_trend": [],
              "body_fat": [], "muscle": [], "band_kg": None}
    k = {"rhr_3d": None, "rhr_7d": None, "rhr_60d": None, "rhr_alert": False,
         "weight": None, "weight_rate": None, "bmi": None, "bmi_band": None, "bmi_note": None,
         "target_kg": None, "target_eta_weeks": None, "body_fat": None, "muscle_rate": None,
         "age": None, "hr_max_tanaka": None, "waist": None}

    age = M.age_on(profile.get("birth_date"), end)
    height = profile.get("height_cm")
    if height and height < 3:
        height *= 100
    k["age"] = age
    if age:
        k["hr_max_tanaka"] = int(round(208 - 0.7 * age))
    k["waist"] = M.waist_assessment(waist_cm, height, profile.get("gender"))

    if dr.empty:
        return health, k

    if 'rhr' in dr.columns and dr['rhr'].notna().any():
        rhr = pd.to_numeric(dr['rhr'], errors='coerce')
        rhr = rhr.reindex(pd.date_range(rhr.index.min(), rhr.index.max()))
        r3 = rhr.rolling(3, min_periods=2).mean()
        r7 = rhr.rolling(7, min_periods=3).mean()
        r60 = rhr.rolling(60, min_periods=14).mean()
        sel = rhr.index >= start_ts
        health.update(rhr_dates=_ds(rhr.index[sel]), rhr=_clean(rhr[sel]), rhr_7d=_clean(r7[sel]), rhr_60d=_clean(r60[sel]))
        k["rhr_7d"], k["rhr_60d"] = _round(_last(r7)), _round(_last(r60))
        # alarm: ostatnie 3 dni >= 5 ud./min powyżej normy z 60 dni (choroba, przemęczenie)
        r3_now = _last(r3)
        k["rhr_3d"] = _round(r3_now)
        k["rhr_alert"] = bool(r3_now and k["rhr_60d"] and r3_now - k["rhr_60d"] >= 5)

    if 'weight' in dr.columns and dr['weight'].notna().any():
        w = pd.to_numeric(dr['weight'], errors='coerce').dropna()
        trend = M.ewm_by_time(w.index, w.values, halflife_days=7)
        sel = w.index >= start_ts
        health.update(w_dates=_ds(w.index[sel]), weight=_clean(w[sel]),
                      weight_trend_dates=_ds(trend.index[trend.index >= start_ts]),
                      weight_trend=_clean(trend[trend.index >= start_ts]))
        for src, dst in (('body_fat', 'body_fat'), ('muscle_mass', 'muscle')):
            if src in dr.columns:
                health[dst] = _clean(pd.to_numeric(dr[src], errors='coerce').reindex(w.index)[sel])
        cur = _last(trend)
        rate = M.slope_per_week(trend, 28)
        k["weight"], k["weight_rate"] = _round(cur), _round(rate, 2)
        if 'body_fat' in dr.columns:
            k["body_fat"] = _round(_last(pd.to_numeric(dr['body_fat'], errors='coerce')))
        if 'muscle_mass' in dr.columns and dr['muscle_mass'].notna().sum() >= 3:
            mm = pd.to_numeric(dr['muscle_mass'], errors='coerce').dropna()
            k["muscle_rate"] = _round(M.slope_per_week(M.ewm_by_time(mm.index, mm.values, 7), 28), 2)
        if height and cur:
            h2 = (height / 100) ** 2
            lo, hi, note = M.bmi_band(age)
            k["bmi"], k["bmi_band"], k["bmi_note"] = round(cur / h2, 1), [lo, hi], note
            health["band_kg"] = [round(lo * h2, 1), round(hi * h2, 1)]
            target = hi * h2 if cur > hi * h2 else (lo * h2 if cur < lo * h2 else None)
            if target:
                k["target_kg"] = round(target, 1)
                if rate and (target - cur) * rate > 0:
                    k["target_eta_weeks"] = int(math.ceil(abs(target - cur) / abs(rate)))
    return health, k
