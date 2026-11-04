"""Obliczenia dashboardu – czyste funkcje na danych z lokalnej bazy (bez sieci).

Logika metryk jest przeniesiona 1:1 z dawnego main.py; poprawki obliczeń
(CTL/ATL, MMP, ważenie tempa) to osobny krok.
"""
import math
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd


class NoData(Exception):
    pass


def format_pace(decimal_pace):
    if pd.isna(decimal_pace) or decimal_pace <= 0: return "0:00"
    mins = int(decimal_pace)
    secs = int((decimal_pace - mins) * 60)
    return f"{mins}:{secs:02d}"


def calculate_trimp(hr_avg, duration_sec, hr_rest, hr_max):
    if pd.isna(hr_avg) or pd.isna(duration_sec) or duration_sec <= 0: return 0
    duration_min = duration_sec / 60.0
    hr_ratio = max(0, min((hr_avg - hr_rest) / (hr_max - hr_rest), 1))
    return duration_min * hr_ratio * 0.64 * math.exp(1.92 * hr_ratio)


def get_trend_and_forecast(df, col_name, span=21, forecast_days=21):
    if col_name not in df.columns: return [], [], [], []
    df_clean = df[['date', col_name]].dropna().sort_values('date')
    if df_clean.empty: return [], [], [], []

    min_d, max_d = df_clean['date'].min(), df_clean['date'].max()
    full_dates = pd.date_range(start=min_d, end=max_d)

    df_cont = df_clean.set_index('date').reindex(full_dates)
    df_cont['smoothed'] = df_cont[col_name].interpolate(method='time').ewm(span=span, adjust=False).mean()

    recent = df_cont['smoothed'].dropna().tail(span)
    if len(recent) > 1:
        slope = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        f_dates = pd.date_range(start=max_d + pd.Timedelta(days=1), periods=forecast_days)
        f_vals = [recent.iloc[-1] + slope * i for i in range(1, forecast_days + 1)]
    else:
        f_dates, f_vals = pd.DatetimeIndex([]), []

    h_dates = df_cont.index.strftime('%Y-%m-%d').tolist()
    h_vals = df_cont['smoothed'].replace({np.nan: None}).tolist()
    f_dates_str = f_dates.strftime('%Y-%m-%d').tolist() if len(f_dates) > 0 else []

    return h_dates, h_vals, f_dates_str, f_vals


def resting_hr_from_daily(daily_rows: list[dict], default: int = 50) -> int:
    """Mediana tętna spoczynkowego z ostatnich 7 dni z danymi."""
    vals = [r["rhr"] for r in daily_rows if r.get("rhr")][-7:]
    return int(round(float(np.median(vals)))) if vals else default


def _mmp(details_list: list[dict]):
    mmp_windows = [10, 30, 60, 300, 1200, 3600]
    mmp_labels = ['10s', '30s', '1 min', '5 min', '20 min', '1 godz.']
    max_powers = {w: 0 for w in mmp_windows}
    for details in details_list:
        if not details:
            continue
        p_idx = next((i.get('metricsIndex') for i in details.get('metricDescriptors', []) if i.get('key') in ['directPower', 'power']), None)
        if p_idx is None:
            continue
        power_data = [vals[p_idx] if len(vals) > p_idx and vals[p_idx] is not None else 0
                      for m in details.get('activityDetailMetrics', []) for vals in [m.get('metrics', [])]]
        if power_data:
            s = pd.Series(power_data)
            for w in mmp_windows:
                if len(s) >= w:
                    w_max = s.rolling(w).mean().max()
                    if w_max > max_powers[w]: max_powers[w] = w_max
    mmp_vals = [round(max_powers[w], 1) if not pd.isna(max_powers[w]) else 0 for w in mmp_windows]
    return mmp_labels, mmp_vals


def analyze(
    activities: list[dict],
    daily_rows: list[dict],
    daily_start: date,
    end: date,
    user_hr_rest: int | None,
    user_hr_max: int | None,
    get_details: Callable[[list[int]], list[dict]],
) -> dict:
    if not activities:
        raise NoData("Brak aktywności we wskazanym zakresie.")

    df = pd.DataFrame(activities)
    df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
    # running, trail_running, treadmill_running, track_running, indoor_running...
    run_df = df[df['type_key'].str.contains('running', na=False)].copy()
    if run_df.empty:
        raise NoData("Brak biegów w wybranym zakresie dat.")

    hr_rest = user_hr_rest if user_hr_rest is not None else resting_hr_from_daily(daily_rows)

    max_hr_col = next((c for c in ['maxHeartRateInBeatsPerMinute', 'maxHR'] if c in run_df.columns), None)
    if user_hr_max is None:
        hr_max = int(run_df[max_hr_col].max()) if max_hr_col and not run_df[max_hr_col].isna().all() else 185
    else: hr_max = user_hr_max

    pow_col = next((c for c in ['averagePower', 'avgPower'] if c in run_df.columns), None)
    hr_col = next((c for c in ['averageHR', 'averageHeartRateInBeatsPerMinute', 'averageBpm'] if c in run_df.columns), None)
    gap_col = next((c for c in ['averageGradeAdjustedSpeed', 'avgGradeAdjustedSpeed', 'averageSpeed'] if c in run_df.columns), None)
    dur_col = next((c for c in ['duration', 'movingDuration'] if c in run_df.columns), None)
    vo_col = next((c for c in ['averageVerticalOscillation', 'avgVerticalOscillation'] if c in run_df.columns), None)
    gct_col = next((c for c in ['averageGroundContactTime', 'avgGroundContactTime'] if c in run_df.columns), None)
    ae_col = next((c for c in ['aerobicTrainingEffect'] if c in run_df.columns), None)
    an_col = next((c for c in ['anaerobicTrainingEffect'] if c in run_df.columns), None)

    if gap_col:
        run_df[gap_col] = pd.to_numeric(run_df[gap_col], errors='coerce')
        valid_speed = run_df[gap_col] > 0
        run_df.loc[valid_speed, 'pace_decimal'] = (1000 / run_df.loc[valid_speed, gap_col]) / 60
    else: run_df['pace_decimal'] = np.nan
    if 'pace_decimal' not in run_df.columns: run_df['pace_decimal'] = np.nan

    run_df['EF'] = run_df[pow_col] / run_df[hr_col] if pow_col and hr_col else np.nan
    run_df['trimp'] = run_df.apply(lambda row: calculate_trimp(row.get(hr_col), row.get(dur_col), hr_rest, hr_max), axis=1)

    # MMP z 5 biegów o najwyższym TRIMP (szczegóły z bazy; brakujące dociągnie get_details)
    top_ids = [int(i) for i in run_df.nlargest(5, 'trimp')['activityId'].dropna().tolist()] if 'activityId' in run_df.columns else []
    mmp_labels, mmp_vals = _mmp(get_details(top_ids))

    agg_dict = {'pace_decimal': 'mean', 'trimp': 'sum'}
    if pow_col: agg_dict[pow_col] = 'mean'
    if hr_col: agg_dict[hr_col] = 'mean'
    if vo_col: agg_dict[vo_col] = 'mean'
    if gct_col: agg_dict[gct_col] = 'mean'
    if ae_col: agg_dict[ae_col] = 'sum'
    if an_col: agg_dict[an_col] = 'sum'
    if 'EF' in run_df.columns: agg_dict['EF'] = 'mean'

    # Grupowanie po DNIU: dwa biegi jednego dnia = jeden wiersz.
    run_df['date'] = pd.to_datetime(run_df['startTimeLocal']).dt.normalize()
    daily = run_df.groupby('date').agg(agg_dict).reset_index()
    daily['pace_str'] = daily['pace_decimal'].apply(format_pace)

    # HRV z bazy
    hrv_dates = pd.date_range(start=pd.Timestamp(daily_start), end=pd.Timestamp(end))
    hrv_records = [{'date': pd.Timestamp(r['date']), 'hrv': r['hrv']} for r in daily_rows if r.get('hrv')]
    if hrv_records:
        hrv_df = pd.DataFrame(hrv_records).set_index('date').reindex(hrv_dates)
        hrv_df['hrv'] = pd.to_numeric(hrv_df['hrv'], errors='coerce').interpolate(method='time')
        hrv_df['hrv_7d'] = hrv_df['hrv'].ewm(span=7, adjust=False).mean()
        hrv_df['hrv_30d_avg'] = hrv_df['hrv'].rolling(window=30, min_periods=1).mean()
        hrv_df['hrv_30d_std'] = hrv_df['hrv'].rolling(window=30, min_periods=1).std().fillna(0)
        hrv_df['hrv_upper'] = hrv_df['hrv_30d_avg'] + hrv_df['hrv_30d_std']
        hrv_df['hrv_lower'] = hrv_df['hrv_30d_avg'] - hrv_df['hrv_30d_std']
        hrv_df = hrv_df.rename_axis('date').reset_index()
    else:
        hrv_df = pd.DataFrame(columns=['date', 'hrv', 'hrv_7d', 'hrv_upper', 'hrv_lower'])
        hrv_df['date'] = pd.to_datetime(hrv_df['date'])

    min_d, max_d = daily['date'].min(), daily['date'].max()
    forecast_end = max(pd.Timestamp.today().normalize(), max_d) + pd.Timedelta(days=21)
    full_dates = pd.date_range(start=min_d, end=forecast_end)

    daily_trimp = pd.DataFrame({'date': full_dates})
    temp_trimp = daily.set_index('date')['trimp']
    daily_trimp['trimp'] = daily_trimp['date'].map(temp_trimp).fillna(0)
    daily_trimp['CTL'] = daily_trimp['trimp'].ewm(span=42, adjust=False).mean()
    daily_trimp['ATL'] = daily_trimp['trimp'].ewm(span=7, adjust=False).mean()

    daily_for_merge = daily.drop(columns=['trimp'])
    final_df = pd.merge(daily_trimp, daily_for_merge, on='date', how='left')
    final_df = pd.merge(final_df, hrv_df, on='date', how='left')

    for col in ['hrv', 'hrv_7d', 'hrv_upper', 'hrv_lower']:
        if col not in final_df.columns: final_df[col] = None
    final_df = final_df.replace({np.nan: None})

    today_ts = pd.Timestamp.today().normalize()
    hist_df = final_df[final_df['date'] <= today_ts].copy()
    hist_df['week_start'] = hist_df['date'] - pd.to_timedelta(hist_df['date'].dt.dayofweek, unit='d')
    hist_df['dow'] = hist_df['date'].dt.dayofweek
    if not hist_df.empty:
        weeks = pd.date_range(start=hist_df['week_start'].min(), end=hist_df['week_start'].max(), freq='7D')
        hm = hist_df.pivot(index='dow', columns='week_start', values='trimp').reindex(index=range(7), columns=weeks).fillna(0)
        heatmap_z, heatmap_x = hm.values.tolist(), hm.columns.strftime('%Y-%m-%d').tolist()
    else: heatmap_z, heatmap_x = [], []

    trends = {}
    for metric, col in [('ef', 'EF'), ('pow', pow_col), ('hr', hr_col), ('pace', 'pace_decimal')]:
        h_d, h_v, f_d, f_v = get_trend_and_forecast(daily, col) if col else ([], [], [], [])
        trends[metric] = {'hist_dates': h_d, 'hist_vals': h_v, 'fut_dates': f_d, 'fut_vals': f_v}

    col = lambda c: final_df[c].tolist() if c and c in final_df.columns else []
    return {
        "debug": {"hr_rest": hr_rest, "hr_max": hr_max},
        "dates": final_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        "ctl": col('CTL'), "atl": col('ATL'), "ef": col('EF'),
        "pow": col(pow_col), "hr": col(hr_col), "vo": col(vo_col), "gct": col(gct_col),
        "ae": col(ae_col), "an": col(an_col),
        "pace": col('pace_decimal'), "pace_str": col('pace_str'),
        "hrv": col('hrv'), "hrv_7d": col('hrv_7d'), "hrv_upper": col('hrv_upper'), "hrv_lower": col('hrv_lower'),
        "trends": trends,
        "hm_z": heatmap_z, "hm_x": heatmap_x, "hm_y": ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Ndz'],
        "mmp_labels": mmp_labels, "mmp_vals": mmp_vals,
    }
