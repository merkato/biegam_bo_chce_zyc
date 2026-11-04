import hashlib
import hmac
import logging
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

log = logging.getLogger("biegam")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Sesje Garmin -----------------------------------------------------------
# Klucz: e-mail. Sesja jest użyta ponownie TYLKO gdy podane hasło zgadza się
# z tym, którym ją utworzono (porównujemy skrót PBKDF2 z losową solą procesu).

SESSION_TTL_S = 6 * 3600
_SALT = secrets.token_bytes(16)


@dataclass
class _Session:
    pw_digest: bytes
    created: float
    client: Garmin


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()

# --- Ochrona przed banem ----------------------------------------------------
# 1) najwyżej jedna analiza naraz (równoległe zapytania wyglądają jak scraper),
# 2) limit prawdziwych logowań SSO (Garmin najostrzej limituje właśnie login),
# 3) po 429 od Garmina: przerwa dla całej aplikacji.

_analysis_lock = threading.Lock()
LOGIN_LIMIT, LOGIN_WINDOW_S = 5, 15 * 60
_login_times: deque[float] = deque()
COOLDOWN_S = 30 * 60
_cooldown_until = 0.0

_GARMIN_FATAL = (GarminConnectTooManyRequestsError, GarminConnectAuthenticationError)


def _pw_digest(email: str, password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), _SALT + email.encode(), 100_000)


def _start_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = time.time() + COOLDOWN_S
    log.warning("Garmin zwrócił 429 – przerwa do %s", time.ctime(_cooldown_until))


def _get_client(email: str, password: str) -> Garmin:
    key = email.strip().lower()
    digest = _pw_digest(key, password)
    now = time.time()
    with _sessions_lock:
        s = _sessions.get(key)
        if s and now - s.created < SESSION_TTL_S and hmac.compare_digest(s.pw_digest, digest):
            return s.client
        while _login_times and now - _login_times[0] > LOGIN_WINDOW_S:
            _login_times.popleft()
        if len(_login_times) >= LOGIN_LIMIT:
            raise HTTPException(429, "Zbyt wiele prób logowania. Spróbuj za kilkanaście minut.")
        _login_times.append(now)
    client = Garmin(email, password)
    client.login()  # może rzucić błąd uwierzytelnienia lub 429 – obsługa w endpoincie
    with _sessions_lock:
        _sessions[key] = _Session(digest, now, client)
    return client


def _drop_session(email: str) -> None:
    with _sessions_lock:
        _sessions.pop(email.strip().lower(), None)

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

# Zwykłe `def` (nie `async`): FastAPI uruchomi je w puli wątków, więc blokujące
# zapytania do Garmina nie zatrzymują pętli zdarzeń ani heartbeatu workera.
@app.post("/api/analyze")
def analyze_garmin(
    email: str = Form(...),
    password: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    user_hr_rest: int = Form(None),
    user_hr_max: int = Form(None)
):
    wait = int(_cooldown_until - time.time())
    if wait > 0:
        raise HTTPException(429, f"Garmin ograniczył zapytania. Spróbuj ponownie za ok. {wait // 60 + 1} min.")
    try:
        start_d, end_d = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(422, "Nieprawidłowy format daty.")
    if start_d > end_d:
        raise HTTPException(422, "Data początkowa jest późniejsza niż końcowa.")
    if not _analysis_lock.acquire(blocking=False):
        raise HTTPException(429, "Trwa inna analiza. Poczekaj chwilę.")
    try:
        return _analyze(email, password, start_date, end_date, user_hr_rest, user_hr_max)
    except HTTPException:
        raise
    except GarminConnectTooManyRequestsError:
        _start_cooldown()
        raise HTTPException(429, "Garmin ograniczył zapytania (429). Odczekaj ok. 30 min.")
    except GarminConnectAuthenticationError:
        _drop_session(email)
        raise HTTPException(401, "Logowanie do Garmin Connect nie powiodło się (błędne dane lub wygasła sesja).")
    except GarminConnectConnectionError:
        log.exception("Błąd połączenia z Garmin Connect")
        raise HTTPException(502, "Garmin Connect nie odpowiada poprawnie. Spróbuj później.")
    except Exception:
        log.exception("Nieoczekiwany błąd analizy")
        raise HTTPException(500, "Wewnętrzny błąd analizy. Szczegóły w logach serwera.")
    finally:
        _analysis_lock.release()


def _analyze(email, password, start_date, end_date, user_hr_rest, user_hr_max):
    client = _get_client(email, password)

    activities = client.get_activities_by_date(start_date, end_date)
    if not activities:
        raise HTTPException(status_code=404, detail="Brak aktywności we wskazanym zakresie.")

    df = pd.DataFrame(activities)
    df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
    # running, trail_running, treadmill_running, track_running, indoor_running...
    run_df = df[df['type_key'].str.contains('running', na=False)].copy()

    if run_df.empty:
        raise HTTPException(status_code=404, detail="Brak biegów w wybranym zakresie dat.")

    if user_hr_rest is None:
        try:
            hr_rest = (client.get_stats(date.today().isoformat()) or {}).get('restingHeartRate') or 50
        except _GARMIN_FATAL:
            raise
        except Exception:
            log.warning("Nie udało się pobrać tętna spoczynkowego", exc_info=True)
            hr_rest = 50
    else: hr_rest = user_hr_rest

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
        
    # Ekstrakcja Training Effect (TE)
    ae_col = next((c for c in ['aerobicTrainingEffect'] if c in run_df.columns), None)
    an_col = next((c for c in ['anaerobicTrainingEffect'] if c in run_df.columns), None)

    if gap_col:
        run_df[gap_col] = pd.to_numeric(run_df[gap_col], errors='coerce')
        valid_speed = run_df[gap_col] > 0
        run_df.loc[valid_speed, 'pace_decimal'] = (1000 / run_df.loc[valid_speed, gap_col]) / 60
    else: run_df['pace_decimal'] = np.nan

    run_df['EF'] = run_df[pow_col] / run_df[hr_col] if pow_col and hr_col else np.nan
    run_df['trimp'] = run_df.apply(lambda row: calculate_trimp(row.get(hr_col), row.get(dur_col), hr_rest, hr_max), axis=1)

    mmp_windows = [10, 30, 60, 300, 1200, 3600]
    mmp_labels = ['10s', '30s', '1 min', '5 min', '20 min', '1 godz.']
    max_powers = {w: 0 for w in mmp_windows}
        
    if not run_df.empty and 'trimp' in run_df.columns:
        top_runs = run_df.nlargest(5, 'trimp')
        for _, row in top_runs.iterrows():
            try:
                act_id = row.get('activityId')
                if act_id:
                    details = client.get_activity_details(act_id)
                    p_idx = next((i.get('metricsIndex') for i in details.get('metricDescriptors', []) if i.get('key') in ['directPower', 'power']), None)
                    if p_idx is not None:
                        power_data = [vals[p_idx] if len(vals) > p_idx and vals[p_idx] is not None else 0 for m in details.get('activityDetailMetrics', []) for vals in [m.get('metrics', [])]]
                        if power_data:
                            s = pd.Series(power_data)
                            for w in mmp_windows:
                                if len(s) >= w:
                                    w_max = s.rolling(w).mean().max()
                                    if w_max > max_powers[w]: max_powers[w] = w_max
            except _GARMIN_FATAL:
                raise
            except Exception:
                log.warning("Pominięto szczegóły aktywności %s", row.get('activityId'), exc_info=True)
            time.sleep(0.5)
                
    mmp_vals = [round(max_powers[w], 1) if not pd.isna(max_powers[w]) else 0 for w in mmp_windows]
        
    agg_dict = {'pace_decimal': 'mean', 'trimp': 'sum'}
    if pow_col: agg_dict[pow_col] = 'mean'
    if hr_col: agg_dict[hr_col] = 'mean'
    if vo_col: agg_dict[vo_col] = 'mean'
    if gct_col: agg_dict[gct_col] = 'mean'
    if ae_col: agg_dict[ae_col] = 'sum'
    if an_col: agg_dict[an_col] = 'sum'
    if 'EF' in run_df.columns: agg_dict['EF'] = 'mean'
            
    # Grupowanie po DNIU (nie po dokładnym czasie startu): dwa biegi jednego dnia
    # dawały zduplikowane daty i wyjątek przy .map() poniżej.
    run_df['date'] = pd.to_datetime(run_df['startTimeLocal']).dt.normalize()
    daily = run_df.groupby('date').agg(agg_dict).reset_index()
    daily['pace_str'] = daily['pace_decimal'].apply(format_pace)

    hrv_records = []
    hrv_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=45)
    hrv_dates = pd.date_range(start=hrv_start_date, end=end_date)
    if len(hrv_dates) > 120:
        hrv_dates = pd.date_range(start=pd.to_datetime(end_date)-pd.Timedelta(days=120), end=end_date)
            
    for d in hrv_dates:
        try:
            hrv_res = client.get_hrv_data(d.strftime('%Y-%m-%d'))
            if isinstance(hrv_res, dict):
                avg = hrv_res['hrvSummary'].get('lastNightAvg') if 'hrvSummary' in hrv_res else hrv_res.get('lastNightAvg')
                if avg: hrv_records.append({'date': d, 'hrv': avg})
        except _GARMIN_FATAL:
            raise
        except Exception:
            log.debug("Brak HRV dla %s", d.date(), exc_info=True)
        time.sleep(0.3)
            
    if hrv_records:
        hrv_df = pd.DataFrame(hrv_records)
        hrv_df['date'] = pd.to_datetime(hrv_df['date'])
        hrv_df = hrv_df.set_index('date').reindex(hrv_dates)
        hrv_df['hrv'] = pd.to_numeric(hrv_df['hrv'], errors='coerce').interpolate(method='time')
        hrv_df['hrv_7d'] = hrv_df['hrv'].ewm(span=7, adjust=False).mean()
        hrv_df['hrv_30d_avg'] = hrv_df['hrv'].rolling(window=30, min_periods=1).mean()
        hrv_df['hrv_30d_std'] = hrv_df['hrv'].rolling(window=30, min_periods=1).std().fillna(0)
        hrv_df['hrv_upper'] = hrv_df['hrv_30d_avg'] + hrv_df['hrv_30d_std']
        hrv_df['hrv_lower'] = hrv_df['hrv_30d_avg'] - hrv_df['hrv_30d_std']
        hrv_df = hrv_df.reset_index().rename(columns={'index': 'date'})
    else:
        hrv_df = pd.DataFrame(columns=['date', 'hrv', 'hrv_7d', 'hrv_upper', 'hrv_lower'])

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

    return {
        "debug": {"hr_rest": hr_rest, "hr_max": hr_max},
        "dates": final_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        "ctl": final_df['CTL'].tolist(),
        "atl": final_df['ATL'].tolist(),
        "ef": final_df.get('EF', pd.Series([])).tolist(),
        "pow": final_df.get(pow_col, pd.Series([])).tolist() if pow_col else [],
        "hr": final_df.get(hr_col, pd.Series([])).tolist() if hr_col else [],
        "vo": final_df.get(vo_col, pd.Series([])).tolist() if vo_col else [],
        "gct": final_df.get(gct_col, pd.Series([])).tolist() if gct_col else [],
        "ae": final_df.get(ae_col, pd.Series([])).tolist() if ae_col else [],
        "an": final_df.get(an_col, pd.Series([])).tolist() if an_col else [],
        "pace": final_df['pace_decimal'].tolist(),
        "pace_str": final_df['pace_str'].tolist(),
        "hrv": final_df['hrv'].tolist(),
        "hrv_7d": final_df['hrv_7d'].tolist(),
        "hrv_upper": final_df['hrv_upper'].tolist(),
        "hrv_lower": final_df['hrv_lower'].tolist(),
        "trends": trends,
        "hm_z": heatmap_z, "hm_x": heatmap_x, "hm_y": ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Ndz'],
        "mmp_labels": mmp_labels, "mmp_vals": mmp_vals
    }
