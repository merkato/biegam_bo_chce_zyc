import math
import time
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from garminconnect import Garmin
from datetime import date
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SESSION_CACHE: Dict[str, Garmin] = {}

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

@app.post("/api/analyze")
async def analyze_garmin(
    email: str = Form(...), 
    password: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    user_hr_rest: int = Form(None),
    user_hr_max: int = Form(None)
):
    try:
        if email in SESSION_CACHE:
            client = SESSION_CACHE[email]
        else:
            client = Garmin(email, password)
            client.login()
            SESSION_CACHE[email] = client
        
        activities = client.get_activities_by_date(start_date, end_date)
        if not activities:
            raise HTTPException(status_code=404, detail="Brak aktywności we wskazanym zakresie.")
            
        df = pd.DataFrame(activities)
        df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
        run_df = df[df['type_key'] == 'running'].copy()
        
        if run_df.empty:
            raise HTTPException(status_code=404, detail="Brak biegów w wybranym zakresie dat.")

        if user_hr_rest is None:
            try: hr_rest = client.get_stats(date.today().isoformat()).get('restingHeartRate', 50)
            except: hr_rest = 50
        else: hr_rest = user_hr_rest

        max_hr_col = next((c for c in ['maxHeartRateInBeatsPerMinute', 'maxHR'] if c in run_df.columns), None)
        if user_hr_max is None:
            hr_max = int(run_df[max_hr_col].max()) if max_hr_col and not run_df[max_hr_col].isna().all() else 185
        else: hr_max = user_hr_max

        pow_col = next((c for c in ['averagePower', 'avgPower'] if c in run_df.columns), None)
        hr_col = next((c for c in ['averageHR', 'averageHeartRateInBeatsPerMinute', 'averageBpm'] if c in run_df.columns), None)
        gap_col = next((c for c in ['averageGradeAdjustedSpeed', 'avgGradeAdjustedSpeed', 'averageSpeed'] if c in run_df.columns), None)
        dur_col = next((c for c in ['duration', 'movingDuration'] if c in run_df.columns), None)

        if gap_col:
            run_df[gap_col] = pd.to_numeric(run_df[gap_col], errors='coerce')
            valid_speed = run_df[gap_col] > 0
            run_df.loc[valid_speed, 'pace_decimal'] = (1000 / run_df.loc[valid_speed, gap_col]) / 60
        else: run_df['pace_decimal'] = np.nan

        run_df['EF'] = run_df[pow_col] / run_df[hr_col] if pow_col and hr_col else np.nan
        run_df['trimp'] = run_df.apply(lambda row: calculate_trimp(row.get(hr_col), row.get(dur_col), hr_rest, hr_max), axis=1)
        
        agg_dict = {'pace_decimal': 'mean', 'trimp': 'sum'}
        if pow_col: agg_dict[pow_col] = 'mean'
        if hr_col: agg_dict[hr_col] = 'mean'
        if 'EF' in run_df.columns: agg_dict['EF'] = 'mean'
            
        daily = run_df.groupby('startTimeLocal').agg(agg_dict).reset_index()
        daily['date'] = pd.to_datetime(daily['startTimeLocal']).dt.normalize()
        daily['pace_str'] = daily['pace_decimal'].apply(format_pace)

        # --- MODUŁ HRV ---
        hrv_records = []
        hrv_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=30)
        hrv_dates = pd.date_range(start=hrv_start_date, end=end_date)
        
        # Ochrona przed Timeoutem - blokada na max 60 dni dla nocnego tętna (30 dni bufora + 30 wyliczeń)
        if len(hrv_dates) > 60:
            hrv_dates = pd.date_range(start=pd.to_datetime(end_date)-pd.Timedelta(days=60), end=end_date)
            
        for d in hrv_dates:
            try:
                hrv_res = client.get_hrv_data(d.strftime('%Y-%m-%d'))
                if isinstance(hrv_res, dict):
                    avg = hrv_res['hrvSummary'].get('lastNightAvg') if 'hrvSummary' in hrv_res else hrv_res.get('lastNightAvg')
                    if avg: hrv_records.append({'date': d, 'hrv': avg})
            except: pass
            time.sleep(0.02) 
            
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

        # --- MODELE I ZŁĄCZENIA ---
        min_d, max_d = daily['date'].min(), daily['date'].max()
        forecast_end = max(pd.Timestamp.today().normalize(), max_d) + pd.Timedelta(days=21)
        full_dates = pd.date_range(start=min_d, end=forecast_end)
        
        daily_trimp = pd.DataFrame({'date': full_dates})
        temp_trimp = daily.set_index('date')['trimp']
        daily_trimp['trimp'] = daily_trimp['date'].map(temp_trimp).fillna(0)
        daily_trimp['CTL'] = daily_trimp['trimp'].ewm(span=42, adjust=False).mean()
        daily_trimp['ATL'] = daily_trimp['trimp'].ewm(span=7, adjust=False).mean()

        # FIX: Usunięcie 'trimp' z 'daily' by uniknąć duplikatów trimp_x i trimp_y!
        daily_for_merge = daily.drop(columns=['trimp'])
        final_df = pd.merge(daily_trimp, daily_for_merge, on='date', how='left')
        final_df = pd.merge(final_df, hrv_df, on='date', how='left')
        
        for col in ['hrv', 'hrv_7d', 'hrv_upper', 'hrv_lower']:
            if col not in final_df.columns: final_df[col] = None
        final_df = final_df.replace({np.nan: None})

        # --- MODUŁ MAPY CIEPLNEJ (TRIMP) ---
        today_ts = pd.Timestamp.today().normalize()
        hist_df = final_df[final_df['date'] <= today_ts].copy()
        
        hist_df['week_start'] = hist_df['date'] - pd.to_timedelta(hist_df['date'].dt.dayofweek, unit='d')
        hist_df['dow'] = hist_df['date'].dt.dayofweek
        
        if not hist_df.empty:
            min_week = hist_df['week_start'].min()
            max_week = hist_df['week_start'].max()
            weeks = pd.date_range(start=min_week, end=max_week, freq='7D')
            
            hm = hist_df.pivot(index='dow', columns='week_start', values='trimp').reindex(index=range(7), columns=weeks).fillna(0)
            heatmap_z = hm.values.tolist()
            heatmap_x = hm.columns.strftime('%Y-%m-%d').tolist()
        else:
            heatmap_z, heatmap_x = [], []
            
        heatmap_y = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Ndz']

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
            "pace": final_df['pace_decimal'].tolist(),
            "pace_str": final_df['pace_str'].tolist(),
            "hrv": final_df['hrv'].tolist(),
            "hrv_7d": final_df['hrv_7d'].tolist(),
            "hrv_upper": final_df['hrv_upper'].tolist(),
            "hrv_lower": final_df['hrv_lower'].tolist(),
            "trends": trends,
            "hm_z": heatmap_z,
            "hm_x": heatmap_x,
            "hm_y": heatmap_y
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
