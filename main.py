import math
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

# Pamięć podręczna RAM
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
            raise HTTPException(status_code=404, detail="Brak aktywności.")
            
        df = pd.DataFrame(activities)
        df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
        run_df = df[df['type_key'] == 'running'].copy()
        
        if run_df.empty:
            raise HTTPException(status_code=404, detail="Brak biegów.")

        if user_hr_rest is None:
            try:
                hr_rest = client.get_stats(date.today().isoformat()).get('restingHeartRate', 50)
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

        # Oś czasu rozciągnięta o 21 dni dla CTL/ATL
        min_d, max_d = daily['date'].min(), daily['date'].max()
        forecast_end = max(pd.Timestamp.today().normalize(), max_d) + pd.Timedelta(days=21)
        full_dates = pd.date_range(start=min_d, end=forecast_end)
        
        daily_trimp = pd.DataFrame({'date': full_dates})
        temp_trimp = daily.set_index('date')['trimp']
        daily_trimp['trimp'] = daily_trimp['date'].map(temp_trimp).fillna(0)
        daily_trimp['CTL'] = daily_trimp['trimp'].ewm(span=42, adjust=False).mean()
        daily_trimp['ATL'] = daily_trimp['trimp'].ewm(span=7, adjust=False).mean()

        final_df = pd.merge(daily_trimp, daily, on='date', how='left').replace({np.nan: None})

        # Wyliczanie trendów i predykcji
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
            "trends": trends
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))