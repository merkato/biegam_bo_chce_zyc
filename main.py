from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from garminconnect import Garmin
from datetime import date
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze")
async def analyze_garmin(
    email: str = Form(...), 
    password: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    user_hr_rest: int = Form(None),  # Opcjonalne
    user_hr_max: int = Form(None)    # Opcjonalne
):
    try:
        client = Garmin(email, password)
        client.login()
        
        # 1. Pobieranie aktywności ze wskazanego zakresu dat
        activities = client.get_activities_by_date(start_date, end_date)
        if not activities:
            raise HTTPException(status_code=404, detail="Brak aktywności w tym zakresie dat.")
            
        df = pd.DataFrame(activities)
        
        # Filtrujemy tylko biegi
        df['type_key'] = df['activityType'].apply(lambda x: x.get('typeKey', '') if isinstance(x, dict) else '')
        run_df = df[df['type_key'] == 'running'].copy()
        
        if run_df.empty:
            raise HTTPException(status_code=404, detail="Brak biegów w wybranych datach.")

        # 2. Logika dynamicznego tętna (Fallback na auto-detekcję)
        today_iso = date.today().isoformat()
        
        # HR REST: Pobieramy dzisiejsze statystyki zdrowotne z Garmina
        if user_hr_rest is None:
            try:
                stats = client.get_stats(today_iso)
                hr_rest = stats.get('restingHeartRate', 50) # domyślnie 50, jeśli API zawiedzie
            except:
                hr_rest = 50
        else:
            hr_rest = user_hr_rest

        # HR MAX: Szukamy najwyższego zarejestrowanego tętna w pobranych aktywnościach
        if user_hr_max is None:
            max_hr_col = next((c for c in ['maxHeartRateInBeatsPerMinute', 'maxHR'] if c in run_df.columns), None)
            if max_hr_col and not run_df[max_hr_col].isna().all():
                hr_max = int(run_df[max_hr_col].max())
            else:
                hr_max = 185 # Bezpieczny fallback
        else:
            hr_max = user_hr_max

        # Tutaj wykonujesz swoje funkcje (calculate_trimp, ewma itp.)
        # używając dynamicznie przypisanych zmiennych 'hr_rest' oraz 'hr_max'
        
        # Zwracanie JSON z danymi dla Plotly.js + info o wykorzystanym tętnie do celów weryfikacji
        return {
            "debug": {"hr_rest": hr_rest, "hr_max": hr_max},
            "dates": run_df['date'].tolist(), # Twoja oś X
            "ctl": run_df['CTL'].tolist(),
            "atl": run_df['ATL'].tolist(),
            "ef": run_df['EF'].tolist(),
            "pow": run_df['averagePower'].tolist(),
            "hr": run_df['averageHR'].tolist(),
            "pace": run_df['pace_decimal'].tolist(), # Oś Y dla tempa
            "pace_str": run_df['pace_str'].tolist()  # Dymki MM:SS po najechaniu
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))