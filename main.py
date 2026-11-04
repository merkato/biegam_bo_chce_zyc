import logging
import threading
from datetime import date, timedelta

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

import analysis
import garmin_session as gs
import store
import sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("biegam")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
store.init()

HRV_LEAD_DAYS = 45      # HRV/RHR sprzed zakresu – na rozbieg średnich kroczących
DAILY_MAX_DAYS = 365    # dłużej wstecz nie pobieramy danych dziennych

# Najwyżej jedna analiza naraz – równoległe zapytania wyglądają jak scraper.
_analysis_lock = threading.Lock()


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
    wait = gs.cooldown_remaining()
    if wait > 0:
        raise HTTPException(429, f"Garmin ograniczył zapytania. Spróbuj ponownie za ok. {wait // 60 + 1} min.")
    try:
        start_d, end_d = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(422, "Nieprawidłowy format daty.")
    if start_d > end_d:
        raise HTTPException(422, "Data początkowa jest późniejsza niż końcowa.")
    email = email.strip().lower()
    if not _analysis_lock.acquire(blocking=False):
        raise HTTPException(429, "Trwa inna analiza. Poczekaj chwilę.")
    try:
        return _analyze(email, password, start_d, end_d, user_hr_rest, user_hr_max)
    except HTTPException:
        raise
    except analysis.NoData as e:
        raise HTTPException(404, str(e))
    except GarminConnectTooManyRequestsError:
        gs.start_cooldown()
        raise HTTPException(429, "Garmin ograniczył zapytania (429). Odczekaj ok. 30 min.")
    except GarminConnectAuthenticationError:
        gs.drop_session(email)
        raise HTTPException(401, "Logowanie do Garmin Connect nie powiodło się (błędne dane lub wygasła sesja).")
    except GarminConnectConnectionError:
        log.exception("Błąd połączenia z Garmin Connect")
        raise HTTPException(502, "Garmin Connect nie odpowiada poprawnie. Spróbuj później.")
    except Exception:
        log.exception("Nieoczekiwany błąd analizy")
        raise HTTPException(500, "Wewnętrzny błąd analizy. Szczegóły w logach serwera.")
    finally:
        _analysis_lock.release()


def _analyze(email: str, password: str, start: date, end: date, user_hr_rest, user_hr_max) -> dict:
    client = gs.get_client(email, password)
    gw = gs.Gateway(client)
    report = sync.SyncReport()
    daily_start = max(start - timedelta(days=HRV_LEAD_DAYS), end - timedelta(days=DAILY_MAX_DAYS))

    # 1) dociągnij braki do bazy, 2) licz wyłącznie z bazy
    sync.sync_range(gw, email, start, end, daily_start, report)

    def get_details(ids: list[int]) -> list[dict]:
        sync.sync_details(gw, email, ids, report)
        return [store.load_details(email, i) for i in ids]

    try:
        result = analysis.analyze(
            activities=store.load_activities(email, start.isoformat(), end.isoformat()),
            daily_rows=store.load_daily(email, daily_start.isoformat(), end.isoformat()),
            daily_start=daily_start,
            end=end,
            user_hr_rest=user_hr_rest,
            user_hr_max=user_hr_max,
            get_details=get_details,
        )
    finally:
        if gw.calls:
            gs.persist_tokens(email, client)
        log.info("Analiza %s..%s: %d zapytań do Garmina%s", start, end, report.calls,
                 " (częściowa)" if report.partial else "")

    result["sync"] = {"calls": report.calls, "partial": report.partial, "notes": report.notes}
    return result
