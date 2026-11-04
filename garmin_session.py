"""Logowanie do Garmin Connect i wszystkie wywołania API w jednym miejscu.

Zasady ochrony przed banem:
- hasło idzie do SSO Garmina tylko przy pierwszym logowaniu albo zmianie hasła;
  potem używamy zapisanych tokenów (odświeżają się same),
- najwyżej LOGIN_LIMIT prawdziwych logowań SSO na LOGIN_WINDOW_S,
- każde zapytanie: odstęp >= MIN_INTERVAL_S + losowy jitter, budżet na jedną analizę,
- po 429: przerwa dla całej aplikacji, zapamiętana w bazie (przetrwa restart).
"""
import hashlib
import hmac
import logging
import random
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi import HTTPException
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectTooManyRequestsError,
)

import store

log = logging.getLogger("biegam.garmin")

SESSION_TTL_S = 6 * 3600
LOGIN_LIMIT, LOGIN_WINDOW_S = 5, 15 * 60
COOLDOWN_S = 30 * 60
MIN_INTERVAL_S, JITTER_S = 1.0, 0.5
CALL_BUDGET = 80
PBKDF2_ITER = 200_000

FATAL = (GarminConnectTooManyRequestsError, GarminConnectAuthenticationError)


class BudgetExceeded(Exception):
    """Wyczerpany limit zapytań na jedną analizę – reszta przy następnym razie."""


# --- przerwa po 429 -------------------------------------------------------

def cooldown_remaining() -> int:
    return max(0, int(store.get_meta("*", "cooldown_until", 0) - time.time()))


def start_cooldown() -> None:
    until = time.time() + COOLDOWN_S
    store.set_meta("*", "cooldown_until", until)
    log.warning("Garmin zwrócił 429 – przerwa do %s", time.ctime(until))


# --- zapytania z ograniczeniem tempa -------------------------------------

_throttle_lock = threading.Lock()
_last_call = 0.0


class Gateway:
    """Opakowanie klienta Garmin: tempo, budżet, liczenie zapytań, obsługa 429."""

    def __init__(self, client: Garmin, budget: int = CALL_BUDGET):
        self.client, self.budget, self.calls = client, budget, 0

    def call(self, method: str, *args, **kwargs):
        global _last_call
        if self.calls >= self.budget:
            raise BudgetExceeded(method)
        with _throttle_lock:
            wait = _last_call + MIN_INTERVAL_S + random.uniform(0, JITTER_S) - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        self.calls += 1
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except GarminConnectTooManyRequestsError:
            start_cooldown()
            raise


# --- logowanie -----------------------------------------------------------

@dataclass
class _Session:
    pw_digest: bytes
    created: float
    client: Garmin


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()
_login_times: deque[float] = deque()


def _digest(salt: bytes, email: str, password: str) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt + email.encode(), PBKDF2_ITER)


def _token_dir(email: str):
    d = store.DATA_DIR / "tokens" / hashlib.sha256(email.encode()).hexdigest()[:20]
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _take_login_slot() -> None:
    now = time.time()
    with _sessions_lock:
        while _login_times and now - _login_times[0] > LOGIN_WINDOW_S:
            _login_times.popleft()
        if len(_login_times) >= LOGIN_LIMIT:
            raise HTTPException(429, "Zbyt wiele prób logowania. Spróbuj za kilkanaście minut.")
        _login_times.append(now)


def get_client(email: str, password: str) -> Garmin:
    """Zwraca zalogowanego klienta. `email` musi być już znormalizowany (lower/strip)."""
    user = store.get_user(email)
    known_pw = user is not None and hmac.compare_digest(
        user["pw_digest"], _digest(user["salt"], email, password)
    )
    if known_pw:
        with _sessions_lock:
            s = _sessions.get(email)
            if s and time.time() - s.created < SESSION_TTL_S:
                return s.client
        token_dir = _token_dir(email)
        if (token_dir / "garmin_tokens.json").exists():
            # Hasło zweryfikowane lokalnie -> wznawiamy z tokenów, bez SSO.
            client = Garmin(email, password)
            client.login(tokenstore=str(token_dir))
            return _remember(email, user["pw_digest"], client)

    # Nowy użytkownik, inne hasło albo brak tokenów: pełne logowanie SSO.
    # Celowo BEZ tokenstore – tylko tak Garmin naprawdę sprawdzi hasło.
    _take_login_slot()
    client = Garmin(email, password)
    client.login()
    salt = secrets.token_bytes(16)
    digest = _digest(salt, email, password)
    store.upsert_user(email, salt, digest)
    persist_tokens(email, client)
    return _remember(email, digest, client)


def _remember(email: str, digest: bytes, client: Garmin) -> Garmin:
    with _sessions_lock:
        _sessions[email] = _Session(digest, time.time(), client)
    return client


def persist_tokens(email: str, client: Garmin) -> None:
    """Zapisuje (ew. odświeżone) tokeny, żeby po restarcie nie logować się ponownie."""
    try:
        client.client.dump(str(_token_dir(email)))
    except Exception:
        log.warning("Nie udało się zapisać tokenów", exc_info=True)


def drop_session(email: str) -> None:
    with _sessions_lock:
        _sessions.pop(email, None)
