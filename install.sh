#!/bin/bash

# Zatrzymanie skryptu w przypadku błędu
set -e

echo "🚀 Rozpoczynam instalację aplikacji Garmin Dashboard..."

# Sprawdzenie, czy skrypt jest uruchomiony jako root
if [ "$EUID" -ne 0 ]; then
  echo "❌ Uruchom ten skrypt jako root (np. 'sudo ./install.sh' lub po zalogowaniu na roota)."
  exit 1
fi

# Pobranie pełnej ścieżki do obecnego katalogu (tam, gdzie sklonowano repo)
APP_DIR=$(pwd)
echo "📂 Wykryty katalog aplikacji: $APP_DIR"

echo "📦 Aktualizacja repozytoriów i instalacja pakietów systemowych..."
apt update
apt install -y python3-pip python3-venv

echo "🐍 Tworzenie środowiska wirtualnego Pythona (venv)..."
# --clear: zawsze świeży venv. Skrypty w venv/bin mają na sztywno wpisaną ścieżkę,
# więc po przeniesieniu katalogu (np. /root -> /opt) stary venv przestaje działać.
python3 -m venv --clear venv

echo "📚 Instalowanie zależności z użyciem pip..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt

echo "👤 Tworzenie użytkownika systemowego 'garmin' (bez powłoki i hasła)..."
APP_USER=garmin
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin "$APP_USER"
# Kod zostaje własnością roota (usługa nie może go zmienić, a `git pull` jako root
# nie zgłasza „dubious ownership”). Usługa pisze tylko do data/ (baza i tokeny).
mkdir -p "$APP_DIR/data"
chown -R "$APP_USER":"$APP_USER" "$APP_DIR/data"
chmod 700 "$APP_DIR/data"
if ! runuser -u "$APP_USER" -- test -r "$APP_DIR/main.py"; then
  echo "❌ Użytkownik '$APP_USER' nie ma dostępu do $APP_DIR (np. katalog w /root)."
  echo "   Przenieś repozytorium np. do /opt/biegam i uruchom skrypt ponownie."
  exit 1
fi

echo "⚙️ Tworzenie konfiguracji systemd (garmin.service)..."
SERVICE_FILE="/etc/systemd/system/garmin.service"

# Dynamiczne generowanie pliku usługi z uwzględnieniem ścieżki
cat <<EOF > $SERVICE_FILE
[Unit]
Description=Garmin Trends Analyzer
After=network.target

[Service]
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python -m gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind [::]:8000 --timeout 180 --access-logfile -
Restart=always
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR/data

[Install]
WantedBy=multi-user.target
EOF

echo "🔄 Przeładowanie systemd i uruchamianie usługi..."
systemctl daemon-reload
systemctl enable garmin
systemctl restart garmin

echo "✅ Instalacja zakończona sukcesem!"
echo "--------------------------------------------------------"
echo "Aplikacja nasłuchuje lokalnie na porcie 8000."
echo "Aby sprawdzić status działania, wpisz: systemctl status garmin"
echo "Teraz wystarczy podpiąć ten port pod domenę w panelu mikr.us!"
echo "--------------------------------------------------------"