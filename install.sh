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
python3 -m venv venv

echo "📚 Instalowanie zależności z użyciem pip..."
./venv/bin/pip install --upgrade pip

# Jeśli masz plik requirements.txt w repozytorium, skrypt go użyje.
# Jeśli nie, zainstaluje pakiety z listy "z palca".
if [ -f "requirements.txt" ]; then
    echo "Znaleziono plik requirements.txt. Instaluję..."
    ./venv/bin/pip install -r requirements.txt
else
    echo "Brak pliku requirements.txt. Instaluję standardowy pakiet..."
    ./venv/bin/pip install fastapi uvicorn pandas garminconnect jinja2 python-multipart gunicorn
fi

echo "⚙️ Tworzenie konfiguracji systemd (garmin.service)..."
SERVICE_FILE="/etc/systemd/system/garmin.service"

# Dynamiczne generowanie pliku usługi z uwzględnieniem ścieżki
cat <<EOF > $SERVICE_FILE
[Unit]
Description=Garmin Trends Analyzer
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind [::]]:8000
Restart=always

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