FROM python:3.10-slim

WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Python aktualisieren und Abhängigkeiten installieren
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi uvicorn numpy pandas scikit-learn

# Projektdateien kopieren
COPY ./src ./src
COPY ./data ./data
COPY ./models ./models
COPY ./config.json ./config.json

# Port freigeben
EXPOSE 8000

# API starten
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
