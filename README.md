# Logische Satzerkennung mit Transformer

Dieses Projekt implementiert ein Transformer-basiertes Modell zur Erkennung logisch sinnvoller Sätze. Das System kann zwischen semantisch korrekten Aussagen und grammatikalisch richtigen, aber logisch unsinnigen Sätzen unterscheiden.

## 🎯 Überblick

Das Modell klassifiziert Sätze in zwei Kategorien:
- **Logisch (1)**: Faktisch korrekte Aussagen (z.B. "Die Sonne geht im Osten auf")
- **Nicht logisch (0)**: Semantisch unsinnige Aussagen (z.B. "Der Tisch ist traurig")

### Kernfunktionen

- Eigenständiges Embedding und Transformer-Modell
- Tokenisierung und Vokabulargenerierung
- Pre-Normalization Transformer-Architektur mit Multi-Head Attention
- Umfangreiche Diagnose- und Analysewerkzeuge
- REST-API für Produktivbetrieb

## 📋 Projektstruktur

```
├── config.json             # Modellkonfiguration
├── data/                   # Datensätze
│   └── sentences.csv       # Trainingsdaten
│   └── vocab.json          # Generiertes Vokabular
├── models/                 # Gespeicherte Modelle
├── src/                    # Quellcode
│   ├── api.py              # FastAPI-Schnittstelle
│   ├── dataset.py          # Daten-Handling und Tokenisierung
│   ├── diagnostic_tool.py  # Umfassendes Diagnosewerkzeug
│   ├── evaluate.py         # Modellauswertung
│   ├── generate_sentences_claude.py # Datengenerierung mit Claude
│   ├── generate_sentences_gpt4o.py  # Datengenerierung mit GPT-4o
│   ├── model.py            # Transformer-Modell
│   ├── rebuild-vocab.py    # Vokabular-Neuerstellung
│   ├── test_dataset.py     # Dataset-Tests
│   ├── test_model.py       # Modell-Tests
│   └── train.py            # Trainingslogik
├── docs/                   # Zusätzliche Dokumentation
└── requirements.txt        # Abhängigkeiten
```

## 🛠️ Installation

### Voraussetzungen

- Python 3.10+
- PyTorch
- FastAPI (für API-Betrieb)
- Pandas, NumPy, Scikit-learn, Matplotlib

### Einrichtung

1. Repository klonen:
   ```bash
   git clone https://github.com/yourusername/sentence-logic-transformer.git
   cd sentence-logic-transformer
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unter Windows: venv\Scripts\activate
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Umgebungsvariablen für API-Zugriff (optional für Datengenerierung):
   Erstelle eine `.env`-Datei im Wurzelverzeichnis:
   ```
   OPENAI_API_KEY=dein_openai_key
   ANTHROPIC_API_KEY=dein_anthropic_key
   ```

## 📊 Datensatz

Der Datensatz besteht aus deutschen Sätzen, die als "logisch" oder "nicht logisch" gekennzeichnet sind. Es wird eine `sentences.csv` mit folgendem Format verwendet:

```csv
sentence,label
"Die Sonne geht im Osten auf.",1
"Wasser besteht aus Wasserstoff und Sauerstoff.",1
"Der Tisch ist traurig über die Situation.",0
"Berge können schwimmen und Flüsse klettern.",0
```

### Datengenerierung

Das Projekt enthält zwei Skripte zur Datengenerierung:

- **Mit OpenAI GPT-4o**:
  ```bash
  python src/generate_sentences_gpt4o.py --num_per_type 100 --balance-categories
  ```

- **Mit Anthropic Claude**:
  ```bash
  python src/generate_sentences_claude.py --num_per_type 100 --balance-categories
  ```

Weitere Optionen:
- `--force-logical`: Generiert nur logische Sätze
- `--force-illogical`: Generiert nur unlogische Sätze
- `--stats`: Zeigt nur Statistiken des vorhandenen Datensatzes

## 🧠 Modellarchitektur

Das Modell besteht aus drei Hauptkomponenten:

1. **Embedding-Schicht**:
   - Wandelt Wörter in numerische Embeddings um
   - Positionsenkodierung für sequentielle Information

2. **Transformer-Encoder**:
   - Multi-Head Attention für kontextuelle Verarbeitung
   - Pre-Normalization für stabiles Training
   - Residual-Verbindungen für besseren Gradientenfluss

3. **Klassifikationsschicht**:
   - Pooling der Tokenrepräsentationen
   - Lineare Projektion für binäre Klassifikation

### Konfiguration

Die `config.json` steuert die Modellparameter:

```json
{
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "num_classes": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.0001
}
```

## 🚀 Training

Das Training des Modells erfolgt mit:

```bash
cd src
python train.py
```

Mit Neuaufbau des Vokabulars:

```bash
python train.py --rebuild-vocab
```

Das Modell wird in `models/transformer_model.pth` gespeichert.

## 📈 Evaluation und Diagnose

Das Projekt bietet umfangreiche Diagnosefunktionen:

```bash
python src/diagnostic_tool.py --all --visualize --output-dir results
```

Einzelne Diagnoseoptionen:
- `--evaluate`: Modellbewertung
- `--vocab`: Vokabularanalyse
- `--errors`: Fehleranalyse
- `--test`: Beispielsätze testen
- `--sentence "Dein Testsatz"`: Einzelnen Satz testen

Für schnelle Auswertung:
```bash
python src/evaluate.py
```

## 🌐 API-Nutzung

Starten der API:

```bash
cd src
uvicorn api:app --reload
```

Die API ist unter http://localhost:8000 erreichbar mit folgenden Endpunkten:

- **GET /**: Willkommensseite
- **GET /health**: Healthcheck
- **GET /vocab_info**: Vokabularinformationen
- **GET /test**: Standardbeispiele testen
- **POST /predict**: Satzvorhersage

Beispielanfrage:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "Die Sonne scheint."}'
```

Antwort:
```json
{
  "sentence": "Die Sonne scheint.",
  "prediction": "logisch",
  "confidence": 0.9876,
  "token_count": 4,
  "unknown_words": null,
  "unknown_ratio": 0.0,
  "processing_time": 5.432
}
```

Eine Swagger-UI ist unter http://localhost:8000/docs verfügbar.

## 🐳 Docker

Das Projekt kann containerisiert werden:

```bash
# Docker-Image bauen
docker build -t sentence-logic-api .

# Container starten
docker run -p 8000:8000 sentence-logic-api
```

## 🔧 Fehlerbehebung

### Häufige Probleme

1. **Unbekannte Wörter**: Überprüfen Sie mit `diagnostic_tool.py --vocab`, ob wichtige Wörter im Vokabular fehlen. Bei Bedarf:
   ```bash
   python src/rebuild-vocab.py
   ```

2. **Falsche Vorhersagen**: Führen Sie eine Fehleranalyse durch:
   ```bash
   python src/diagnostic_tool.py --errors --visualize
   ```

3. **API-Fehler**: Überprüfen Sie, ob die richtigen Modell- und Vokabulardateien geladen wurden.

## 📝 Leistung und Ergebnisse

Bei einem ausgewogenen Datensatz mit ~1000 Beispielen kann das Modell eine Genauigkeit von über 95% erreichen. Die Leistung hängt stark von der Qualität und Größe der Trainingsdaten ab.

Empfehlungen für optimale Ergebnisse:
- Mindestens 500 Sätze je Klasse
- Ausgewogene Verteilung verschiedener Satztypen
- Vokabular mit häufigen Wörtern und Domainbegriffen

## 🤝 Mitwirken

Beiträge sind willkommen! Mögliche Verbesserungen:
- Erweiterte Tokenisierungsmethoden
- Mehrsprachige Unterstützung
- Feinere Klassifikationskategorien
- Integration in größere NLP-Pipelines

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
