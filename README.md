# Logische Satz-Erkennung mit Transformer

Dieses Projekt implementiert einen Transformer-basierten Ansatz zur Erkennung, ob ein Satz logisch sinnvoll ist.

## Projektstruktur

```
├── config.json             # Konfigurationsparameter für das Modell
├── data/                   # Datensätze (werden nicht im Repository gespeichert)
├── models/                 # Trainierte Modelle (werden nicht im Repository gespeichert)
├── src/                    # Quellcode
│   ├── api.py              # FastAPI-Schnittstelle für Inferenz
│   ├── dataset.py          # Daten-Handling und Tokenisierung
│   ├── evaluate.py         # Modellauswertung und Visualisierung
│   ├── generate_sentences.py # Tool zur Datensatzgenerierung
│   ├── model.py            # Definition des Transformer-Modells
│   ├── test_dataset.py     # Tests für das Dataset
│   ├── test_model.py       # Tests für das Modell
│   └── train.py            # Trainingslogik
└── .env                    # Umgebungsvariablen (API-Keys, etc.)
```

## Setup

1. Repository klonen:
   ```bash
   git clone https://github.com/sinasan/logical-sentence-transformer.git
   cd logical-sentence-transformer
   ```

2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   pip install -r requirements.txt 
   ```

3. `.env`-Datei erstellen:
   ```
   OPENAI_API_KEY=dein_api_key_hier
   ```

## Datensatz generieren

Der Datensatz wird nicht im Repository gespeichert, aber kann mit dem beigefügten Skript generiert werden:

```bash
cd src
python generate_sentences.py --num_per_type 500
```

Dies erzeugt 500 logische und 500 unlogische Sätze in `data/sentences.csv`. Je nach Bedarf kann die Anzahl angepasst werden.

### Optionen für Datengenerierung:

- Nur Statistiken anzeigen: `python generate_sentences.py --stats`
- Vorhandenen Datensatz erweitern: `python generate_sentences.py --num_per_type 100`
- Bestehende Datei laden und in neuer speichern: `python generate_sentences.py --input input.csv --output output.csv --num_per_type 200`

## Modell trainieren

Nach der Datengenerierung kann das Modell trainiert werden:

```bash
cd src
python train.py
```

Die Trainingsparameter können in `config.json` angepasst werden:

```json
{
    "embedding_dim": 256,
    "num_heads": 8,
    "num_layers": 2,
    "num_classes": 2,
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 0.0001,
    "dropout": 0.2,
    "use_relative_pos": true,
    "activation": "gelu"
}
```

## Modell evaluieren

Nach dem Training kann das Modell evaluiert werden:

```bash
cd src
python evaluate.py
```

Dies erzeugt einen Bericht mit Genauigkeit, Precision, Recall und F1-Score, sowie eine Confusion Matrix.

## API starten

Um die API zu starten und Inferenzen durchzuführen:

```bash
cd src
uvicorn api:app --reload
```

Die API ist dann unter http://localhost:8000 erreichbar. Es gibt einen Endpunkt `/predict`, der einen Satz entgegennimmt und zurückgibt, ob er logisch ist oder nicht.

Beispiel-Anfrage:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "Der Hund bellt laut im Garten."}'
```

## Modell anpassen

Das Transformer-Modell in `model.py` enthält erweiterte Attention-Mechanismen mit:
- Relativem Positions-Encoding
- Verbesserten Feed-Forward-Netzwerken mit Gated Linear Units
- Skalierten Residualverbindungen

Diese Verbesserungen können in der Konfigurationsdatei aktiviert oder deaktiviert werden.

## Hinweise

- Um gute Ergebnisse zu erzielen, wird ein Datensatz mit mindestens 1000 Sätzen (je 500 logisch und unlogisch) empfohlen.
- Die Generierung des Datensatzes kann aufgrund der API-Anfragen einige Zeit in Anspruch nehmen.
- Das Modell benötigt je nach Datensatzgröße etwa 10-20 Epochen für gute Ergebnisse.
