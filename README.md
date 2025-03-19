# Logische Satz-Erkennung mit Transformer

Dieses Projekt implementiert einen Transformer-basierten Ansatz zur Erkennung, ob ein Satz logisch sinnvoll ist oder nicht. Das Modell kann mit hoher Genauigkeit zwischen logisch kohärenten Sätzen und solchen, die grammatikalisch korrekt, aber semantisch unsinnig sind, unterscheiden.

## Projektstruktur

```
├── config.json             # Konfigurationsparameter für das Modell
├── data/                   # Datensätze
│   └── sentences_demo.csv  # Demo-Datensatz mit 200 Beispielsätzen
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

## Projektergebnisse

Das trainierte Modell erreicht auf einem Datensatz mit ~1300 Beispielen eine Genauigkeit von 100%, mit perfekten Precision und Recall-Werten für beide Klassen. Für optimale Ergebnisse empfiehlt sich ein Datensatz mit mindestens 1000 Beispielen.

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

4. Demo-Datensatz vorbereiten:
   ```bash
   # Falls noch nicht vorhanden, die Demo-Datei umbenennen
   cp data/sentences_demo.csv data/sentences.csv
   ```

## Datensatz generieren oder erweitern

Der Demo-Datensatz enthält 200 Beispielsätze, die für erste Tests ausreichen. Für bessere Ergebnisse kann ein größerer Datensatz generiert werden:

```bash
cd src
python generate_sentences.py --num_per_type 500
```

Dies erzeugt 500 logische und 500 unlogische Sätze in `data/sentences.csv`.

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
    "embedding_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "num_classes": 2,
    "batch_size": 16,
    "num_epochs": 25,
    "learning_rate": 0.0002,
    "dropout": 0.3
}
```

Weitere Parameter wie `weight_decay`, `scheduler` und `early_stopping` sind in der Konfiguration vorbereitet, werden aber vom aktuellen Code noch nicht vollständig genutzt.

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

## Modell-Architektur

Das Transformer-Modell in `model.py` nutzt die folgende Architektur:
- PyTorch's `nn.MultiheadAttention` für effizientes und stabiles Multi-Head-Attention
- Positional Encoding zur Erhaltung der Sequenzinformationen
- Pre-Normalization für stabileres Training
- Ein vereinfachtes, aber effektives Feedforward-Netzwerk
- Durchdachte Aggregation der Token-Repräsentationen mit Berücksichtigung von Padding

Die Implementierung ist auf Effizienz und Stabilität optimiert, mit besonderem Augenmerk auf robuste Dimensionsbehandlung.

## Hinweise

- Der mitgelieferte Demo-Datensatz enthält 200 Beispielsätze und reicht für erste Tests
- Für optimale Ergebnisse wird ein Datensatz mit mindestens 1000 Sätzen empfohlen
- Die Generierung des Datensatzes kann aufgrund der API-Anfragen einige Zeit in Anspruch nehmen
- Mit ausreichenden Daten konvergiert das Modell typischerweise nach 15-20 Epochen
