# Motivation

Ich will ein kleines Embedding-Model selbst erstellen und darauf ein kleines Transformer-Model entwerfen.

Ich möchte das System ganzheitlich verstehen. Thema ist egal. Ob es Objekte in Bildern benennen kann, oder sagen kann ob Elefanten Schnitzel essen.

# Projekt: transformer-core

Einfaches, aber vollständiges Transformer-basiertes Modell, das entscheiden kann, ob ein gegebener Satz logisch Sinn ergibt oder nicht.

Beispielhafte Eingabe:
* „Elefanten essen gerne Äpfel" → Sinnvoll
* „Elefanten essen Schnitzel" → Weniger sinnvoll

Damit hätten wir ein kleines, selbst erstelltes Embedding-Modell (für Wörter/Sätze) und ein einfaches Transformer-Modell, das die Bedeutung eines Satzes lernt. Dockerisierung und Optimierung währen dann nächste Schritte.

## Warum dieses Projekt?

* Es demonstriert die vollständige Pipeline (Embedding → Transformer → Klassifikation).
* Man kann nachvollziehen, wie Embeddings erstellt werden.
* Lernen, wie ein Transformer-Modell aufgebaut ist und wie Attention funktioniert.

## Vorgehensweise in Schritten:

1. **Erstellung eines Datensatzes**  
   Eine kleine Menge an Trainingsdaten (Sätze mit Labels sinnvoll/nicht sinnvoll).

2. **Eigenes Embedding erstellen**  
   Aufbau eines simplen Wort-Embeddings (z.B. auf Basis eines kleinen, zufällig initialisierten Lookup-Tables).

3. **Aufbau eines einfachen Transformer-Encoders**  
   Ein kleiner Transformer (Self-Attention + Feed-Forward Layers), um die Satzlogik zu lernen.

4. **Training und Inferenz**  
   Trainieren und evaluieren auf Beispiel-Daten.

# Technologien & Frameworks:

* Python mit PyTorch
* Keine externen Embeddings wie GloVe oder FastText (nur selbsterstelltes, eigenes kleines Embedding).

# Projektplan: Embedding + Transformer-Modell (logische Satzerkennung)

## Projektziel:

Implementiere von Grund auf ein kleines Embedding- und Transformer-Modell in PyTorch, das einfache Sätze danach bewertet, ob sie logisch sinnvoll sind oder nicht.

## Technologien:

* Python (3.10+)
* PyTorch
* NumPy
* Optional: Docker (Containerisierung, reproduzierbare Umgebung)

## Roadmap

### Schritt 1: Projektvorbereitung & Setup

* [1.1] Anlegen einer sauberen Python-Umgebung (venv oder Docker-Container)
* [1.2] Installation von benötigten Paketen:
  * torch, torchvision
  * numpy
  * pandas (optional, für komfortables Datenhandling)
  * u.s.w.

### Schritt 2: Datensatz erstellen und vorbereiten

* [2.1] Erstellung eines kleinen Beispiel-Datensatzes:
  * Etwa 50-100 kurze Sätze (z.B. 50 sinnvoll, 50 unsinnig)
  * Klar definiertes Format (z.B. CSV mit Spalten: Satz, Label)
* [2.2] Tokenisierung der Sätze:
  * Erstellen eines einfachen Tokenizers (Whitespace-Tokenizer oder simpelste Tokenisierung)
* [2.3] Wortschatz-Generierung (Vocabulary):
  * Erstellung eines kleinen Wörterbuches (Index ↔ Wort Mapping)
* [2.4] Datensatz-Split:
  * Training-Set (~80%), Test-Set (~20%)

### Schritt 3: Eigenes Embedding-Modell erstellen

* [3.1] Erstellen einer Embedding-Schicht (PyTorch Embedding Layer)
  * Parameter festlegen (z.B. Embedding-Dimension = 32)
* [3.2] Validierung der Embedding-Ausgabe:
  * Testen, ob Wörter korrekt in Embeddings umgewandelt werden (Shape: [batch_size, seq_len, embedding_dim])

### Schritt 4: Transformer-Modell implementieren

* [4.1] Implementierung eines Transformer-Encoder-Layers (Self-Attention):
  * Multi-Head-Attention-Schicht
  * Feedforward-Netzwerk (Position-wise Feedforward)
  * Layer Normalization und Residual Connections
* [4.2] Kombination der Schichten zum Transformer-Encoder:
  * Erstellen einer Transformer-Klasse (nn.Module) mit mehreren Encoder-Schichten
* [4.3] Ausgabeschicht für Klassifikation:
  * Pooling (z.B. Mittelwertbildung oder [CLS]-Token)
  * Fully-Connected-Schicht (Linear) für binäre Klassifikation (logisch/unlogisch)
* [4.4] Validierung der Modell-Ausgabe:
  * Sicherstellen, dass der Output die korrekte Dimension hat und Werte ausgibt ([batch_size, num_classes])

### Schritt 5: Modelltraining durchführen

* [5.1] Definition von Loss-Funktion & Optimierer:
  * CrossEntropyLoss
  * Adam-Optimierer
* [5.2] Training-Loop implementieren:
  * Vorwärtsdurchlauf
  * Backpropagation
  * Logging von Trainingsverlust und Genauigkeit
* [5.3] Evaluation auf Testdaten:
  * Berechnung von Genauigkeit und Verlust

### Schritt 6: Ergebnisse auswerten & interpretieren

* [6.1] Untersuchung der Ergebnisse:
  * Analyse der Fehler und korrekten Klassifikationen
* [6.2] Visualisierung der Ergebnisse:
  * Confusion Matrix
  * Entwicklung von Verlust & Genauigkeit über die Epochen

### Schritt 7: Optional - Deployment & Erweiterungen

* [7.1] Containerisierung mit Docker (Dockerfile erstellen)
* [7.2] Erstellen eines einfachen REST-Endpoints mit FastAPI zur Nutzung des Modells
* [7.3] Erweiterungsmöglichkeiten dokumentieren:
  * Erweiterung des Datensatzes
  * Tuning der Transformer-Architektur
  * Einbau von Positionsencodings, weiteren Regularisierungstechniken, etc.

## Zeitliche Empfehlung zur Umsetzung:

| Schritt | Aufwand (ca.) | Empfehlung Zeitplan |
|---------|---------------|---------------------|
| Schritt 1 | ~30 Min. | Tag 1 |
| Schritt 2 | ~2 Std. | Tag 1 |
| Schritt 3 | ~1 Std. | Tag 2 |
| Schritt 4 | ~3 Std. | Tag 2-3 |
| Schritt 5 | ~2 Std. | Tag 3 |
| Schritt 6 | ~1 Std. | Tag 4 |
| Schritt 7 | Optional | Tag 5 |

## Genauere Definition des Zielmodells:

Wir erstellen ein kleines Modell, das anhand einfacher Sätze erkennt, ob sie logisch korrekt oder inkorrekt sind.

### Beispiel:

* logisch korrekt:
  * „Vögel fliegen"
  * „Fische schwimmen"
* logisch inkorrekt:
  * „Alle Tiere fliegen" (nicht korrekt)

## Konkrete Datensatzbeispiele:

| Satz | Label (1 = logisch, 0 = nicht logisch) |
|------|----------------------------------------|
| Vögel fliegen | 1 |
| Fische schwimmen | 1 |
| Elefanten laufen | 1 |
| Steine schlafen | 0 |
| Bäume schwimmen | 0 |
| Alle Tiere fliegen | 0 |

## Die Modell-Architektur (Übersicht):

Das Transformer-basierte Modell umfasst drei Komponenten:

1. **Embedding-Schicht:**  
   Wörter werden in numerische Embeddings umgewandelt (lernt Semantik automatisch).

2. **Transformer-Encoder:**  
   Versteht Kontext und erkennt logische Zusammenhänge im Satz durch Self-Attention.

3. **Klassifikationsschicht:**  
   Erzeugt eine binäre Aussage: „logisch" oder „nicht logisch".

## Technische Umsetzung (Detailschritte):

### Step 1 – Vorbereitung:
* Python-Umgebung vorbereiten, PyTorch installieren.

### Step 2 – Datensatz definieren und vorbereiten:
* CSV-Datei oder Liste mit Sätzen und Labels erstellen.
* Tokenisierung der Sätze (Wörter trennen).
* Erstellung eines kleinen Vocabulary (Wort → Index).

### Step 3 – Embedding-Layer erstellen:
* PyTorch-Embedding-Schicht (z.B. Dimension = 16 oder 32).

### Step 4 – Transformer-Encoder implementieren:
* Multihead-Self-Attention-Layer + Feedforward-Netzwerk.
* Position-Encodings für die Reihenfolge der Wörter.

### Step 5 – Klassifikations-Layer hinzufügen:
* Pooling (Mittelwert der Token-Embeddings oder spezielles Klassifikations-Token).
* Lineare Schicht für binäre Entscheidung.

### Step 6 – Training und Evaluation:
* Loss-Funktion (CrossEntropy) und Optimizer (Adam).
* Training-Loop und Accuracy-Messung.

## Schritt 1 – Projektvorbereitung & Setup

### Ziel:
Eine saubere Python-Umgebung aufsetzen und alle notwendigen Pakete installieren, damit wir im nächsten Schritt mit Datensatz und Modell beginnen können.

### Python-Umgebung einrichten (virtuelle Umgebung)
Nutze hierfür Python 3.10 oder neuer.

```bash
python3 -m venv venv
source venv/bin/activate
```

Prüfe die Python-Version (>= 3.10):
```bash
python --version
```

### Pakete installieren

#### Pflichtpakete:
* PyTorch (stabilste Version)
* NumPy
* Pandas (Datenhandling)
* scikit-learn (Evaluierung)

Installiere Pakete:
```bash
# Standard
pip install torch torchvision torchaudio numpy pandas scikit-learn

# API, Rest
pip install fastapi uvicorn

# Visualisierung
pip install matplotlib seaborn
```

**Hinweis:**
torchvision und torchaudio werden standardmäßig mit PyTorch empfohlen, sind für diesen Use Case nicht zwingend notwendig, stören jedoch nicht.

### Projektstruktur erstellen

Empfohlene Struktur:
```
transformer-project/
├── data/
│   └── sentences.csv
├── models/
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
└── requirements.txt
```

### Projekt-Abhängigkeiten speichern (optional aber empfohlen):

Erstelle requirements.txt mit allen installierten Paketen:
```bash
pip freeze > requirements.txt
```

Dann jederzeit notwendige Pakete installierbar mit:
```bash
pip install -r requirements.txt
```

### Checkliste (Schritt 1 abgeschlossen, wenn):
* virtuelle Umgebung läuft (venv)
* Pakete installiert
* Projektstruktur angelegt
* Anforderungen gespeichert (requirements.txt)

## Schritt 2 – Datensatz erstellen und vorbereiten

In diesem Schritt wird ein kleiner
* klar strukturierter Datensatz erstellt
* eine einfache Tokenisierung implementiert
* Vocabulary erstellt.

### 2.1 Datensatz erstellen (sentences.csv)

Erstelle die Datei data/sentences.csv mit folgendem Inhalt:
```
sentence,label
Vögel fliegen,1
Fische schwimmen,1
Elefanten laufen,1
Menschen denken,1
Katzen miauen,1
Hunde bellen,1
Fische fliegen,0
Steine schlafen,0
Bäume schwimmen,0
Alle Tiere fliegen,0
Alle Steine essen,0
Menschen bellen,0
```

**Spalten:**
* sentence: Eingabesatz
* label: 1 (logisch) oder 0 (nicht logisch)

### 2.2 Daten laden & Tokenizer implementieren (dataset.py)

Erstelle src/dataset.py:
```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, csv_file, vocab=None):
        self.data = pd.read_csv(csv_file)
        self.sentences = self.data['sentence'].tolist()
        self.labels = self.data['label'].tolist()
        self.vocab = vocab or self.build_vocab(self.sentences)
    
    def build_vocab(self, sentences):
        words = set()
        for sent in sentences:
            words.update(sent.lower().split())
        word2idx = {word: idx+2 for idx, word in enumerate(sorted(words))}
        word2idx['<PAD>'] = 0
        word2idx['<UNK>'] = 1
        return word2idx
    
    def tokenize(self, sentence):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence.lower().split()]
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenize(sentence)
        label = self.labels[idx]
        return torch.tensor(tokens), torch.tensor(label)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    padded_sentences = torch.zeros(len(sentences), max_len).long()
    for i, sentence in enumerate(sentences):
        padded_sentences[i, :lengths[i]] = sentence
    labels = torch.tensor(labels).long()
    return padded_sentences, labels
```

**Erklärung:**
* Einfacher Whitespace-Tokenizer.
* Vocabulary: enthält spezielle Tokens <PAD> und <UNK> (unbekannt).
* collate_fn zum Auffüllen (Padding) der Sätze auf gleiche Länge.

### 2.3 Datensatz kurz testen:

Prüfe kurz, ob alles korrekt geladen wird (test_dataset.py):
```python
from dataset import SentenceDataset, collate_fn
from torch.utils.data import DataLoader

dataset = SentenceDataset(csv_file='../data/sentences.csv')
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for sentences, labels in loader:
    print('Sätze (Token-Indizes):', sentences)
    print('Labels:', labels)
    break
```

Testen per:
```bash
python test_dataset.py
```

### Checkliste (Schritt 2 abgeschlossen, wenn):
* Datensatz erstellt (sentences.csv)
* Tokenisierung & Dataset-Klasse erstellt (dataset.py)
* Tokenisierung getestet (Token-Indizes sichtbar und korrekt gepadded)

## Schritt 3 – Embedding-Layer erstellen und testen

Ziel ist es, eine Embedding-Schicht zu definieren und deren Funktion sicherzustellen.

### 3.1 Embedding-Layer implementieren (model.py)

Erstelle jetzt die Datei src/model.py:
```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_classes, num_layers=1):
        super(SimpleTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x == 0)
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        pooled = transformer_output.mean(dim=1)
        logits = self.fc(pooled)
        return logits
```

**Wichtige Parameter (Vorschläge):**
* vocab_size: Größe des Wortschatzes (len(dataset.vocab)).
* embedding_dim: 32 (kann klein bleiben, leichtes Modell).
* num_heads: 2 Attention-Heads reichen für den Anfang.
* num_classes: 2 (logisch / nicht logisch).

### 3.2 Embedding-Layer testen (test_model.py)

Erstelle src/test_model.py:
```python
import torch
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer
from torch.utils.data import DataLoader

# Lade Dataset und Vokabular
dataset = SentenceDataset(csv_file='../data/sentences.csv')

# Modellparameter
vocab_size = len(dataset.vocab)
embedding_dim = 32
num_heads = 2
num_classes = 2

# Modellinstanz erstellen
model = SimpleTransformer(vocab_size, embedding_dim, num_heads, num_classes)

# Daten laden
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Embedding testen
for sentences, labels in loader:
    logits = model(sentences)
    print("Input Shape:", sentences.shape)
    print("Output Shape:", logits.shape)
    print("Logits:", logits)
    break
```

Ausführen:
```bash
python test_model.py
```

**Erwartete Ausgabe (ca.):**
```
Input Shape: torch.Size([4, X])  # X = maximale Satzlänge
Output Shape: torch.Size([4, 2])
Logits: tensor([[..., ...],
        [..., ...],
        [..., ...],
        [..., ...]], grad_fn=<AddmmBackward0>)
```

**Bedeutung:**
* Output-Shape muss [batch_size, num_classes] sein.
* Ausgabe (Logits) sind rohe Scores, noch nicht normalisiert (Softmax erfolgt später im Training).

### Checkliste (Schritt 3 abgeschlossen, wenn):
* Embedding-Layer in model.py implementiert
* Modell testweise erfolgreich ausgeführt (test_model.py)
* Ausgabe korrekt (Input und Output Shapes passen)

## Schritt 4 – Training-Loop implementieren

Im diesem Schritt implementieren wir die vollständige Training-Logik.

### 4.1 Training-Skript (train.py) erstellen

Erstelle src/train.py:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer

# Hyperparameter
embedding_dim = 32
num_heads = 2
num_classes = 2
num_epochs = 20
batch_size = 4
learning_rate = 0.001

# Datensatz laden
dataset = SentenceDataset(csv_file='../data/sentences.csv')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Modell initialisieren
model = SimpleTransformer(
    vocab_size=len(dataset.vocab),
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_classes=num_classes
)

# Loss und Optimizer definieren
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training-Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for sentences, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(sentences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    
    # Einfache Accuracy-Messung nach jeder Epoche
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for sentences, labels in data_loader:
            outputs = model(sentences)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())
    
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}')

print("Training abgeschlossen.")
```

### 4.2 Modelltraining starten

Starte das Training mit folgendem Befehl im src/-Ordner:
```bash
python train.py
```

### 4.3 Erwartete Trainingsausgabe

```
Epoch [1/20] Loss: 0.6954 | Accuracy: 0.50
Epoch [2/20] Loss: 0.6783 | Accuracy: 0.58
...
Epoch [20/20] Loss: 0.1234 | Accuracy: 1.00
```

Die Genauigkeit sollte relativ schnell steigen und idealerweise auf 100% (bei diesem kleinen Datensatz) konvergieren.

### Checkliste (Schritt 4 abgeschlossen, wenn):
* train.py implementiert
* Training läuft fehlerfrei durch
* Verlust sinkt, Genauigkeit steigt deutlich an

## Schritt 5 – Modell evaluieren und Ergebnisse visualisieren

In diesem Schritt wird
* eine strukturierte Evaluation durchgeführt
* die Ergebnisse mit einer Confusion Matrix visualisierst, um besser zu verstehen, wie gut das Modell funktioniert.

### 5.1 Evaluationsskript (evaluate.py)

src/evaluate.py:
```python
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer

# Parameter (wie zuvor im Training definiert)
embedding_dim = 32
num_heads = 2
num_classes = 2
batch_size = 4

# Datensatz laden
dataset = SentenceDataset(csv_file='../data/sentences.csv')
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# Modell laden (ggf. trainiertes Modell hier direkt erneut laden)
model = SimpleTransformer(
    vocab_size=len(dataset.vocab),
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_classes=num_classes
)

# Für einfache Tests nutzen wir direkt trainiertes Modell (alternativ: Modell speichern und laden!)
model.eval()

# Vorhersagen und Labels sammeln
all_preds, all_labels = [], []
with torch.no_grad():
    for sentences, labels in data_loader:
        outputs = model(sentences)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['nicht logisch', 'logisch']))

# Confusion Matrix erstellen
cm = confusion_matrix(all_labels, all_preds)

# Confusion Matrix visualisieren
sns.set(font_scale=1.2)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['nicht logisch', 'logisch'], yticklabels=['nicht logisch', 'logisch'])
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
```

### 5.2 Evaluation ausführen

Führe die Evaluation aus:
```bash
python evaluate.py
```

### 5.3 Erwartete Ergebnisse:

Beispielhafte Ausgabe:
```
Classification Report:
              precision    recall  f1-score   support
nicht logisch      1.00      1.00      1.00         6
      logisch      1.00      1.00      1.00         6
    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12
```

**Confusion Matrix (Visualisierung):**
* Übersichtliche Darstellung der Ergebnisse.
* Klare Einsicht, welche Klassen ggf. verwechselt wurden.

### Checkliste (Schritt 5 abgeschlossen, wenn):
* Evaluation (evaluate.py) läuft fehlerfrei
* Classification Report und Confusion Matrix erstellt
* Ergebnisse sind verständlich und nachvollziehbar

## Schritt 6 – Modell-Deployment via REST-API (FastAPI)

Bereitstellung des trainierten Transformer-Modells über eine REST-API mit FastAPI.

### 6.1 Installation von FastAPI & Uvicorn

Installiere die benötigten Pakete in deiner Umgebung (schon während der Installation geschehen):
```bash
pip install fastapi uvicorn
```

### 6.2 REST-API erstellen (api.py)

Erstelle src/api.py:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.dataset import SentenceDataset, collate_fn
from src.model import SimpleTransformer

app = FastAPI(title="Transformer API für logische Sätze")

# Parameter (wie zuvor definiert)
embedding_dim = 32
num_heads = 2
num_classes = 2

# Dataset (nur fürs Vokabular)
dataset = SentenceDataset(csv_file='./data/sentences.csv')

# Modell laden (trainiertes Modell)
model = SimpleTransformer(
    vocab_size=len(dataset.vocab),
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_classes=num_classes
)

model.eval()

# Pydantic-Datenmodell zur Validierung der Eingabe
class SentenceInput(BaseModel):
    sentence: str

@app.post("/predict")
async def predict(input: SentenceInput):
    tokens = dataset.tokenize(input.sentence)
    tensor_tokens = torch.tensor(tokens).unsqueeze(0)  # batch-size = 1
    logits = model(tensor_tokens)
    pred = torch.argmax(logits, dim=1).item()
    result = "logisch" if pred == 1 else "nicht logisch"
    return {"sentence": input.sentence, "prediction": result}
```

### 6.3 API lokal starten

Starte den API-Server aus dem Ordner src/ mit folgendem Befehl:
```bash
uvicorn api:app --reload
```

* Die API läuft jetzt lokal unter:
  http://127.0.0.1:8000/docs
* Die automatische Dokumentation (Swagger UI) ermöglicht direkte Tests im Browser.

### 6.4 API testen

Teste direkt über die Swagger-Dokumentation (/docs) oder via CURL/Postman:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"sentence": "Elefanten essen Schnitzel"}'
```

Erwartete Antwort:
```json
{
    "sentence": "Elefanten essen Schnitzel",
    "prediction": "nicht logisch"
}
```

### Checkliste (Schritt 6 abgeschlossen, wenn):
* FastAPI läuft lokal
* REST-Endpoint /predict getestet & funktioniert

Nach Schritt 6 läuft das Modell, wenn auch mit einer falschen antwort:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"sentence": "Elefanten essen Schnitzel"}'
{"sentence":"Elefanten essen Schnitzel","prediction":"logisch"}
```
aber egal, zunächst der nächste schritt im plan.

## Schritt 7 – Docker Containerisierung

Ziel ist es, das gesamte Projekt inklusive FastAPI-API als Docker-Container zur Verfügung zu stellen. Das erleichtert das Deployment auf eine VM.

### 7.1 Dockerfile erstellen

Erstelle im Root-Ordner des Projekts (transformer-project/) eine Datei ./Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Systemabhängigkeiten installieren
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Python aktualisieren und Abhängigkeiten installieren
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi uvicorn numpy pandas scikit-learn

# Automatische Daten
RUN pip install pandas openai

# Projektdateien kopieren
COPY ./src ./src
COPY ./data ./data

# Port freigeben
EXPOSE 8000

# API starten
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Docker-Image bauen

Im Projektordner transformer-project/ in Shell ausführen:

```bash
docker build -t transformer-api .
```

### 7.3 Docker-Container starten

Starte den Container lokal:

```bash
docker run -p 8000:8000 transformer-api
```

Testen im Browser:

* http://localhost:8000/docs

oder mit curl:

```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"sentence": "Elefanten essen Schnitzel"}'
```

### Checkliste (Schritt 7 abgeschlossen, wenn):

* Dockerfile erstellt
* Docker-Image gebaut
* Container erfolgreich gestartet
* REST-API läuft fehlerfrei im Container
