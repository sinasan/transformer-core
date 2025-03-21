import os
import sys
import logging
import time
import json
import torch

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

# Pfad für den Import aus src korrigieren
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if current_dir.endswith('src') else current_dir
sys.path.append(project_root)

# Module importieren
if current_dir.endswith('src'):
    # Wenn wir im src-Verzeichnis sind
    from dataset import SentenceDataset
    from model import SimpleTransformer
else:
    # Wenn wir im Projektroot oder woanders sind
    from src.dataset import SentenceDataset
    from src.model import SimpleTransformer

# Konfigurieren des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hilfsfunktion für absolute Pfade
def get_absolute_path(relative_path):
    """Konvertiert relativen Pfad in absoluten Pfad basierend auf dem Skriptverzeichnis"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

# FastAPI-App erstellen
app = FastAPI(
    title="Transformer API für logische Sätze",
    description="API zum Klassifizieren von Sätzen als logisch oder nicht logisch",
    version="1.0.0"
)

# CORS-Middleware hinzufügen für Anfragen von anderen Domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion einschränken!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modell beim Start der Anwendung laden
# Globale Variablen für das Modell und Dataset
MODEL = None
DATASET = None

# Datenmodelle für die API
class SentenceInput(BaseModel):
    sentence: str = Field(..., example="Die Sonne geht im Osten auf", description="Der zu analysierende Satz")

class SentenceResponse(BaseModel):
    sentence: str = Field(..., description="Der analysierte Satz")
    prediction: str = Field(..., description="Vorhersage: 'logisch' oder 'nicht logisch'")
    confidence: float = Field(..., description="Konfidenz der Vorhersage (0-1)")
    token_count: int = Field(..., description="Anzahl der Tokens im Satz")
    unknown_words: Optional[List[str]] = Field(None, description="Liste der unbekannten Wörter im Satz")
    unknown_ratio: Optional[float] = Field(None, description="Anteil unbekannter Wörter im Satz")
    processing_time: float = Field(..., description="Verarbeitungszeit in Millisekunden")

@app.on_event("startup")
async def startup_event():
    """Wird beim Start der API ausgeführt - Lädt Modell und Dataset"""
    global MODEL, DATASET

    try:
        # Konfiguration laden
        config_path = get_absolute_path("../config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Konfiguration geladen aus: {config_path}")

        # Vokabular laden oder generieren
        data_path = get_absolute_path("../data/sentences.csv")
        vocab_path = get_absolute_path("../data/vocab.json")

        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            DATASET = SentenceDataset(csv_file=data_path, vocab=vocab)
            logger.info(f"Vokabular aus {vocab_path} geladen mit {len(vocab)} Tokens")
        else:
            DATASET = SentenceDataset(csv_file=data_path)
            logger.info(f"Vokabular aus Datensatz generiert mit {len(DATASET.vocab)} Tokens")

        # Modell initialisieren und Gewichte laden
        MODEL = SimpleTransformer(vocab_size=len(DATASET.vocab))
        model_path = get_absolute_path("../models/transformer_model.pth")
        MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        MODEL.eval()
        logger.info(f"Modell geladen aus: {model_path}")

        # Kurzer Test zur Validierung
        test_sentence = "Die Sonne scheint."
        tokens = DATASET.tokenize(test_sentence)
        tensor = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            output = MODEL(tensor)
        logger.info(f"Modell-Test erfolgreich. Ausgabe-Shape: {output.shape}")

    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {str(e)}")
        # Wir lassen die Exception durchsickern, damit die API nicht startet, wenn das Modell nicht geladen werden kann
        raise

@app.get("/")
async def root():
    """API-Willkommensnachricht"""
    return {
        "message": "Transformer API für logische Sätze ist aktiv",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health():
    """Healthcheck-Endpunkt für Monitoring"""
    if MODEL is None or DATASET is None:
        raise HTTPException(status_code=500, detail="Modell oder Dataset nicht geladen")
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/vocab_info")
async def vocab_info():
    """Informationen über das Vokabular"""
    if DATASET is None:
        raise HTTPException(status_code=500, detail="Dataset nicht geladen")

    # Anzahl der unbekannten Wörter berechnen
    unknown_tokens = 0
    for sentence in DATASET.sentences[:min(100, len(DATASET.sentences))]:
        tokens = DATASET.tokenize(sentence)
        unknown_tokens += tokens.count(0) if 0 in tokens else 0

    return {
        "vocab_size": len(DATASET.vocab),
        "samples_analyzed": min(100, len(DATASET.sentences)),
        "unknown_tokens_in_samples": unknown_tokens,
        "pad_token_id": 0,
        "sample_tokens": list(DATASET.vocab.items())[:10]
    }

@app.post("/predict", response_model=SentenceResponse)
async def predict(input: SentenceInput):
    """Einen Satz analysieren und die logische Klassifikation vorhersagen"""
    if MODEL is None or DATASET is None:
        raise HTTPException(status_code=500, detail="Modell oder Dataset nicht geladen")

    start_time = time.time()
    try:
        # Informationen über den Satz sammeln
        sentence = input.sentence
        processed = DATASET.preprocess_text(sentence)
        words = processed.split()

        # Tokenisieren
        tokens = DATASET.tokenize(sentence)

        # Unbekannte Wörter identifizieren
        unknown_words = []
        for word in words:
            if word not in DATASET.vocab and word != "<PAD>":
                unknown_words.append(word)

        # Tensor erstellen und Vorhersage durchführen
        tensor_tokens = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            logits = MODEL(tensor_tokens)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            pred_idx = torch.argmax(logits, dim=1).item()

        # Ergebnis ermitteln
        result = "logisch" if pred_idx == 1 else "nicht logisch"

        # Verarbeitungszeit berechnen (in ms)
        processing_time = (time.time() - start_time) * 1000

        # Detaillierte Vorhersage-Informationen
        logger.info(f"Vorhersage: '{sentence}' -> {result} (Konfidenz: {confidence:.4f})")
        if unknown_words:
            logger.warning(f"Unbekannte Wörter in '{sentence}': {unknown_words}")

        # Ergebnis zurückgeben
        return {
            "sentence": sentence,
            "prediction": result,
            "confidence": confidence,
            "token_count": len(tokens),
            "unknown_words": unknown_words if unknown_words else None,
            "unknown_ratio": len(unknown_words) / len(words) if words else 0,
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Fehler bei der Vorhersage für '{input.sentence}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Vorhersage: {str(e)}"
        )

@app.get("/test")
async def test():
    """Test des Modells mit einigen Beispielsätzen"""
    if MODEL is None or DATASET is None:
        raise HTTPException(status_code=500, detail="Modell oder Dataset nicht geladen")

    examples = [
        "Die Sonne geht im Osten auf.",
        "Wasser besteht aus Wasserstoff und Sauerstoff.",
        "Der Tisch ist traurig über die Situation.",
        "Berge können schwimmen und Flüsse klettern.",
        "Computer arbeiten mit elektrischen Signalen.",
        "Die meisten Menschen haben zwei Beine und zwei Arme."
    ]

    results = []
    for sentence in examples:
        try:
            # Tokenisieren und Tensor erstellen
            processed = DATASET.preprocess_text(sentence)
            words = processed.split()
            tokens = DATASET.tokenize(sentence)

            # Unbekannte Wörter identifizieren
            unknown_words = []
            for word in words:
                if word not in DATASET.vocab and word != "<PAD>":
                    unknown_words.append(word)

            tensor_tokens = torch.tensor(tokens).unsqueeze(0)

            # Vorhersage durchführen
            with torch.no_grad():
                logits = MODEL(tensor_tokens)
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max().item()
                pred_idx = torch.argmax(logits, dim=1).item()

            result = "logisch" if pred_idx == 1 else "nicht logisch"
            expected = "logisch" if sentence.startswith(("Die Sonne", "Wasser", "Computer", "Die meisten")) else "nicht logisch"

            results.append({
                "sentence": sentence,
                "prediction": result,
                "expected": expected,
                "correct": result == expected,
                "confidence": confidence,
                "unknown_words": unknown_words if unknown_words else None,
                "unknown_ratio": len(unknown_words) / len(words) if words else 0
            })
        except Exception as e:
            results.append({
                "sentence": sentence,
                "error": str(e)
            })

    return {
        "test_results": results,
        "accuracy": sum(1 for r in results if "correct" in r and r["correct"]) / len(results)
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware zum Loggen aller Anfragen"""
    start_time = time.time()
    path = request.url.path
    method = request.method

    # Request-Body nicht loggen (kann sensible Daten enthalten)
    logger.info(f"{method} {path} - Anfrage empfangen")

    # Anfrage weiterleiten
    response = await call_next(request)

    # Verarbeitungszeit und Statuscode loggen
    process_time = (time.time() - start_time) * 1000
    logger.info(f"{method} {path} - Status: {response.status_code}, Zeit: {process_time:.2f}ms")

    return response
