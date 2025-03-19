from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
import logging
from src.dataset import SentenceDataset
from src.model import SimpleTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformer API für logische Sätze")

# Get absolute paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Korrigierte Pfade für Docker-Kompatibilität
config_path = os.path.join(PROJECT_ROOT, "config.json")
if not os.path.exists(config_path):
    # Fallback, wenn die config.json im selben Verzeichnis wie die API ist
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        # Fallback auf absoluten Pfad für Docker
        config_path = "/app/config.json"

data_path = os.path.join(PROJECT_ROOT, "data", "sentences.csv")
if not os.path.exists(data_path):
    # Fallback auf absoluten Pfad für Docker
    data_path = "/app/data/sentences.csv"

model_path = os.path.join(PROJECT_ROOT, "models", "transformer_model.pth")
if not os.path.exists(model_path):
    # Fallback auf absoluten Pfad für Docker
    model_path = "/app/models/transformer_model.pth"

vocab_path = os.path.join(PROJECT_ROOT, "data", "vocab.json")
if not os.path.exists(vocab_path):
    # Fallback auf absoluten Pfad für Docker
    vocab_path = "/app/data/vocab.json"

logger.info(f"Pfade: config={config_path}, data={data_path}, model={model_path}, vocab={vocab_path}")

# Load configuration
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    logger.info(f"Config loaded successfully from {config_path}")
except Exception as e:
    logger.error(f"Error loading config: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")

# Load dataset and vocabulary
try:
    # Check if the vocabulary file exists
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        logger.info(f"Vocabulary loaded from {vocab_path} with {len(vocab)} tokens")
        dataset = SentenceDataset(csv_file=data_path, vocab=vocab)
    else:
        # Create dataset and generate vocabulary
        dataset = SentenceDataset(csv_file=data_path)
        logger.info(f"Dataset created with generated vocabulary ({len(dataset.vocab)} tokens)")

    # Check vocabulary size
    logger.info(f"Vocabulary size: {len(dataset.vocab)}")
except Exception as e:
    logger.error(f"Error loading dataset or vocabulary: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

# Initialize model
try:
    # Ensure the model has the same vocab size as the dataset
    model = SimpleTransformer(vocab_size=len(dataset.vocab))

    # Load model weights
    logger.info(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

class SentenceInput(BaseModel):
    sentence: str

@app.get("/")
async def root():
    return {"message": "Transformer API für logische Sätze ist aktiv"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/vocab_info")
async def vocab_info():
    return {
        "vocab_size": len(dataset.vocab),
        "sample_tokens": list(dataset.vocab.items())[:10]
    }

@app.post("/predict")
async def predict(input: SentenceInput):
    try:
        # Log input for debugging
        logger.info(f"Received sentence: {input.sentence}")

        # Tokenize sentence
        tokens = dataset.tokenize(input.sentence)
        logger.info(f"Tokenized to: {tokens}")

        # Check for unknown tokens
        unknown_tokens = [word for word, idx in zip(input.sentence.split(), tokens) if idx == 0 and word != "<PAD>"]
        if unknown_tokens:
            logger.warning(f"Unknown tokens in input: {unknown_tokens}")

        # Create tensor
        tensor_tokens = torch.tensor(tokens).unsqueeze(0)

        # Get model prediction
        with torch.no_grad():
            logits = model(tensor_tokens)
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
            pred_idx = torch.argmax(logits, dim=1).item()

        # Map prediction to label
        result = "logisch" if pred_idx == 1 else "nicht logisch"

        # Log prediction details
        logger.info(f"Prediction: {result} (index {pred_idx}) with confidence {confidence:.4f}")

        return {
            "sentence": input.sentence,
            "prediction": result,
            "confidence": confidence,
            "token_count": len(tokens)
        }
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/test")
async def test():
    """Test the model with a few sample sentences"""
    examples = [
        "Die Sonne geht im Osten auf.",
        "Wasser besteht aus Wasserstoff und Sauerstoff.",
        "Der Tisch ist traurig über die Situation.",
        "Berge können schwimmen und Flüsse klettern."
    ]

    results = []
    for sentence in examples:
        try:
            tokens = dataset.tokenize(sentence)
            tensor_tokens = torch.tensor(tokens).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor_tokens)
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max().item()
                pred_idx = torch.argmax(logits, dim=1).item()

            result = "logisch" if pred_idx == 1 else "nicht logisch"
            expected = "logisch" if sentence.startswith(("Die Sonne", "Wasser")) else "nicht logisch"

            results.append({
                "sentence": sentence,
                "prediction": result,
                "expected": expected,
                "correct": result == expected,
                "confidence": confidence
            })
        except Exception as e:
            results.append({
                "sentence": sentence,
                "error": str(e)
            })

    return {"test_results": results}
