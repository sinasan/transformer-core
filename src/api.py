from fastapi import FastAPI
from pydantic import BaseModel
import torch
import json
import os
from src.dataset import SentenceDataset
from src.model import SimpleTransformer

app = FastAPI(title="Transformer API für logische Sätze")

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

dataset = SentenceDataset(csv_file='./data/sentences.csv')

# Modell definieren (Parameter aus config.json)
model = SimpleTransformer(vocab_size=len(dataset.vocab))

# Modell laden
model.load_state_dict(torch.load("./models/transformer_model.pth", map_location=torch.device('cpu')))
model.eval()

class SentenceInput(BaseModel):
    sentence: str

@app.post("/predict")
async def predict(input: SentenceInput):
    tokens = dataset.tokenize(input.sentence)
    tensor_tokens = torch.tensor(tokens).unsqueeze(0)
    logits = model(tensor_tokens)
    pred = torch.argmax(logits, dim=1).item()

    result = "logisch" if pred == 1 else "nicht logisch"
    return {"sentence": input.sentence, "prediction": result}

