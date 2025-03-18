import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer
import json
import os


config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)


# Hyperparameter
embedding_dim = config["embedding_dim"]
num_heads = config["num_heads"]
num_layers = config["num_layers"]
num_classes = config["num_classes"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]

# Datensatz laden
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(PROJECT_ROOT, "data", "sentences.csv")
dataset = SentenceDataset(csv_file=data_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Modell initialisieren
model = SimpleTransformer(vocab_size=len(dataset.vocab))

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

torch.save(model.state_dict(), "../models/transformer_model.pth")
print("Training abgeschlossen.")
print("Model gespeichert in ../models/transformer_model.pth")

