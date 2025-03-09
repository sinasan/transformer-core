import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from dataset import SentenceDataset, collate_fn
from model import SimpleTransformer

# Konfiguration laden
import json
config_path = "../config.json"
with open(config_path, "r") as f:
    config = json.load(f)

batch_size = config["batch_size"]

dataset = SentenceDataset(csv_file='../data/sentences.csv')
data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

model = SimpleTransformer(vocab_size=len(dataset.vocab))
model.load_state_dict(torch.load("../models/transformer_model.pth", map_location=torch.device('cpu')))
model.eval()

# Korrekt initialisieren (außerhalb der Schleife!)
all_preds, all_labels = [], []

with torch.no_grad():
    for sentences, labels in data_loader:
        outputs = model(sentences)
        preds = torch.argmax(outputs, dim=1)

        # extend() verwenden (nicht überschreiben!)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Gesamt-Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['nicht logisch', 'logisch']))

cm = confusion_matrix(all_labels, all_preds)

sns.set(font_scale=1.2)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['nicht logisch', 'logisch'], yticklabels=['nicht logisch', 'logisch'])
plt.ylabel('Tatsächliche Klasse')
plt.xlabel('Vorhergesagte Klasse')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

