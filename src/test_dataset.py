
from dataset import SentenceDataset, collate_fn
from torch.utils.data import DataLoader
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

dataset = SentenceDataset(csv_file='../data/sentences.csv')
loader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn)

for sentences, labels in loader:
    print('Sätze (Token-Indizes):', sentences)
    print('Labels:', labels)
    break

