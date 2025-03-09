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

