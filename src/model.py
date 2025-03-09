import torch
import torch.nn as nn
import json
import os

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleTransformer, self).__init__()

        embedding_dim = config["embedding_dim"]
        num_heads = config["num_heads"]
        num_classes = config["num_classes"]
        num_layers = config.get("num_layers", 1)

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
