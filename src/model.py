import torch
import torch.nn as nn
import json
import os
import math

config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r") as f:
    config = json.load(f)


class EnhancedAttention(nn.Module):
    """
    Ein vereinfachter Attention-Mechanismus mit robusterer Implementierung.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_relative_pos=False):
        super(EnhancedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim muss durch num_heads teilbar sein"

        # Multi-head attention für robusteres Training
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Dropout nach der Attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Umwandlung der Maske in ein Format, das nn.MultiheadAttention versteht
        attn_mask = None
        if mask is not None:
            attn_mask = mask.squeeze(1).squeeze(1)  # Anpassung je nach Maskenformat

        # Verwendung des eingebauten MultiheadAttention-Moduls
        output, _ = self.mha(x, x, x, key_padding_mask=attn_mask)
        output = self.dropout(output)

        return output


class EnhancedTransformerEncoderLayer(nn.Module):
    """
    Verbesserter Transformer-Encoder-Layer mit vereinfachtem Attention-Mechanismus.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super(EnhancedTransformerEncoderLayer, self).__init__()

        # Vereinfachte Attention
        self.self_attn = EnhancedAttention(d_model, nhead, dropout=dropout)

        # Feedforward-Netzwerk
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Aktivierungsfunktion
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Normalisierung und Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-Attention Block mit Pre-Normalization
        src2 = self.norm1(src)
        src_attn = self.self_attn(src2, mask=src_mask)
        src = src + self.dropout1(src_attn)

        # Feedforward-Block mit Pre-Normalization
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleTransformer, self).__init__()

        embedding_dim = config["embedding_dim"]
        num_heads = config["num_heads"]
        num_classes = config["num_classes"]
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.1)

        # Word-Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Transformer Encoder-Schichten
        self.layers = nn.ModuleList([
            EnhancedTransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Klassifikationsschicht
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

        # Initialisierung
        self._reset_parameters()

    def _reset_parameters(self):
        """Parameter-Initialisierung"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # Maske für Padding-Tokens
        padding_mask = (x == 0)

        # Embedding und Positional Encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Transformer-Schichten mit korrekter Maskenbehandlung
        for layer in self.layers:
            x = layer(x, src_mask=padding_mask)

        # Pooling: Ignoriere Padding-Tokens beim Mitteln
        mask = padding_mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(mask, 0.0)

        # Summe geteilt durch Anzahl der nicht-Padding-Tokens
        num_tokens = (~padding_mask).sum(dim=1, keepdim=True).float()
        x = x.sum(dim=1) / num_tokens.clamp(min=1.0)

        # Klassifikation
        x = self.dropout(x)
        x = self.fc(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Sinusförmiges Positional Encoding
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
