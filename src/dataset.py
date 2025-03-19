import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
import re
import string  # NEU: für komplettes string.punctuation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_file = os.path.join(PROJECT_ROOT, "data", "sentences.csv")
vocab_file = os.path.join(PROJECT_ROOT, "data", "vocab.json")

class SentenceDataset(Dataset):
    def __init__(self, csv_file=csv_file, vocab=None, vocab_file=vocab_file, force_rebuild_vocab=False):
        self.data = pd.read_csv(csv_file)
        self.sentences = self.data['sentence'].tolist()
        self.labels = self.data['label'].tolist()

        # Wenn force_rebuild_vocab=True ist oder kein Vokabular übergeben wird und vocab_file nicht existiert,
        # dann erstellen wir ein neues Vokabular
        if force_rebuild_vocab or (vocab is None and (not os.path.exists(vocab_file))):
            print(f"Erstelle neues Vokabular basierend auf {len(self.sentences)} Sätzen...")
            processed_sentences = [self.preprocess_text(s) for s in self.sentences]
            self.vocab = self.build_vocab(processed_sentences)
            # Vokabular speichern
            with open(vocab_file, 'w') as f:
                json.dump(self.vocab, f)
            print(f"Neues Vokabular mit {len(self.vocab)} Tokens in {vocab_file} gespeichert.")
        elif vocab is not None:
            self.vocab = vocab
            print(f"Verwende übergebenes Vokabular mit {len(self.vocab)} Tokens.")
        else:
            # Lade existierendes Vokabular aus Datei
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            print(f"Vokabular aus {vocab_file} geladen mit {len(self.vocab)} Tokens.")

    def build_vocab(self, sentences):
        """Erstellt ein Vokabular aus einer Liste von Sätzen"""
        # Verarbeiten Sie jeden Satz mit der improved_tokenize Methode
        all_tokens = []
        for sentence in sentences:
            processed = self.preprocess_text(sentence)
            all_tokens.extend(processed.split())

        # Erstellen Sie ein Set von einzigartigen Tokens
        unique_tokens = set(all_tokens)

        # NEU: Füge alle Satzzeichen als separate Tokens hinzu
        for punct in string.punctuation:
            unique_tokens.add(punct)

        # NEU: Füge häufig problematische Wörter hinzu
        problem_words = ["traurig", "Arme", "Beine", "Signalen", "Menschen", "haben"]
        for word in problem_words:
            unique_tokens.add(word)

        # Erstellen Sie das Vokabular
        vocab = {word: idx+1 for idx, word in enumerate(sorted(unique_tokens))}
        vocab["<PAD>"] = 0

        return vocab

    def preprocess_text(self, text):
        """Verbesserte Vorverarbeitung von Text für Tokenisierung"""
        # NEU: Punktuation mit Leerzeichen umgeben (alle Satzzeichen)
        pattern = r'([' + re.escape(string.punctuation) + r'])'
        text = re.sub(pattern, r' \1 ', text)

        # Mehrfache Leerzeichen entfernen
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, sentence):
        """Tokenisiert einen Satz mit verbesserter Vorverarbeitung"""
        processed = self.preprocess_text(sentence)
        tokens = processed.split()

        # NEU: Verbesserte Token-zu-ID Konvertierung mit Fallback
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            elif token.lower() in self.vocab:  # Versuche Kleinschreibung
                token_ids.append(self.vocab[token.lower()])
            else:
                token_ids.append(0)  # Unbekanntes Token

        return token_ids

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        tokenized = self.tokenize(sentence)
        return torch.tensor(tokenized), torch.tensor(label)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sentences, labels
