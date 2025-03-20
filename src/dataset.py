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
        """Erstellt ein erweitertes Vokabular aus einer Liste von Sätzen"""
        # Verarbeite jeden Satz 
        all_tokens = []
        for sentence in sentences:
            processed = self.preprocess_text(sentence)
            all_tokens.extend(processed.split())

        # Token-Häufigkeiten zählen
        token_counter = {}
        for token in all_tokens:
            if token in token_counter:
                token_counter[token] += 1
            else:
                token_counter[token] = 1

        # Tokens nach Häufigkeit sortieren
        sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
        unique_tokens = [token for token, _ in sorted_tokens]

        # Häufigste Tokens ausgeben (für Debug-Zwecke)
        print(f"Top 10 häufigste Tokens: {unique_tokens[:10]}")

        # Spezielle Tokens hinzufügen
        # Füge alle Satzzeichen als separate Tokens hinzu
        for punct in string.punctuation:
            if punct not in unique_tokens:
                unique_tokens.append(punct)

        # Füge häufig problematische Wörter hinzu, die im Kontext wichtig sein könnten
        problem_words = ["traurig", "arme", "beine", "signalen", "menschen", "haben",
                         "zwei", "sonne", "mond", "tier", "tiere", "schwimmen", "fliegen"]
        for word in problem_words:
            if word not in unique_tokens:
                unique_tokens.append(word)

        # Erstelle das Vokabular (mit <PAD> als 0 und <UNK> als 1)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, token in enumerate(unique_tokens):
            vocab[token] = idx + 2  # +2 wegen <PAD> und <UNK>

        # Vokabulargröße ausgeben
        print(f"Vokabulargröße: {len(vocab)} Tokens")
        return vocab

    def preprocess_text(self, text):
        # Satzzeichen mit Leerzeichen umgeben (alle Satzzeichen aus string.punctuation)
        pattern = r'([' + re.escape(string.punctuation) + r'])'
        text = re.sub(pattern, r' \1 ', text)

        # Mehrfache Leerzeichen entfernen und Text in Kleinbuchstaben umwandeln
        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text

    def tokenize(self, sentence):
        """Tokenisiert einen Satz mit verbesserter Vorverarbeitung und Fallback-Strategien"""
        processed = self.preprocess_text(sentence)
        tokens = processed.split()

        # Verbesserte Token-zu-ID Konvertierung mit mehreren Fallback-Strategien
        token_ids = []
        for token in tokens:
            # Strategie 1: Exakte Übereinstimmung
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            # Strategie 2: Kleinschreibung probieren (wenn nicht bereits gemacht)
            elif token.lower() in self.vocab:
                token_ids.append(self.vocab[token.lower()])
            # Strategie 3: Ohne Satzzeichen probieren (für Fälle wie "Wort.")
            elif token.strip(string.punctuation) in self.vocab:
                token_ids.append(self.vocab[token.strip(string.punctuation)])
            # Strategie 4: Kleinschreibung ohne Satzzeichen
            elif token.lower().strip(string.punctuation) in self.vocab:
                token_ids.append(self.vocab[token.lower().strip(string.punctuation)])
            # Wenn alles fehlschlägt: unbekanntes Token
            else:
                token_ids.append(0)  # <PAD> oder <UNK> Token

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
