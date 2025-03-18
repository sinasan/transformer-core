import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_file = os.path.join(PROJECT_ROOT, "data", "sentences.csv")
vocab_file = os.path.join(PROJECT_ROOT, "data", "vocab.json")

class SentenceDataset(Dataset):
    def __init__(self, csv_file=csv_file, vocab=None, vocab_file=vocab_file):
        self.data = pd.read_csv(csv_file)
        self.sentences = self.data['sentence'].tolist()
        self.labels = self.data['label'].tolist()

        if vocab is None:
            self.vocab = self.build_vocab(self.sentences)
            # Vokabular speichern
            with open(vocab_file, 'w') as f:
                json.dump(self.vocab, f)
        else:
            self.vocab = vocab

    def build_vocab(self, sentences):
        unique_tokens = set(token for s in sentences for token in s.split())
        vocab = {word: idx+1 for idx, word in enumerate(sorted(unique_tokens))}
        vocab["<PAD>"] = 0
        return vocab

    def tokenize(self, sentence):
        return [self.vocab.get(token, 0) for token in sentence.split()]

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

