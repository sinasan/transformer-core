import pandas as pd
import torch
from torch.utils.data import Dataset

class SentenceDataset(Dataset):
    def __init__(self, csv_file, vocab=None):
        self.data = pd.read_csv(csv_file)
        self.sentences = self.data['sentence'].tolist()
        self.labels = self.data['label'].tolist()
        self.vocab = vocab or self.build_vocab(self.sentences)

    def build_vocab(self, sentences):
        words = set()
        for sent in sentences:
            words.update(sent.lower().split())
        word2idx = {word: idx+2 for idx, word in enumerate(sorted(words))}
        word2idx['<PAD>'] = 0
        word2idx['<UNK>'] = 1
        return word2idx

    def tokenize(self, sentence):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in sentence.lower().split()]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = self.tokenize(sentence)
        label = self.labels[idx]
        return torch.tensor(tokens), torch.tensor(label)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    padded_sentences = torch.zeros(len(sentences), max_len).long()

    for i, sentence in enumerate(sentences):
        padded_sentences[i, :lengths[i]] = sentence

    labels = torch.tensor(labels).long()
    return padded_sentences, labels

