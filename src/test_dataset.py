
from dataset import SentenceDataset, collate_fn
from torch.utils.data import DataLoader

dataset = SentenceDataset(csv_file='../data/sentences.csv')
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for sentences, labels in loader:
    print('Sätze (Token-Indizes):', sentences)
    print('Labels:', labels)
    break

