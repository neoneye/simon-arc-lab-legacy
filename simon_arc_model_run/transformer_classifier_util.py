import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        src = torch.tensor(self.xs[idx], dtype=torch.long)
        ys = torch.tensor(self.ys[idx], dtype=torch.long)
        return src, ys
