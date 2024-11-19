import torch
from torch.utils.data import Dataset
import json
import os

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

    @classmethod
    def load_jsonl_files(cls, path_to_dir: str) -> 'MyDataset':
        """
        Create a MyDataset object from a directory containing JSONL files.
        """
        jsonl_file_paths = []
        for subdir, dirs, files in os.walk(path_to_dir):
            for file in sorted(files):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(subdir, file)
                    jsonl_file_paths.append(file_path)
        print('MyDataset.load_jsonl_files() jsonl_file_paths:', len(jsonl_file_paths))

        all_xs = []
        all_ys = []
        for input_file in jsonl_file_paths:
            with open(input_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    xs = data['xs']
                    ys = data['ys']
                    all_xs.append(xs)
                    all_ys.append(ys)

        print('MyDataset.load_jsonl_files() all_xs:', len(all_xs))
        return cls(all_xs, all_ys)
