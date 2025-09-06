import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class FactCheckDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Map nhãn -> id
        self.label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = f"Context: {row['context']} Prompt: {row['prompt']} Response: {row['response']}"
        label = self.label2id[row["label"]]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(train_path, test_path, tokenizer, batch_size=16, max_len=256):
    train_dataset = FactCheckDataset(train_path, tokenizer, max_len=max_len)
    test_dataset = FactCheckDataset(test_path, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.label2id