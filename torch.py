import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class SarcasmDataset(torch.utils.data.Dataset):
    def _init_(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }