import pandas as pd
from transformers import BertTokenizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["text"].tolist(), df["label"].tolist()

def tokenize_texts(texts, tokenizer, max_len=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )