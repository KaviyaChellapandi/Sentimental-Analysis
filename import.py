import warnings
import numpy as np
warnings.filterwarnings("ignore")
     

import os
from sklearn.model_selection import train_test_split
import pandas as pd
     

def load_data(file_path):
    """
    Load text and labels from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'label' columns.")
    return df["text"].tolist(), df["label"].tolist()

def split_data(texts, labels, test_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.
    """
    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

def save_data(file_path, texts, labels):
    """
    Save processed data to a CSV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {"text": [text.strip() for text in texts], "label": labels}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if _name_ == "_main_":
    # Input and output paths
    INPUT_PATH = "/content/train.csv"  # Input raw data path
    OUTPUT_TRAIN_PATH = "/content/processed_train.csv"
    OUTPUT_VAL_PATH = "/content/processed_val.csv"

    # Load and preprocess the data
    texts, labels = load_data(INPUT_PATH)
    train_texts, val_texts, train_labels, val_labels = split_data(texts, labels)

    # Save processed train and validation data
    save_data(OUTPUT_TRAIN_PATH, train_texts, train_labels)
    save_data(OUTPUT_VAL_PATH, val_texts, val_labels)