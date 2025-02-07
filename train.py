def train_model(train_data, val_data, model, tokenizer, epochs=3, batch_size=16):
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: val.to("cuda") for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to("cuda")
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

    torch.save(model.state_dict(), "/content/sarcasm_model.pt")

def main():
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split

    # Load data
    texts, labels = load_data("/content/train.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2
    )

    train_data = SarcasmDataset(train_texts, train_labels, tokenizer)
    val_data = SarcasmDataset(val_texts, val_labels, tokenizer)

    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to("cuda")

    # Train
    train_model(train_data, val_data, model, tokenizer)

if _name_ == "_main_":
    main()
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_state_dict(torch.load("../content/sarcasm_model.pt"))
    model.eval()
    return model

def predict(texts, model, tokenizer):
    # Ensure texts is a list, even for single input
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize the input texts
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    # Move the tensors to the same device as the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: val.to(device) for key, val in encodings.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Map predictions to human-readable labels
    label_mapping = {0: "not sarcasm", 1: "sarcasm"}  # Adjust based on your dataset
    labeled_predictions = [label_mapping[pred.item()] for pred in predictions]

    return labeled_predictions