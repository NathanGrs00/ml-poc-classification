import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from config import Config
from utils.dataset import MultiLabelDataset
from models.model import BertForMultiLabelClassification
from utils.metrics import compute_metrics
import os

def train():
    cfg = Config()
    df = pd.read_csv(cfg.csv_path, delimiter=';')
    cfg.label_cols = [col for col in df.columns if col != cfg.text_col]

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[cfg.text_col].tolist(),
        df[cfg.label_cols].values.tolist(),
        test_size=0.2,
        random_state=42
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=cfg.max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=cfg.max_length)

    train_dataset = MultiLabelDataset(train_encodings, train_labels)
    val_dataset = MultiLabelDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size)

    model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(cfg.num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(model.state_dict(), cfg.model_save_path)

if __name__ == "__main__":
    train()