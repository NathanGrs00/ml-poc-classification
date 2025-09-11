import torch
from config import Config
from model import BertForMultiLabelClassification
from dataset import MultiLabelDataset
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import DataLoader
from metrics import compute_metrics

def evaluate():
    cfg = Config()

    df = pd.read_csv(cfg.csv_path, delimiter=';')
    cfg.label_cols = [col for col in df.columns if col != cfg.text_col]

    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    texts = df[cfg.text_col].tolist()
    labels = df[cfg.label_cols].values

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length)
    dataset = MultiLabelDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size)

    model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
    model.load_state_dict(torch.load(cfg.model_save_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    print(metrics)

if __name__ == "__main__":
    evaluate()