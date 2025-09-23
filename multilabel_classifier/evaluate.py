import torch
from config import Config
from models.model import BertForMultiLabelClassification
from utils.dataset import MultiLabelDataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics

# This script is used to evaluate the model using the test dataset
def evaluate():
    # Initialize configuration file
    cfg = Config()

    # df stands for dataframe, this reads the data from the csv file
    df = pd.read_csv('data/data.csv', delimiter=';')
    # Decide which columns are labels
    cfg.label_cols = [col for col in df.columns if col != cfg.text_col]

    # Tokenizer to convert text to tokens, this is needed for BERT
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    # Prepare the test dataset
    texts = df[cfg.text_col].tolist()
    labels = df[cfg.label_cols].values

    # encodings is the tokenized version of the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=cfg.max_length)
    # dataset is used to create batches for evaluation
    dataset = MultiLabelDataset(encodings, labels)
    # loader is used to load the data in batches
    loader = DataLoader(dataset, batch_size=cfg.eval_batch_size)

    # Load the trained model
    model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
    model.load_state_dict(torch.load(cfg.model_save_path))
    # Set the model to evaluation mode
    model.eval()

    # Check if GPU is available, if yes use it, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the device (GPU or CPU)
    model.to(device)

    # Store all predictions and labels
    all_preds = []
    all_labels = []

    # no_grad() is used to disable gradient calculation, this saves memory and computations
    with torch.no_grad():
        # Loop over each batch in the evaluation loader
        for batch in loader:
            # ids are the token ids
            input_ids = batch['input_ids'].to(device)
            # mask is used to ignore padding tokens
            attention_mask = batch['attention_mask'].to(device)
            # labels are the true labels for the batch
            labels = batch['labels'].cpu().numpy()
            # Get the model predictions
            logits = model(input_ids, attention_mask)
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits).cpu().numpy()

            # extend the lists with current batch predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Now convert predictions to binary (0 or 1) using a threshold of 0.5
    all_preds = (np.array(all_preds) > 0.5).astype(int)
    # Also convert labels to int
    all_labels = np.array(all_labels).astype(int)
    # Compute and print evaluation metrics
    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    # Finally, print the metrics
    print(metrics)

# Run the evaluation function if this script is executed
if __name__ == "__main__":
    evaluate()