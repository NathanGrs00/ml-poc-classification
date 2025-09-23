import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from config import Config
from utils.dataset import MultiLabelDataset
from models.model import BertForMultiLabelClassification
import os

# Training function, this makes the model using the training dataset
def train():
    # Initialize configuration file
    cfg = Config()
    # df stands for dataframe, this reads the data from the csv file
    df = pd.read_csv('data/data.csv', delimiter=';')
    # Decide which columns are labels
    cfg.label_cols = [col for col in df.columns if col != cfg.text_col]

    # Tokenizer to convert text to tokens, this is needed for BERT
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
    # Split the data into training set
    train_texts, train_labels = df[cfg.text_col].tolist(), df[cfg.label_cols].values.tolist()

    # encodings is the tokenized version of the texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=cfg.max_length)
    # dataset is used to create batches for training
    train_dataset = MultiLabelDataset(train_encodings, train_labels)
    # loader is used to load the data in batches
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True)

    # model calls the BERT model for multi-label classification
    model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
    # Set the model to training mode
    model.train()

    # AdamW is the optimizer, it helps to update the model weights
    # it needs the model parameters and learning rate.
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    # Check if GPU is available, if yes use it, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the device (GPU or CPU)
    model.to(device)

    # Loop over the number of epochs
    for epoch in range(cfg.num_epochs):
        # Every epoch, reset the total loss
        total_loss = 0
        # Loop over each batch in the training loader
        for batch in train_loader:
            # Reset the gradients
            optimizer.zero_grad()
            # Move the batch data to the device
            input_ids = batch['input_ids'].to(device)
            # mask is used to ignore padding tokens
            attention_mask = batch['attention_mask'].to(device)
            # labels are the labels for the batch.
            labels = batch['labels'].to(device)
            # loss is the output of the model, it calculates how far the predictions are from the actual labels
            loss, _ = model(input_ids, attention_mask, labels)
            # Backpropagation, this updates the model weights
            loss.backward()
            # optimizer step, this applies the weight updates
            optimizer.step()
            # Add the loss of this batch to the total loss
            total_loss += loss.item()
        # Print the average loss for this epoch
        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Save the trained model to the specified path
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Save the model state dictionary
    torch.save(model.state_dict(), cfg.model_save_path)

# Run the training function if this script is executed
if __name__ == "__main__":
    train()