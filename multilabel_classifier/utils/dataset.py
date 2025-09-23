from torch.utils.data import Dataset
import torch
# This is a custom dataset class for multi-label classification tasks.
class MultiLabelDataset(Dataset):
    # Initialize the dataset with encodings and labels
    def __init__(self, encodings, labels):
        # Store the encodings and labels
        # Encodings are the tokenized inputs
        self.encodings = encodings
        self.labels = labels

    # Return the length of the dataset
    def __len__(self):
        return len(self.labels)

    # Get an item by index
    # This is needed by PyTorch's DataLoader
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item