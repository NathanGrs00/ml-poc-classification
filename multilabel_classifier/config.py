from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    csv_path: str = "data/data.csv"
    label_cols: list = None
    text_col: str = "comment"
    output_dir: str = "outputs"
    model_save_path: str = "outputs/model.pt"