from dataclasses import dataclass

# this is a config file to store all the hyperparameters and file paths
@dataclass
class Config:
    # We are using a pre-trained BERT model called "bert-base-uncased"
    model_name: str = "bert-base-uncased"
    # Maximum length of input text after tokenization
    max_length: int = 128
    # Batch size means how many samples we process before updating the model
    train_batch_size: int = 16
    # Evaluation batch size is the number of samples to process at once during evaluation
    eval_batch_size: int = 32
    # Learning rate controls how much to change the model in response to the estimated error
    #2e-5 is a common choice for fine-tuning BERT
    learning_rate: float = 2e-5
    # num_epochs is how many times we go through the entire training dataset
    # 1 epoch is 1 full pass through the training data
    num_epochs: int = 3
    # csv_path is the path to the CSV file containing our dataset
    csv_path: str = "multilabel_classifier/data/data.csv"
    # label_cols is a list of column names in the CSV that contain the labels
    label_cols: list = None
    # text_col is the name of the column that contains the text data
    text_col: str = "comment"
    # output_dir is where we save the model and other outputs
    output_dir: str = "outputs"
    # model_save_path is the specific path to save the trained model
    model_save_path: str = "outputs/model.pt"