from transformers import BertTokenizer
import torch
from multilabel_classifier.models.model import BertForMultiLabelClassification
# This is a Singleton class to ensure that only one instance of the model is created.
# We use this because initializing the model and tokenizer can be resource-intensive.
class ModelSingleton:
    # _instance is a class variable to hold the single instance, it is initially None
    _instance = None

    # This is a function to create a new instance if one doesn't exist yet.
    # It takes the class itself (cls) and a configuration object (cfg) as parameters.
    def __new__(cls, cfg):
        # If no instance exists,
        if cls._instance is None:
            # Create a new instance of the class
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            # Tokenizer is needed to make smaller pieces of text (tokens) that the model can understand
            # Tokens also include special characters.
            cls._instance.tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
            # Load the pre-trained model for multi-label classification,
            # it takes the model name and the number of labels as parameters.
            cls._instance.model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
            # Load the model's learned parameters from the model file
            cls._instance.model.load_state_dict(torch.load('multilabel_classifier/outputs/model.pt', map_location='cpu'))
            # Set the model to evaluation mode, which is necessary for inference
            cls._instance.model.eval()
        # Return the single instance of the class
        return cls._instance