from transformers import BertTokenizer
import torch
from multilabel_classifier.models.model import BertForMultiLabelClassification

class ModelSingleton:
    _instance = None

    def __new__(cls, cfg):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
            cls._instance.model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
            cls._instance.model.load_state_dict(torch.load('multilabel_classifier/outputs/model.pt', map_location='cpu'))
            cls._instance.model.eval()
        return cls._instance