from transformers import BertModel
import torch.nn as nn

# Define a BERT-based model for multi-label classification, module is taken as a parameter
class BertForMultiLabelClassification(nn.Module):
    # Constructor takes model_name and num_labels as parameters
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        # Classification layer to output logits for each label
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    # Forward method defines how the input data flows through the model
    def forward(self, input_ids, attention_mask, labels=None):
        # Get outputs from BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pass the outputs to the classifier to get logits, logits are raw predictions
        logits = self.classifier(outputs.pooler_output)
        # If labels are provided, compute the loss using BCEWithLogitsLoss
        if labels is not None:
            # loss is needed during training to measure how well the model is performing
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits