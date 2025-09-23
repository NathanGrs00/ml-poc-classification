from sklearn.metrics import f1_score, accuracy_score

# Function to compute F1 score and accuracy for multi-label classification
# This is used by the evaluation script
def compute_metrics(preds, labels, threshold=0.5):
    # predictions are probabilities, convert them to binary using the threshold
    preds = (preds > threshold).astype(int)
    # Calculate micro F1 score and accuracy
    f1 = f1_score(labels, preds, average='micro')
    # accuracy is not typically used for multi-label, but included here for completeness
    acc = accuracy_score(labels, preds)
    # Return the metrics as a dictionary
    return {'f1': f1, 'accuracy': acc}