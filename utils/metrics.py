from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(preds, labels, threshold=0.5):
    preds = (preds > threshold).astype(int)
    f1 = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {'f1': f1, 'accuracy': acc}