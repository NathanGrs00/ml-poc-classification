from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from multilabel_classifier.models.model import BertForMultiLabelClassification
from multilabel_classifier.config import Config
import numpy as np
import pandas as pd

app = Flask(__name__)

cfg = Config()
df = pd.read_csv(cfg.csv_path, delimiter=';')
cfg.label_cols = [col for col in df.columns if col != cfg.text_col]
tokenizer = BertTokenizer.from_pretrained(cfg.model_name)
model = BertForMultiLabelClassification(cfg.model_name, num_labels=len(cfg.label_cols))
model.load_state_dict(torch.load('multilabel_classifier/outputs/model.pt', map_location='cpu'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    encoding = tokenizer([text], truncation=True, padding=True, max_length=cfg.max_length, return_tensors='pt')
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        result = (probs > 0.5).astype(int).tolist()
    return jsonify({'result': result, 'probs': probs.tolist()})

if __name__ == '__main__':
    app.run(debug=True)