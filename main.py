from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from multilabel_classifier.models.model import BertForMultiLabelClassification
from multilabel_classifier.config import Config
import pandas as pd

app = Flask(__name__)

cfg = Config()
df = pd.read_csv(cfg.csv_path, delimiter=';')
cfg.label_cols = [col for col in df.columns if col != cfg.text_col]
from multilabel_classifier.models.model_singleton import ModelSingleton
model_singleton = ModelSingleton(cfg)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '') if data else ''

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    encoding = model_singleton.tokenizer([text], truncation=True, padding=True, max_length=cfg.max_length, return_tensors='pt')
    with torch.no_grad():
        logits = model_singleton.model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        result = (probs > 0.5).astype(int).tolist()

    return jsonify({
        'result': result,
        'probs': probs.tolist(),
        'labels': cfg.label_cols
    })

if __name__ == '__main__':
    app.run(debug=True)