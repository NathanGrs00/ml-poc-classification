# Importing necessary libraries
from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer
from multilabel_classifier.models.model import BertForMultiLabelClassification
from multilabel_classifier.config import Config
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load configuration
cfg = Config()

# Load dataset and determine label columns
df = pd.read_csv(cfg.csv_path, delimiter=';')
cfg.label_cols = [col for col in df.columns if col != cfg.text_col]

# Import and initialize singleton for model and tokenizer
from multilabel_classifier.models.model_singleton import ModelSingleton
model_singleton = ModelSingleton(cfg)

# @app.route is a Flask decorator to define routes
# '/' route serves the main page
@app.route('/')
def index():
    # Render the main HTML template.
    return render_template('index.html')

# Flask route for handling predictions via POST request
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    # Extract text from the JSON data
    text = data.get('text', '') if data else ''

    # If somehow no text is provided, return an error
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize and encode the input text, using the singleton pattern
    encoding = model_singleton.tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors='pt'
    )

    # This runs inference without tracking gradients
    # Inference means we are using the model to make predictions on new data
    with torch.no_grad():
        # logits are the raw model outputs
        logits = model_singleton.model(encoding['input_ids'], encoding['attention_mask'])
        # probabilities are obtained by applying sigmoid to logits
        # these probabilities indicate the likelihood of each label being present
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        # Convert probabilities to binary results (0 or 1) based on a threshold of 0.5
        result = (probs > 0.5).astype(int).tolist()

    # Return the results as a JSON response
    return jsonify({
        # The json has the predictions, probabilities, and label names
        'result': result,
        'probs': probs.tolist(),
        'labels': cfg.label_cols
    })

# Run the Flask app
if __name__ == '__main__':
    # debug=True enables auto-reload for code changes
    app.run(debug=True)