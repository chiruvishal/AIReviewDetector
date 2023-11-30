from flask import Flask, render_template, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

# Load the fine-tuned model
model = RobertaForSequenceClassification.from_pretrained("fine_tuned_roberta")

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Function to predict
def predict_review(text):
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    with torch.no_grad():
        output = model(input_ids)
    predicted_label = int(np.argmax(output.logits))
    return "ChatGPT" if predicted_label == 0 else "Human"

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_review(text)
        return render_template('results.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
