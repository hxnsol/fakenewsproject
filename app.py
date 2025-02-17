from flask import Flask, request, jsonify, render_template
import joblib
import os
from waitress import serve
from flask import Flask, send_from_directory

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError("Model file 'model.pkl' not found. Train and save the model first.")

# Serve images from a custom folder
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory("images", filename)

# Home page
@app.route("/")
def home():
    return render_template("prediction.html", prediction_text="")

# Metrics page
@app.route("/metrics")
def metrics():
    return render_template("metrics.html")

# FAQs page
@app.route("/faqs")
def faqs():
    return render_template("faqs.html")

# Contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Handle form submission & API requests

from urllib.parse import urlparse
from datetime import datetime
import numpy as np

def extract_source(news_url):
    parsed_url = urlparse(news_url)
    return parsed_url.netloc

def analyze_text(input_text):
    # Get the prediction probability from the model (assuming it's a probabilistic model)
    prediction_prob = model.predict_proba([input_text])[0]  # Get probabilities

    fake_score = prediction_prob[0] * 100  # Fake probability
    real_score = prediction_prob[1] * 100  # Real probability
    prediction = 0 if fake_score > real_score else 1  # Final prediction

    words = input_text.split()
    highlighted_text = ""

    for word in words:
        if prediction == 0:  # Fake News
            highlighted_text += f' <span class="fake">{word}</span>'
        else:  # Real News
            highlighted_text += f' <span class="real">{word}</span>'

    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return highlighted_text.strip(), analysis_timestamp, "Fake News" if prediction == 0 else "Real News", fake_score, real_score

@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type == "application/json":
        data = request.json
        news_text = data.get("news", "")
        news_url = data.get("news_url", "")
    else:
        news_text = request.form.get("news", "")
        news_url = request.form.get("news_url", "")

    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    news_source = extract_source(news_url) if news_url else "Unknown Source"
    highlighted_text, analysis_timestamp, prediction_text, fake_score, real_score = analyze_text(news_text)

    if request.content_type == "application/json":
        return jsonify({"prediction": prediction_text, "highlighted_text": highlighted_text, "timestamp": analysis_timestamp, "fake_score": fake_score, "real_score": real_score})
    else:
        return render_template("prediction.html", 
                               highlighted_text=highlighted_text, 
                               analysis_timestamp=analysis_timestamp,
                               news_source=news_source,
                               news_url=news_url,
                               prediction_text=prediction_text,
                               fake_score=fake_score,
                               real_score=real_score)

