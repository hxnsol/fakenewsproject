
from flask import Flask, request, jsonify, render_template
import joblib
import os
from waitress import serve
from urllib.parse import urlparse
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError("Model file 'model.pkl' not found. Train and save the model first.")

def extract_source(news_url):
    parsed_url = urlparse(news_url)
    return parsed_url.netloc

def analyze_text(input_text):
    # Predict using the model
    prediction = model.predict([input_text])[0]
    
    # Simulated segmentation (you can replace with NLP-based segmentation)
    words = input_text.split()
    highlighted_text = ""
    
    for word in words:
        if prediction == 0:  # Fake News
            highlighted_text += f' <span class="fake">{word}</span>'
        else:  # Real News
            highlighted_text += f' <span class="real">{word}</span>'

    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return highlighted_text.strip(), analysis_timestamp, "Fake News" if prediction == 0 else "Real News"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        news_url = request.form.get("news_url")
        news_content = request.form.get("news_content")
        
        news_source = extract_source(news_url) if news_url else "Unknown Source"
        
        highlighted_text, analysis_timestamp, prediction_text = analyze_text(news_content)

        return render_template("prediction.html", 
                               highlighted_text=highlighted_text, 
                               analysis_timestamp=analysis_timestamp,
                               news_source=news_source,
                               news_url=news_url,
                               prediction_text=prediction_text)

    return render_template("prediction.html", highlighted_text=None)

if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:8080/")
    serve(app, host="0.0.0.0", port=8080)


from urllib.parse import urlparse
from datetime import datetime

def extract_source(news_url):
    parsed_url = urlparse(news_url)
    return parsed_url.netloc

def analyze_text(input_text):
    prediction = model.predict([input_text])[0]
    
    words = input_text.split()
    highlighted_text = ""
    
    for word in words:
        if prediction == 0:  # Fake News
            highlighted_text += f' <span class="fake">{word}</span>'
        else:  # Real News
            highlighted_text += f' <span class="real">{word}</span>'

    analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return highlighted_text.strip(), analysis_timestamp, "Fake News" if prediction == 0 else "Real News"

@app.route("/predict", methods=["POST"])
def predict():
    news_url = request.form.get("news_url", "")
    news_content = request.form.get("news", "")

    if not news_content:
        return jsonify({"error": "No text provided"}), 400

    news_source = extract_source(news_url) if news_url else "Unknown Source"
    highlighted_text, analysis_timestamp, prediction_text = analyze_text(news_content)

    return render_template("prediction.html", 
                           highlighted_text=highlighted_text, 
                           analysis_timestamp=analysis_timestamp,
                           news_source=news_source,
                           news_url=news_url,
                           prediction_text=prediction_text)

