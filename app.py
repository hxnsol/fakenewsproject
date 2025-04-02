from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import os
from waitress import serve
from urllib.parse import urlparse
from datetime import datetime
from pytz import timezone
import re
import string
from newspaper import Article
import validators
from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError("Model file 'model.pkl' not found. Train and save the model first.")

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

LOG_FILE = "logs/analysis_log.csv"

# Write header if log file is new
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "URL", "Source", "Verdict", "Fake Score", "Real Score", "Author", "Publish Date"])

# Log prediction

def log_analysis(data):
    with open(LOG_FILE, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)

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

# History page
@app.route("/history")
def history():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        records = df.to_dict(orient='records')
        return render_template("history.html", records=records)
    else:
        return render_template("history.html", records=[])

# Extract domain/source from URL
def extract_source(news_url):
    parsed_url = urlparse(news_url)
    return parsed_url.netloc

# Fallback metadata extractor

def fallback_extract_metadata(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        author = (
            soup.find('meta', attrs={'name': 'author'}) or
            soup.find('meta', attrs={'name': 'byl'}) or
            soup.find('meta', attrs={'property': 'article:author'})
        )
        publish_date = (
            soup.find('meta', attrs={'property': 'article:published_time'}) or
            soup.find('meta', attrs={'name': 'pubdate'}) or
            soup.find('meta', attrs={'name': 'dc.date'})
        )

        return {
            'author': author['content'] if author and 'content' in author.attrs else "Unavailable",
            'publish_date': publish_date['content'] if publish_date and 'content' in publish_date.attrs else "Unavailable"
        }
    except:
        return {'author': "Unavailable", 'publish_date': "Unavailable"}

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    return text

# Analyze the input text and generate output

def analyze_text(input_text):
    prediction_prob = model.predict_proba([input_text])[0]
    fake_score = prediction_prob[0] * 100
    real_score = prediction_prob[1] * 100
    prediction = 0 if fake_score > real_score else 1

    words = input_text.split()
    highlighted_text = ""

    for word in words:
        if prediction == 0:
            highlighted_text += f' <span class="fake">{word}</span>'
        else:
            highlighted_text += f' <span class="real">{word}</span>'

    ph_tz = timezone('Asia/Manila')
    analysis_timestamp = datetime.now(ph_tz).strftime("%Y-%m-%d %H:%M:%S PHT")

    return highlighted_text.strip(), analysis_timestamp, "Fake News" if prediction == 0 else "Real News", fake_score, real_score

# Handle form submission & API requests

from nltk.tokenize import sent_tokenize

# Analyze text sentence-by-sentence
def analyze_by_sentence(text, model, vectorizer):
    sentences = sent_tokenize(text)
    X = vectorizer.transform(sentences)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    highlighted = ""
    for i, sentence in enumerate(sentences):
        label = preds[i]
        confidence = max(probs[i])
        if confidence < 0.55:
            color_class = "neutral"
        elif label == 0:
            color_class = "fake"
        else:
            color_class = "real"

        highlighted += f'<span class="{color_class}">{sentence.strip()}</span> '

    # Majority vote for overall label
    final_pred = max(set(preds), key=list(preds).count)
    final_label = "Fake News" if final_pred == 0 else "Real News"
    avg_fake = np.mean(probs[:, 0]) * 100
    avg_real = np.mean(probs[:, 1]) * 100
    if abs(avg_fake - avg_real) < 10:
        final_label = "Neutral"

    ph_tz = timezone('Asia/Manila')
    timestamp = datetime.now(ph_tz).strftime("%Y-%m-%d %H:%M:%S PHT")
    return highlighted.strip(), timestamp, final_label, avg_fake, avg_real


@app.route("/predict", methods=["POST"])
def predict():
    news_url = request.form.get("news", "").strip()

    if not validators.url(news_url):
        return render_template("prediction.html", prediction_text="Please enter a valid URL only.")

    try:
        article = Article(news_url)
        article.download()
        article.parse()
        article_text = article.text
        article_authors = ", ".join(article.authors) if article.authors else None
        article_publish_date = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else None
    except:
        return render_template("prediction.html", prediction_text="Failed to extract article from the URL.")

    if not article_text or len(article_text.strip()) < 30:
        return render_template("prediction.html", prediction_text="The extracted article is too short or invalid.")

    if not article_authors or not article_publish_date:
        fallback_meta = fallback_extract_metadata(news_url)
        article_authors = article_authors or fallback_meta['author']
        article_publish_date = article_publish_date or fallback_meta['publish_date']

    clean_text = preprocess_text(article_text)
    news_source = extract_source(news_url)
    highlighted_text, analysis_timestamp, prediction_text, fake_score, real_score = analyze_text(clean_text)

    # Log the analysis to CSV
    log_analysis([
        analysis_timestamp, news_url, news_source, prediction_text,
        f"{fake_score:.2f}", f"{real_score:.2f}",
        article_authors or "Unavailable", article_publish_date or "Unavailable"
    ])

    return render_template("prediction.html",
                       headline=article.title,
                       full_text=article_text,
                       highlighted_text=highlighted_text,
                       analysis_timestamp=analysis_timestamp,
                       news_source=news_source,
                       news_url=news_url,
                       prediction_text=prediction_text,
                       fake_score=fake_score,
                       real_score=real_score,
                       author=article_authors or "Unavailable",
                       publish_date=article_publish_date or "Unavailable",
                       article_image=article.top_image if article.top_image else None)




# Handle image upload and OCR to extract text

    file = request.files['file']
    if file.filename == '':
        return render_template("prediction.html", prediction_text="No file selected.")

    # Save the image temporarily
    image_path = "temp_image.png"
    file.save(image_path)

    try:
        # Extract text from the image
        extracted_text = pytesseract.image_to_string(Image.open(image_path))

        if not extracted_text.strip():
            return render_template("prediction.html", prediction_text="No text detected in the image.")

        clean_text = preprocess_text(extracted_text)
        highlighted_text, analysis_timestamp, prediction_text, fake_score, real_score = analyze_text(clean_text)

        return render_template("prediction.html",
                               highlighted_text=highlighted_text,
                               analysis_timestamp=analysis_timestamp,
                               prediction_text=prediction_text,
                               fake_score=fake_score,
                               real_score=real_score)
    except Exception as e:
        return render_template("prediction.html", prediction_text=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:8080/")
    serve(app, host="0.0.0.0", port=8080)
