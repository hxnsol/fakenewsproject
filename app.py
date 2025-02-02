from flask import Flask, request, jsonify, render_template
import joblib
import os
from waitress import serve
from flask import Flask, send_from_directory

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "C:/Users/Nico/Desktop/thesisproj/model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError("Model file 'model.pkl' not found. Train and save the model first.")

# Serve images from a custom folder
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory("C:/Users/Nico/Desktop/thesisproj/Model deployment using Flask/images", filename)

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
@app.route("/predict", methods=["POST"])
def predict():
    if request.content_type == "application/json":
        data = request.json
        news_text = data.get("news", "")
    else:
        news_text = request.form.get("news", "")

    if not news_text:
        return jsonify({"error": "No text provided"}), 400

    # Predict using the model
    prediction = model.predict([news_text])[0]  # Use .predict() method
    result = "Real News" if prediction == 1 else "Fake News"

    # Return result for API or HTML form
    if request.content_type == "application/json":
        return jsonify({"prediction": result})
    else:
        return render_template("prediction.html", prediction_text=f"{result}")

if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:8080/")
    serve(app, host="0.0.0.0", port=8080)

