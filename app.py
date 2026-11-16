from pathlib import Path
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
MODEL_PATH = Path("plant_disease_model.keras")
CLASSES_PATH = Path("class_names.json")

# Advice is intentionally general. Confirm diagnosis with a local agriculture expert.
ADVICE = {
    "healthy": {
        "title": "Healthy leaf",
        "effects": "No clear disease symptoms were detected in this photo.",
        "treatment": [
            "Keep watering at the base of the plant when the soil needs it.",
            "Give the plant suitable sunlight and enough space for airflow.",
            "Check the leaves regularly for new spots, insects, or yellowing."
        ],
        "prevention": "Use clean tools, healthy soil, and avoid overwatering."
    },
    "default": {
        "title": "Possible plant disease",
        "effects": "The model detected a possible disease pattern. Symptoms and severity can vary by plant and growing conditions.",
        "treatment": [
            "Remove badly affected leaves using clean tools, if appropriate for the plant.",
            "Avoid wetting the leaves while watering and improve airflow around the plant.",
            "Keep the plant separate from other plants until you have checked whether the disease can spread.",
            "Ask a local agriculture or horticulture expert before using any pesticide or fungicide."
        ],
        "prevention": "Use disease-free seedlings, clean gardening tools, suitable spacing, and watering suited to the plant."
    }
}

def load_model_and_classes():
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return model, classes

MODEL, CLASS_NAMES = load_model_and_classes()

def prepare_image(file):
    image = Image.open(file.stream).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


@app.route("/robots.txt")
def robots():
    return "User-agent: *\\nAllow: /\\nSitemap: " + request.url_root.rstrip("/") + "/sitemap.xml\\n", 200, {"Content-Type": "text/plain"}

@app.route("/sitemap.xml")
def sitemap():
    from xml.sax.saxutils import escape
    home_url = escape(request.url_root.rstrip("/") + "/")
    return f'<?xml version="1.0" encoding="UTF-8"?>\\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{home_url}</loc></url></urlset>', 200, {"Content-Type": "application/xml"}

@app.route("/")
def home():
    return render_template("index.html", model_ready=MODEL is not None)

@app.post("/predict")
def predict():
    if MODEL is None:
        return jsonify({
            "error": "The AI model is not installed yet. Follow the README steps to train the model, then restart the website."
        }), 503

    photo = request.files.get("photo")
    if not photo or not photo.filename:
        return jsonify({"error": "Please take or choose a leaf photo first."}), 400

    try:
        batch = prepare_image(photo)
        # MobileNetV2 preprocessing: convert pixel values from [0,255] to [-1,1].
        batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
        probabilities = MODEL.predict(batch, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index]) * 100
        raw_label = CLASS_NAMES[best_index]
        label = raw_label.replace("___", " — ").replace("_", " ")
        normalized = raw_label.lower()
        is_healthy = "healthy" in normalized
        advice = ADVICE["healthy"] if is_healthy else ADVICE["default"]

        return jsonify({
            "disease": label,
            "confidence": round(confidence, 1),
            "effects": advice["effects"],
            "treatment": advice["treatment"],
            "prevention": advice["prevention"],
            "notice": "AI predictions can be wrong. This is a screening aid, not a confirmed diagnosis. Compare with other symptoms and consult a local agriculture expert before treatment."
        })
    except Exception as exc:
        return jsonify({"error": f"Could not analyse this image. Try a clear, well-lit leaf photo. Details: {str(exc)}"}), 400

if __name__ == "__main__":
    app.run(debug=True)
