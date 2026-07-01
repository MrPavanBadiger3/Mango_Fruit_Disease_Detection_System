"""
Mango Disease Detection - Flask Web Application
Run: python app.py
Then open: http://localhost:5000
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max upload

# ─────────────────────────────────────────────
# Load model & class names once at startup
# ─────────────────────────────────────────────
MODEL_PATH         = "mango_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE           = (224, 224)

print("⏳ Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

with open(CLASS_INDICES_PATH) as f:
    CLASS_NAMES = json.load(f)

# Disease info shown to farmers
DISEASE_INFO = {
    "Anthracnose": {
        "description": "A fungal disease causing dark, sunken lesions on mango fruits and leaves.",
        "severity": "High",
        "remedy": "Apply copper-based fungicides. Remove and destroy infected plant parts. Ensure good air circulation.",
        "emoji": "🍂"
    },
    "Alternaria": {
        "description": "A fungal disease causing dark brown to black spots on mango fruits, especially after harvest.",
        "severity": "Medium",
        "remedy": "Apply mancozeb or copper oxychloride fungicide. Store fruits in cool dry conditions. Avoid injuries to fruits during handling.",
        "emoji": "🟤"
    },
    "Black Mould Rot": {
        "description": "Caused by Aspergillus niger fungus, appears as black powdery growth on mango surface.",
        "severity": "High",
        "remedy": "Handle fruits carefully to avoid wounds. Apply hot water treatment at 52°C for 5 minutes. Store in cool ventilated areas.",
        "emoji": "⬛"
    },
    "Stem end Rot": {
        "description": "Fungal disease starting at the stem end of the fruit, causing brown rot that spreads inward.",
        "severity": "High",
        "remedy": "Apply prochloraz or thiabendazole post-harvest. Harvest fruits with long stalks. Avoid water stress on trees.",
        "emoji": "🌿"
    },
    "Healthy": {
        "description": "Your mango fruit appears healthy with no signs of disease!",
        "severity": "None",
        "remedy": "Continue regular watering, fertilization, and monitor periodically.",
        "emoji": "✅"
    }
}

# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────
def predict_disease(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]

    results = []
    for idx in top3_idx:
        label = CLASS_NAMES[str(idx)]
        results.append({
            "disease": label,
            "confidence": round(float(preds[idx] * 100), 2),  # 2 decimal places
            "info": DISEASE_INFO.get(label, {})
        })
    return results

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    allowed_ext = {"jpg", "jpeg", "png", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_ext:
        return jsonify({"error": "Only JPG/PNG images allowed"}), 400

    try:
        img_bytes = file.read()
        results   = predict_disease(img_bytes)
        return jsonify({"success": True, "predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classes")
def get_classes():
    return jsonify({"classes": list(CLASS_NAMES.values())})

if __name__ == "__main__":
    print("\n🌿 Mango Disease Detection App running at http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)