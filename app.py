from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load model at startup with error handling
MODEL_PATH = "drawing_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    model = None

@app.route("/")
def home():
    return "ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"].read()
        if not file:
            return jsonify({"error": "Empty file"}), 400

        file_stream = np.asarray(bytearray(file), dtype=np.uint8)
        image = cv2.imdecode(file_stream, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Preprocess image
        image = cv2.resize(image, (28, 28)) / 255.0
        image = image.reshape(1, 28, 28, 1).astype(np.float32)

        # Make prediction
        predictions = model.predict(image)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({"predictions": class_index, "confidence": confidence})
    
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
