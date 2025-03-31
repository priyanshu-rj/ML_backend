import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce TensorFlow memory usage
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU config error: {e}")

# Define class labels (same as before)
CLASS_NAMES = [...]  # Your class names here

# Model Path
MODEL_PATH = "drawing_model.h5"
model = None

def load_model():
    global model
    if model is None:
        try:
            logger.info("Loading model...")
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    return model

# Add favicon route to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def home():
    return "✅ Sketch Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model could not be loaded!"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read and process image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Preprocess image
        image = cv2.resize(image, (28, 28)) / 255.0
        image = image.reshape(1, 28, 28, 1).astype(np.float32)

        # Make prediction
        predictions = model.predict(image, verbose=0)[0]
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        top_3_results = [
            {
                "class_name": CLASS_NAMES[idx],
                "confidence": round(float(predictions[idx] * 100), 2)
            }
            for idx in top_3_indices
        ]

        return jsonify({
            "success": True,
            "top_predictions": top_3_results
        })

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please try again."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
