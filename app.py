import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class labels
CLASS_NAMES = [
    "airplane", "alarm_clock", "ambulance", "apple", "arm", "backpack", "banana", "baseball",
    "baseball_bat", "basket", "basketball", "bat", "bed", "bee", "bicycle", "bird", "book",
    "bottlecap", "bowtie", "brain", "bread", "bridge", "broom", "bus", "butterfly", "cactus",
    "cake", "calculator", "camera", "candle", "car", "carrot", "castle", "cat", "chair",
    "clock", "cloud", "coffee_cup", "computer", "cookie", "couch", "cow", "crab", "crown",
    "cup", "dog", "dolphin", "donut", "door", "dragon", "drums", "duck", "ear", "elephant",
    "envelope", "eye", "eyeglasses", "face", "fan", "feather", "fire_hydrant", "fish",
    "flashlight", "flower", "fork", "frog", "frying_pan", "giraffe", "guitar", "hammer",
    "hand", "hat", "headphones", "helicopter", "horse", "hospital", "hot_air_balloon",
    "house", "ice_cream", "jail", "kangaroo", "key", "ladder", "laptop", "light_bulb",
    "lightning", "lion", "lollipop", "map", "monkey", "moon", "mountain", "mouse", "mug",
    "octopus", "paintbrush", "palm_tree", "panda", "parrot", "pencil", "penguin", "pizza",
    "police_car", "rabbit", "rainbow", "rhinoceros", "roller_coaster", "sandwich", "school_bus",
    "scissors", "shark", "sheep", "skateboard", "snail", "snake", "snowman", "soccer_ball",
    "star", "strawberry", "sun", "swan", "tiger", "train", "tree", "umbrella", "violin",
    "whale", "zebra"
]

# Model Path
MODEL_PATH = "drawing_model.h5"
model = None

def load_model():
    global model
    if model is None:
        try:
            logger.info("Loading model...")
            logger.info(f"Current directory contents: {os.listdir()}")
            
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file '{MODEL_PATH}' not found!")
                return None
                
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

@app.route("/")
def home():
    return "✅ Sketch Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model could not be loaded!"}), 500

    try:
        # Check if file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        logger.info(f"Received file: {file.filename}, {file.mimetype}")

        # Read image file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Debug: Save received image
        debug_img_path = "received_image.png"
        cv2.imwrite(debug_img_path, image)
        logger.info(f"Saved received image to {debug_img_path}")

        # Preprocess image
        image = cv2.resize(image, (28, 28)) / 255.0
        image = image.reshape(1, 28, 28, 1).astype(np.float32)

        # Make prediction
        predictions = model.predict(image)[0]
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
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
