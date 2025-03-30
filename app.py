import os
import gc
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

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

# Load the trained model once
MODEL_PATH = "model.h5"  # Ensure this file is available
model = None

def load_model():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

@app.route("/")
def home():
    return "Sketch Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"].read()
        if not file:
            return jsonify({"error": "Empty file"}), 400

        file_stream = np.frombuffer(file, np.uint8)
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

        # Get class name from index
        class_name = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"

        # Cleanup memory
        del image, file_stream
        gc.collect()

        return jsonify({
            "class_index": class_index,
            "class_name": class_name,
            "confidence": confidence
        })

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
