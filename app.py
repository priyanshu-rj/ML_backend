from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import logging

app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load model at startup
model = tf.keras.models.load_model("drawing_model.h5")

@app.route("/")
def home():
    return "ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
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

        image = cv2.resize(image, (28, 28)) / 255.0
        image = image.reshape(1, 28, 28, 1)

        predictions = model.predict(image)
        class_index = int(np.argmax(predictions))

        return jsonify({"predictions": class_index})

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
