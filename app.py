from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)


model = tf.keras.models.load_model("drawing_model.h5")

@app.route("/")
def home():
    return "ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"].read()
        image = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (28, 28)) / 255.0
        image = image.reshape(1, 28, 28, 1)

        predictions = model.predict(image)
        class_index = np.argmax(predictions)

        return jsonify({"predictions": [int(class_index)]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
