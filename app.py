from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("drawing_model.h5")


with open("mini_classes.txt", "r") as f:
    class_names = [c.strip().replace(" ", "_") for c in f.readlines()]

@app.route("/", methods=["GET"])
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file).convert("L").resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0 
    predictions = model.predict(img_array)[0]
    
    top_indices = (-predictions).argsort()[:3]
    results = [class_names[i] for i in top_indices]

    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000) 
