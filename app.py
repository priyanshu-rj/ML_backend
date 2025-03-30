from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


MODEL_PATH = "drawing_model.h5"  
model = tf.keras.models.load_model(MODEL_PATH)


with open("mini_classes.txt", "r") as f:
    class_names = [line.strip().replace(" ", "_") for line in f.readlines()]


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "ML Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)

        
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

       
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  
        img = img / 255.0  
        img = np.reshape(img, (1, 28, 28, 1))  

    
        predictions = model.predict(img)[0]
        top_indices = np.argsort(-predictions)[:3]  
        top_classes = [class_names[i] for i in top_indices]

        return jsonify({'predictions': top_classes})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
