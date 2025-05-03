# Student ID : W1947458
# Student Name : Abhinava Sai Bugudi
# Supervisor : Dr. Dimitris Dracopoulos
# Module : 6COSC023W.Y Computer Science Final Project

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Character map (EMNIST - 62 classes: 0-9, A-Z, a-z)
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Preprocessing function
def preprocess_js_style(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = gray[y:y+h, x:x+w]

    target_dim = 28
    edge_size = 2
    resize_dim = target_dim - edge_size * 2

    h, w = roi.shape
    pad_vertically = w > h
    pad_size = (max(h, w) - min(h, w)) // 2

    if pad_vertically:
        padded = np.pad(roi, ((pad_size, pad_size), (0, 0)), constant_values=255)
    else:
        padded = np.pad(roi, ((0, 0), (pad_size, pad_size)), constant_values=255)

    resized = cv2.resize(padded, (resize_dim, resize_dim))
    final = np.pad(resized, ((edge_size, edge_size), (edge_size, edge_size)), constant_values=255)

    normalized = 1.0 - (final.astype(np.float32) / 255.0)
    return np.expand_dims(normalized, axis=(0, -1))

# Prediction logic
def predict_image(img):
    input_tensor = preprocess_js_style(img)
    if input_tensor is None:
        return [("?", 0.0)]

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    top_indices = output.argsort()[-3:][::-1]  # Top 3 predictions
    results = [(characters[i], float(output[i]) * 100) for i in top_indices]
    return results

# Default route
@app.route("/")
def home():
    return "✅ VoxScribe Character Recognition API running."

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    predictions = predict_image(img)
    top_char, top_confidence = predictions[0]

    return jsonify({
        "prediction": top_char,
        "confidence": round(top_confidence, 2),
        "alternatives": [
            {"char": c, "confidence": round(conf, 2)}
            for c, conf in predictions
        ]
    })

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
