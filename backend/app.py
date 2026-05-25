import sys
print("APP STARTING...", flush=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

print("Flask app initialized", flush=True)

# ======================================
# IMAGE DECODER
# ======================================
def decode_image(image_data):
    try:
        encoded_data = image_data.split(",")[1]

        np_data = np.frombuffer(
            base64.b64decode(encoded_data),
            np.uint8
        )

        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return frame

    except Exception as e:
        print("Decode Error:", e)
        return None


# ======================================
# HOME ROUTE
# ======================================
@app.route("/")
def home():
    return jsonify({
        "message": "GestureSense Backend Running"
    })


# ======================================
# MOTION PREDICTION (LAZY IMPORT)
# ======================================
@app.route("/predict/motion", methods=["POST"])
def motion_prediction():
    try:
        from predict_motion import predict_builder  # lazy import

        data = request.json
        frame = decode_image(data.get("image"))

        if frame is None:
            return jsonify({"prediction": "Image Error"})

        prediction = predict_builder(frame)

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("Motion Error:", e)
        return jsonify({"prediction": "Error"})


# ======================================
# WORD PREDICTION (LAZY IMPORT)
# ======================================
@app.route("/predict/word", methods=["POST"])
def word_prediction():
    try:
        from predict_word import predict_word  # lazy import

        data = request.json
        frame = decode_image(data.get("image"))

        if frame is None:
            return jsonify({"prediction": "Image Error"})

        prediction = predict_word(frame)

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("Word Error:", e)
        return jsonify({"prediction": "Error"})


# ======================================
# SMART BUILDER (LAZY IMPORT)
# ======================================
@app.route("/predict/builder", methods=["POST"])
def builder_prediction():
    try:
        from work_builder import predict_gesture, select_suggestion  # lazy import

        data = request.json
        frame = decode_image(data.get("image"))

        if frame is None:
            return jsonify({"prediction": "Image Error"})

        result = predict_gesture(frame)

        return jsonify(result)

    except Exception as e:
        print("Builder Error:", e)
        return jsonify({"prediction": "Error"})


# ======================================
# SELECT SUGGESTION (LAZY IMPORT)
# ======================================
@app.route("/select-suggestion", methods=["POST"])
def choose_suggestion():
    try:
        from work_builder import select_suggestion  # lazy import

        data = request.json
        selected_word = data.get("suggestion")

        result = select_suggestion(selected_word)

        return jsonify(result)

    except Exception as e:
        print("Suggestion Error:", e)
        return jsonify({"error": "Error processing suggestion"})


# ======================================
# RUN APP (LOCAL ONLY)
# ======================================
if __name__ == "__main__":
    print("Starting GestureSense Backend...")

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )