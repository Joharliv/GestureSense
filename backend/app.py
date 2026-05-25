import sys
import os
import base64
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from predict_motion import predict_builder
from predict_word import predict_word
from work_builder import predict_gesture, select_suggestion

print("APP STARTING...", flush=True)

app = Flask(__name__)
CORS(app)


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
# MOTION / LETTER / WORD / BUILDER
# ======================================

@app.route("/predict/motion", methods=["POST", "OPTIONS"])
def motion():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"prediction": "Invalid Request"}), 400

        frame = decode_image(data["image"])

        if frame is None:
            return jsonify({"prediction": "No Hand"})

        pred = predict_builder(frame)

        return jsonify({"prediction": str(pred)})

    except Exception as e:
        print("Motion Error:", e)
        return jsonify({"prediction": "Error"})


@app.route("/predict/builder", methods=["POST", "OPTIONS"])
def builder():

    print("BUILDER ROUTE HIT", flush=True)

    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:

        data = request.get_json(silent=True)

        print("DATA RECEIVED:", data is not None, flush=True)

        if not data or "image" not in data:
            print("NO IMAGE", flush=True)

            return jsonify({
                "prediction": "Invalid Request"
            }), 400

        frame = decode_image(data["image"])

        print("FRAME DECODED:", frame is not None, flush=True)

        if frame is None:

            return jsonify({
                "prediction": "No Hand",
                "suggestions": []
            })

        result = predict_gesture(frame)

        print("RESULT:", result, flush=True)

        return jsonify(result)

    except Exception as e:

        import traceback

        print(traceback.format_exc(), flush=True)

        return jsonify({
            "prediction": "Error",
            "suggestions": []
        })


@app.route("/predict/builder", methods=["POST", "OPTIONS"])
def builder():
    if request.method == "OPTIONS":
        return jsonify({}), 200
     
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"prediction": "Invalid Request"}), 400

        frame = decode_image(data["image"])

        if frame is None:
            return jsonify({
                "prediction": "No Hand",
                "suggestions": []
            })

        result = predict_gesture(frame)

        # FORCE SAFE OUTPUT FORMAT
        if isinstance(result, dict):
            return jsonify({
                "prediction": str(result.get("prediction", "")),
                "suggestions": result.get("suggestions", [])
            })

        return jsonify({
            "prediction": str(result),
            "suggestions": []
        })

    except Exception as e:
        print("Builder Error:", e)
        return jsonify({
            "prediction": "Error",
            "suggestions": []
        })


# ======================================
# HOME
# ======================================

@app.route("/")
def home():
    return jsonify({
        "message": "GestureSense Backend Running"
    })


@app.route("/select-suggestion", methods=["POST"])
def choose():
    data = request.json
    result = select_suggestion(data["suggestion"])
    return jsonify(result)


# ======================================
# RUN
# ======================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)