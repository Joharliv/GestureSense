print("APP STARTING...")
from flask import Flask, request, jsonify
print("Flask imported")
from flask_cors import CORS
print("CORS imported")
import os
import base64
import cv2
import numpy as np
print("Basic imports imported")

# ======================================
# IMPORT PREDICTION FUNCTIONS
# ======================================
print("Importing predict...")
from predict import predict_static
print("predict imported")

print("Importing motion...")
from predict_motion import predict_builder
print("motion imported")

print("Importing word...")
from predict_word import predict_word
print("word imported")

print("Importing builder...")
from work_builder import (
    predict_gesture,
    select_suggestion
)
print("builder imported")
# ======================================
# FLASK APP
# ======================================

app = Flask(__name__)

CORS(app)


# ======================================
# IMAGE DECODER
# ======================================

def decode_image(image_data):

    try:

        # REMOVE HEADER

        encoded_data = image_data.split(",")[1]

        # BASE64 → NUMPY

        np_data = np.frombuffer(

            base64.b64decode(encoded_data),

            np.uint8
        )

        # NUMPY → OPENCV IMAGE

        frame = cv2.imdecode(

            np_data,

            cv2.IMREAD_COLOR
        )

        return frame

    except Exception as e:

        print("Decode Error:", e)

        return None


# ======================================
# STATIC PREDICTION
# ======================================

@app.route(
    "/predict/static",
    methods=["POST"]
)

def static_prediction():

    try:

        data = request.json

        frame = decode_image(
            data["image"]
        )

        if frame is None:

            return jsonify({

                "prediction":
                    "Image Error"
            })

        prediction = predict_static(
            frame
        )

        return jsonify({

            "prediction":
                prediction
        })

    except Exception as e:

        print("Static Error:", e)

        return jsonify({

            "prediction":
                "Error"
        })


# ======================================
# MOTION PREDICTION
# ======================================

@app.route(
    "/predict/motion",
    methods=["POST"]
)

def motion_prediction():

    try:

        data = request.json

        frame = decode_image(
            data["image"]
        )

        if frame is None:

            return jsonify({

                "prediction":
                    "Image Error"
            })

        prediction = predict_builder(
            frame
        )

        return jsonify({

            "prediction":
                prediction
        })

    except Exception as e:

        print("Motion Error:", e)

        return jsonify({

            "prediction":
                "Error"
        })


# ======================================
# WORD PREDICTION
# ======================================

@app.route(
    "/predict/word",
    methods=["POST"]
)

def word_prediction():

    try:

        data = request.json

        frame = decode_image(
            data["image"]
        )

        if frame is None:

            return jsonify({

                "prediction":
                    "Image Error"
            })

        prediction = predict_word(
            frame
        )

        return jsonify({

            "prediction":
                prediction
        })

    except Exception as e:

        print("Word Error:", e)

        return jsonify({

            "prediction":
                "Error"
        })


# ======================================
# SMART BUILDER
# ======================================

@app.route(
    "/predict/builder",
    methods=["POST"]
)

def builder_prediction():

    try:

        data = request.json

        frame = decode_image(
            data["image"]
        )

        if frame is None:

            return jsonify({

                "prediction":
                    "Image Error"
            })

        result = predict_gesture(
            frame
        )

        return jsonify(result)

    except Exception as e:

        print("Builder Error:", e)

        return jsonify({

            "prediction":
                "Error"
        })


# ======================================
# HOME ROUTE
# ======================================

@app.route("/")

def home():

    return jsonify({

        "message":
            "GestureSense Backend Running"
    })


@app.route(
    "/select-suggestion",
    methods=["POST"]
)

def choose_suggestion():

    data = request.json

    selected_word = data["suggestion"]

    result = select_suggestion(
        selected_word
    )

    return jsonify(result)

# ======================================
# RUN APP
# ======================================

if __name__ == "__main__":

    try:

        print("Starting GestureSense Backend...")

        app.run(
         host="0.0.0.0",
         port=int(os.environ.get("PORT", 5000)),
          debug=False
        )

    except Exception as e:

        import traceback

        print("FULL ERROR:")
        print(traceback.format_exc())