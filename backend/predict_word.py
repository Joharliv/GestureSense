import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from collections import deque, Counter
from cvzone.HandTrackingModule import HandDetector


# ======================================
# LOAD MODEL
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "gesture_lstm.h5")

model = tf.keras.models.load_model(model_path)


label_path = os.path.join(BASE_DIR, "label_encoder.pkl")

with open(label_path, "rb") as f:
    le = pickle.load(f)


# ======================================
# HAND DETECTOR
# ======================================

detector = HandDetector(

    staticMode=False,

    maxHands=2,

    modelComplexity=1,

    detectionCon=0.5,

    minTrackCon=0.5
)


# ======================================
# GLOBAL VARIABLES
# ======================================

sequence = []

predictions = deque(maxlen=10)

frames_to_capture = 30

frame_skip = 1

frame_count = 0

prev_data = None

current_word = "Waiting..."


# ======================================
# MAIN PREDICTION FUNCTION
# ======================================

def predict_word(frame):

    global sequence
    global predictions
    global frame_count
    global prev_data
    global current_word


    frame_count += 1


    # Detect hands

    hands, img = detector.findHands(frame)


    if hands:

        lmList = hands[0]["lmList"]


        wrist_x, wrist_y, wrist_z = lmList[0]


        data = []


        # Normalize landmarks

        for lm in lmList:

            x = lm[0] - wrist_x
            y = lm[1] - wrist_y
            z = lm[2] - wrist_z

            data.extend([x, y, z])


        max_val = max([abs(val) for val in data])


        if max_val != 0:

            data = [val / max_val for val in data]


        # Velocity

        if prev_data is not None:

            velocity = [

                curr - prev

                for curr, prev in zip(data, prev_data)
            ]

        else:

            velocity = [0] * len(data)


        combined = data + velocity

        prev_data = data


        # Add sequence

        if frame_count % frame_skip == 0:

            sequence.append(combined)


        # Keep only 30 frames

        if len(sequence) > frames_to_capture:

            sequence.pop(0)


        # Predict

        if len(sequence) == frames_to_capture:


            input_data = np.array(sequence).reshape(1, 30, -1)


            probs = model.predict(
                input_data,
                verbose=0
            )[0]


            confidence = max(probs)


            if confidence > 0.6:


                pred = le.inverse_transform(

                    [np.argmax(probs)]

                )[0]


                predictions.append(pred)


                current_word = Counter(

                    predictions

                ).most_common(1)[0][0]

            else:

                current_word = "..."


        return current_word


    return "No Hand"