import os
import numpy as np
import pickle
from collections import deque, Counter
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf

# =========================
# GLOBALS
# =========================
model = None
le = None
detector = None

sequence = []
predictions = deque(maxlen=10)
prev_data = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# LOAD ON DEMAND
# =========================
def load_resources():
    global model, le, detector

    if model is None:
        model = tf.keras.models.load_model(
            os.path.join(BASE_DIR, "gesture_lstm.h5")
        )

    if le is None:
        with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)

    if detector is None:
        detector = HandDetector(
            staticMode=False,
            maxHands=2,
            detectionCon=0.5,
            minTrackCon=0.5
        )


# =========================
# MAIN FUNCTION
# =========================
def predict_word(frame):

    global sequence, predictions, prev_data

    load_resources()

    hands, img = detector.findHands(frame)

    if not hands:
        return "No Hand"

    lmList = hands[0]["lmList"]
    wrist_x, wrist_y, wrist_z = lmList[0]

    data = []
    for lm in lmList:
        x = lm[0] - wrist_x
        y = lm[1] - wrist_y
        z = lm[2] - wrist_z
        data.extend([x, y, z])

    max_val = max([abs(v) for v in data])
    if max_val != 0:
        data = [v / max_val for v in data]

    if prev_data is not None:
        velocity = [c - p for c, p in zip(data, prev_data)]
    else:
        velocity = [0] * len(data)

    prev_data = data

    combined = data + velocity
    sequence.append(combined)

    if len(sequence) > 30:
        sequence.pop(0)

    if len(sequence) < 30:
        return "..."

    input_data = np.array(sequence).reshape(1, 30, -1)

    probs = model.predict(input_data, verbose=0)[0]
    confidence = max(probs)

    if confidence > 0.6:
        pred = le.inverse_transform([np.argmax(probs)])[0]
        predictions.append(pred)

        return Counter(predictions).most_common(1)[0][0]

    return "..."