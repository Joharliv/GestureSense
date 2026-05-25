import os
import pandas as pd
import pickle
from cvzone.HandTrackingModule import HandDetector
from collections import deque, Counter

# =========================
# GLOBALS
# =========================
static_model = None
static_columns = None
motion_model = None
motion_columns = None
detector = None

sequence = []
prediction_buffer = deque(maxlen=10)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================
# LOAD ON DEMAND
# =========================
def load_resources():
    global static_model, static_columns, motion_model, motion_columns, detector

    if static_model is None:
        with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
            static_model, static_columns = pickle.load(f)

    if motion_model is None:
        with open(os.path.join(BASE_DIR, "model_motion.pkl"), "rb") as f:
            motion_model, motion_columns = pickle.load(f)

    if detector is None:
        detector = HandDetector(maxHands=1, detectionCon=0.7)


# =========================
# MAIN FUNCTION
# =========================
def predict_builder(frame):

    global sequence
    global stable_prediction
    global final_prediction

    hands, img = detector.findHands(frame)

    if not hands:
        return "No Hand"

    hand = hands[0]
    lmList = hand["lmList"]

    wrist_x, wrist_y, wrist_z = lmList[0]

    data = []

    for lm in lmList:
        x = lm[0] - wrist_x
        y = lm[1] - wrist_y
        z = lm[2] - wrist_z
        data.extend([x, y, z])

    max_value = max([abs(val) for val in data])

    if max_value != 0:
        data = [val / max_value for val in data]

    sequence.append(data)

    if len(sequence) > frames_to_capture:
        sequence.pop(0)

    if len(sequence) < frames_to_capture:
        return "..."

    flattened = []
    for f in sequence:
        flattened.extend(f)

    input_df = pd.DataFrame([flattened], columns=motion_columns)

    probs = motion_model.predict_proba(input_df)[0]
    motion_pred = motion_model.predict(input_df)[0]

    if max(probs) < 0.8:
        return "..."

    return str(motion_pred)