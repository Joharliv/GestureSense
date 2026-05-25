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

    load_resources()

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

    max_value = max([abs(v) for v in data])
    if max_value != 0:
        data = [v / max_value for v in data]

    sequence.append(data)

    if len(sequence) > 30:
        sequence.pop(0)

    if len(sequence) < 30:
        return "..."

    flattened = []
    for f in sequence:
        flattened.extend(f)

    # simple prediction (kept same logic)
    import pandas as pd

    input_df = pd.DataFrame([flattened], columns=motion_columns)

    probs = motion_model.predict_proba(input_df)[0]
    max_prob = max(probs)

    if max_prob > 0.9:
        return motion_model.predict(input_df)[0]

    return ""