import os
import pandas as pd
import cv2
import pickle
from cvzone.HandTrackingModule import HandDetector

# =========================
# GLOBALS (EMPTY INIT)
# =========================
model = None
columns = None
detector = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")


# =========================
# LOAD ON DEMAND
# =========================
def load_resources():
    global model, columns, detector

    if model is None:
        with open(model_path, "rb") as f:
            model, columns = pickle.load(f)

    if detector is None:
        detector = HandDetector(maxHands=1, detectionCon=0.7)


# =========================
# MAIN FUNCTION
# =========================
def predict_static(frame):

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

    max_value = max([abs(val) for val in data])
    if max_value != 0:
        data = [val / max_value for val in data]

    x4 = data[4 * 3]
    y4 = data[4 * 3 + 1]
    x8 = data[8 * 3]
    y8 = data[8 * 3 + 1]
    x12 = data[12 * 3]
    y12 = data[12 * 3 + 1]

    thumb_index_distance = ((x4 - x8) ** 2 + (y4 - y8) ** 2) ** 0.5
    index_middle_distance = ((x8 - x12) ** 2 + (y8 - y12) ** 2) ** 0.5
    thumb_middle_distance = ((x4 - x12) ** 2 + (y4 - y12) ** 2) ** 0.5
    x_diff = abs(x8 - x12)

    data_extended = data + [
        thumb_index_distance,
        index_middle_distance,
        thumb_middle_distance,
        x_diff
    ]

    input_df = pd.DataFrame([data_extended], columns=columns)

    return str(model.predict(input_df)[0])