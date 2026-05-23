import os
import cv2
import pandas as pd
import pickle

from cvzone.HandTrackingModule import HandDetector

from collections import deque, Counter


# ======================================
# PATHS
# ======================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ======================================
# LOAD STATIC MODEL
# ======================================

static_model_path = os.path.join(
    BASE_DIR,
    "model.pkl"
)

with open(static_model_path, "rb") as f:

    static_model, static_columns = pickle.load(f)


# ======================================
# LOAD MOTION MODEL
# ======================================

motion_model_path = os.path.join(
    BASE_DIR,
    "model_motion.pkl"
)

with open(motion_model_path, "rb") as f:

    motion_model, motion_columns = pickle.load(f)


# ======================================
# HAND DETECTOR
# ======================================

detector = HandDetector(

    maxHands=1,

    detectionCon=0.7
)


# ======================================
# GLOBAL VARIABLES
# ======================================

sequence = []

frames_to_capture = 30

prediction_buffer = deque(maxlen=10)

stable_prediction = ""

final_prediction = ""


# ======================================
# MAIN FUNCTION
# ======================================

def predict_builder(frame):

    global sequence
    global stable_prediction
    global final_prediction


    hands, img = detector.findHands(frame)


    if hands:

        hand = hands[0]

        lmList = hand["lmList"]


        wrist_x, wrist_y, wrist_z = lmList[0]


        data = []


        # ==============================
        # NORMALIZE LANDMARKS
        # ==============================

        for lm in lmList:

            x = lm[0] - wrist_x
            y = lm[1] - wrist_y
            z = lm[2] - wrist_z

            data.extend([x, y, z])


        max_value = max([abs(val) for val in data])


        if max_value != 0:

            data = [

                val / max_value

                for val in data
            ]


        # ==============================
        # EXTRA FEATURES
        # ==============================

        x4 = data[4 * 3]
        y4 = data[4 * 3 + 1]

        x8 = data[8 * 3]
        y8 = data[8 * 3 + 1]

        x12 = data[12 * 3]
        y12 = data[12 * 3 + 1]


        thumb_index_distance = (

            (x4 - x8) ** 2 +
            (y4 - y8) ** 2

        ) ** 0.5


        index_middle_distance = (

            (x8 - x12) ** 2 +
            (y8 - y12) ** 2

        ) ** 0.5


        thumb_middle_distance = (

            (x4 - x12) ** 2 +
            (y4 - y12) ** 2

        ) ** 0.5


        x_diff = abs(x8 - x12)


        data_extended = data + [

            thumb_index_distance,
            index_middle_distance,
            thumb_middle_distance,
            x_diff
        ]


        # ==============================
        # STATIC PREDICTION
        # ==============================

        static_input = pd.DataFrame(

            [data_extended],

            columns=static_columns
        )


        static_pred = static_model.predict(

            static_input

        )[0]


        # ==============================
        # MOTION SEQUENCE
        # ==============================

        sequence.append(data)


        # ==============================
        # MOVEMENT DETECTION
        # ==============================

        movement_threshold = 0.01


        if len(sequence) > 1:

            diff = sum(

                abs(a - b)

                for a, b in zip(
                    sequence[-1],
                    sequence[-2]
                )

            ) / len(sequence[-1])

        else:

            diff = 0


        is_moving = diff > movement_threshold


        # ==============================
        # MOTION PREDICTION
        # ==============================

        if len(sequence) == frames_to_capture:


            flattened = []


            for frame_data in sequence:

                flattened.extend(frame_data)


            input_df = pd.DataFrame(

                [flattened],

                columns=motion_columns
            )


            probs = motion_model.predict_proba(

                input_df

            )[0]


            max_prob = max(probs)


            if max_prob > 0.9:

                motion_pred = motion_model.predict(

                    input_df

                )[0]

            else:

                motion_pred = ""


            prediction_buffer.append(
                motion_pred
            )


            if len(prediction_buffer) == prediction_buffer.maxlen:

                stable_prediction = Counter(

                    prediction_buffer

                ).most_common(1)[0][0]


            sequence.pop(0)


            # ==========================
            # FINAL SMART PREDICTION
            # ==========================

            if (

                motion_pred != ""

                and

                motion_pred in ["J", "Z"]

            ):

                final_prediction = motion_pred


            elif (

                is_moving

                and

                stable_prediction in ["J", "Z"]

            ):

                final_prediction = stable_prediction


            else:

                final_prediction = static_pred


        return final_prediction


    return "No Hand"