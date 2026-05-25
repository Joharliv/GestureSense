import os
import cv2
import numpy as np
import pandas as pd
import pickle
import requests
import threading

from collections import deque, Counter

from tensorflow.keras.models import load_model

from cvzone.HandTrackingModule import HandDetector


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# APIs
# =========================================================

DATAMUSE_API = "https://api.datamuse.com/sug"

LANGUAGE_TOOL_API = \
    "https://api.languagetool.org/v2/check"


# =========================================================
# HELPERS
# =========================================================

def replace_last_word(sentence, new_word):

    words = sentence.strip().split(" ")

    if len(words) == 0:

        return new_word.upper() + " "

    words[-1] = new_word.upper()

    return " ".join(words) + " "


# =========================================================
# API FUNCTIONS
# =========================================================

def get_suggestions(word):

    try:

        res = requests.get(

            DATAMUSE_API,

            params={
                "s": word,
                "max": 5
            },

            timeout=1
        )

        return [

            w["word"].upper()

            for w in res.json()
        ]

    except:

        return []


def autocorrect(word):

    try:

        res = requests.post(

            LANGUAGE_TOOL_API,

            data={
                "text": word,
                "language": "en-US"
            },

            timeout=1
        )

        matches = res.json().get(
            "matches",
            []
        )

        if matches:

            rep = matches[0].get(
                "replacements",
                []
            )

            if rep:

                return rep[0]["value"].upper()

    except:

        pass

    return word.upper()


# =========================================================
# LOAD STATIC MODEL
# =========================================================

static_model_path = os.path.join(
    BASE_DIR,
    "model.pkl"
)

with open(static_model_path, "rb") as f:

    static_model, static_columns = \
        pickle.load(f)


# =========================================================
# LOAD MOTION MODEL
# =========================================================

motion_model_path = os.path.join(
    BASE_DIR,
    "model_motion.pkl"
)

with open(motion_model_path, "rb") as f:

    motion_model, motion_columns = \
        pickle.load(f)


# =========================================================
# LOAD WORD MODEL
# =========================================================

word_model_path = os.path.join(
    BASE_DIR,
    "gesture_lstm.h5"
)

word_model = load_model(
    word_model_path
)


label_path = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

with open(label_path, "rb") as f:

    word_le = pickle.load(f)


# =========================================================
# HAND DETECTOR
# =========================================================

detector = HandDetector(

    maxHands=1,

    detectionCon=0.7
)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

sequence = []

frames_to_capture = 30

letter_buffer = deque(maxlen=15)

last_letter = ""

cooldown = 0


word_sequence = []

word_predictions = deque(maxlen=10)

prev_word_data = None

word_cooldown = 0

last_added_word = ""


word = ""

sentence = ""

suggestions = []

last_api_word = ""

pause_frames = 0

SPACE_THRESHOLD = 20


display_prediction = ""

display_type = ""


# =========================================================
# SUGGESTION FETCH
# =========================================================

def fetch_api_results(current_word):

    global suggestions

    sug = get_suggestions(
        current_word.lower()
    )

    corrected = autocorrect(
        current_word.lower()
    )

    if corrected != current_word.upper():

        sug.insert(0, corrected)

    suggestions = sug[:5]


# =========================================================
# MAIN ENGINE
# =========================================================

def predict_gesture(frame):

    global sequence
    global letter_buffer
    global last_letter
    global cooldown

    global word_sequence
    global word_predictions
    global prev_word_data
    global word_cooldown
    global last_added_word

    global word
    global sentence
    global suggestions
    global last_api_word
    global pause_frames

    global display_prediction
    global display_type


    hands, img = detector.findHands(frame)
    print("HANDS:", len(hands), flush=True)

    if hands:

        pause_frames = 0

        lmList = hands[0]["lmList"]

        wrist = lmList[0]


        # ==========================================
        # NORMALIZE LANDMARKS
        # ==========================================

        data = []

        for lm in lmList:

            x = lm[0] - wrist[0]
            y = lm[1] - wrist[1]
            z = lm[2] - wrist[2]

            data.extend([x, y, z])


        max_val = max(map(abs, data)) or 1

        data = [v / max_val for v in data]


        # ==========================================
        # STATIC LETTER PREDICTION
        # ==========================================

        sequence.append(data)

        letter_prediction = ""


        if len(sequence) == frames_to_capture:

            flat = [

                v

                for frame_data in sequence

                for v in frame_data
            ]


            df_motion = pd.DataFrame(

                [flat],

                columns=motion_columns
            )


            probs = motion_model.predict_proba(
                df_motion
            )[0]


            if max(probs) > 0.65:

                letter_prediction = \
                    motion_model.predict(
                        df_motion
                    )[0]


            sequence.pop(0)


        # ==========================================
        # STATIC FALLBACK
        # ==========================================

        if letter_prediction == "":

            df_static = pd.DataFrame(

                [data + [0,0,0,0]],

                columns=static_columns
            )

            letter_prediction = \
                static_model.predict(
                    df_static
                )[0]


        # ==========================================
        # WORD PREDICTION
        # ==========================================

        if prev_word_data is not None:

            velocity = [

                curr - prev

                for curr, prev in zip(
                    data,
                    prev_word_data
                )
            ]

        else:

            velocity = [0] * len(data)


        combined_word_data = \
            data + velocity

        prev_word_data = data


        word_sequence.append(
            combined_word_data
        )


        if len(word_sequence) > 30:

            word_sequence.pop(0)


        word_prediction = ""


        if len(word_sequence) == 30:

            input_data = np.array(
                word_sequence
            ).reshape(1, 30, -1)


            probs = word_model.predict(

                input_data,

                verbose=0
            )[0]


            word_conf = max(probs)


            if word_conf > 0.60:

                pred_word = \
                    word_le.inverse_transform(

                        [np.argmax(probs)]

                    )[0]


                word_predictions.append(
                    pred_word
                )


                word_prediction = Counter(

                    word_predictions

                ).most_common(1)[0][0]


        # ==========================================
        # PRIORITY LOGIC
        # ==========================================

        if word_prediction != "":

            display_prediction = \
                word_prediction.upper()

            display_type = "WORD"


            if word_cooldown > 0:

                word_cooldown -= 1


            if (

                word_cooldown == 0

                and

                word_prediction != \
                last_added_word
            ):

                sentence += \
                    word_prediction.upper() + " "

                last_added_word = \
                    word_prediction

                word = ""

                last_letter = ""

                letter_buffer.clear()

                word_cooldown = 50


        else:

            display_prediction = \
                letter_prediction.upper()

            display_type = "LETTER"


            letter_buffer.append(
                letter_prediction
            )


            if len(letter_buffer) == \
                letter_buffer.maxlen:

                current_letter = Counter(

                    letter_buffer

                ).most_common(1)[0][0]

            else:

                current_letter = ""


            if cooldown > 0:

                cooldown -= 1


            if (

                current_letter

                and

                current_letter != \
                last_letter

                and

                cooldown == 0
            ):

                word += \
                    current_letter.upper()

                last_letter = \
                    current_letter

                cooldown = 20


                current_word = \
                    word.strip().split(" ")[-1]


                if (

                    current_word != \
                    last_api_word

                    and

                    len(current_word) >= 1
                ):

                    last_api_word = \
                        current_word

                    threading.Thread(

                        target=fetch_api_results,

                        args=(current_word,),

                        daemon=True

                    ).start()


    else:

        pause_frames += 1


        if (

            pause_frames == SPACE_THRESHOLD

            and

            word.strip()
        ):

            sentence += word.strip() + " "

            word = ""

            last_letter = ""

            letter_buffer.clear()

            suggestions = []


    # ==========================================
    # RETURN DATA
    # ==========================================

    return {

        "prediction": display_prediction,

        "type": display_type,

        "word": word,

        "sentence": sentence,

        "suggestions": suggestions
    }

# ==========================================
# SELECT SUGGESTION
# ==========================================

def select_suggestion(selected_word):

    global word
    global sentence
    global suggestions


    sentence += selected_word.upper() + " "

    word = ""

    suggestions = []


    return {

        "sentence": sentence
    }