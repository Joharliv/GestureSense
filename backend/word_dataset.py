import cv2
from cvzone.HandTrackingModule import HandDetector
import csv

cap = cv2.VideoCapture(0)
detector = HandDetector(
    staticMode=False,
    maxHands=2,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

file = open("dataset_lstm.csv", "a", newline="")
writer = csv.writer(file)

label = "hello"

sequence = []
recording = False

frames_to_capture = 30
frame_skip = 1
frame_count = 0

prev_data = None

print("h=hello | i=hi | t=thanks | y=yes | n=no | w=water | e=eat")
print("Press R to record | ESC to exit")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    frame_count += 1

    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]["lmList"]

        wrist_x, wrist_y, wrist_z = lmList[0]

        data = []
        for lm in lmList:
            x = lm[0] - wrist_x
            y = lm[1] - wrist_y
            z = lm[2] - wrist_z
            data.extend([x, y, z])

        max_val = max([abs(val) for val in data])
        if max_val != 0:
            data = [val / max_val for val in data]

        # 🔥 ADD VELOCITY
        if prev_data is not None:
            velocity = [curr - prev for curr, prev in zip(data, prev_data)]
        else:
            velocity = [0] * len(data)

        combined = data + velocity
        prev_data = data

        if recording and frame_count % frame_skip == 0:
            sequence.append(combined)

            cv2.putText(img, f"Recording {len(sequence)}/{frames_to_capture}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if len(sequence) == frames_to_capture:
            flattened = []
            for frame in sequence:
                flattened.extend(frame)

            writer.writerow(flattened + [label])
            print(f"{label} saved")

            sequence = []
            recording = False

        cv2.putText(img, f"Label: {label}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dataset", img)
    key = cv2.waitKey(1)

    if key == ord('h'): label = "hello"
    if key == ord('t'): label = "thanks"
    if key == ord('y'): label = "yes"
    if key == ord('n'): label = "no"
    if key == ord('1'): label = "My"
    if key == ord('2'): label = "name"
    if key == ord('3'): label = "is"
    if key == ord('4'): label = "how are you"
    if key == ord('5'): label = "sorry"
    if key == ord('6'): label = "I am Fine"
    if key == ord('7'): label = "please"
    if key == ord('8'): label = "what"
    if key == ord('9'): label = "your"





    



    if key == ord('r'):
        recording = True
        sequence = []
        prev_data = None
        print(f"Recording {label}")

    if key == 27:
        break

file.close()
cap.release()
cv2.destroyAllWindows()

