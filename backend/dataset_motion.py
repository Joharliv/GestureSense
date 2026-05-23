import cv2
from cvzone.HandTrackingModule import HandDetector
import csv

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.7)

file = open("dataset_motion.csv", "a", newline="")
writer = csv.writer(file)

label = "J"

sequence = []
recording = False
frames_to_capture = 30

frame_skip = 3
frame_count = 0

print("Press J/Z to select label")
print("Press R to start recording")
print("ESC to exit")

while True:
    success, img = cap.read()
    frame_count += 1
    hands, img = detector.findHands(img)



    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        wrist_x, wrist_y, wrist_z = lmList[0]

        data = []

        # relative coordinates
        for lm in lmList:
            x = lm[0] - wrist_x
            y = lm[1] - wrist_y
            z = lm[2] - wrist_z
            data.extend([x, y, z])

        # normalize
        max_value = max([abs(val) for val in data])
        if max_value != 0:
            data = [val / max_value for val in data]

        if recording and hands:
            if frame_count % frame_skip == 0:
               sequence.append(data)

               cv2.putText(img, f"Recording {len(sequence)}/{frames_to_capture}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if len(sequence) == frames_to_capture:
               flattened = []
               for frame in sequence:
                  flattened.extend(frame)

               writer.writerow(flattened + [label])
               print(f"{label} sequence saved")

               sequence = []
               recording = False 

            cv2.putText(img, f"Recording {len(sequence)}/{frames_to_capture}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        cv2.putText(img, f"Label: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Dataset Motion", img)
    key = cv2.waitKey(1)

    if key == ord('j'):
        label = "J"
        print("Label = J")

    if key == ord('z'):
        label = "Z"
        print("Label = Z")

    if key == ord('r'):
        recording = True
        sequence = []
        print("Recording started")

    if key == 27:
        break

file.close()
cap.release()
cv2.destroyAllWindows()