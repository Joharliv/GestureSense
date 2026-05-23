import cv2
from cvzone.HandTrackingModule import HandDetector
import csv


cap = cv2.VideoCapture(0)


detector = HandDetector(maxHands=1, detectionCon=0.7)


file = open("dataset.csv", "a", newline="")
writer = csv.writer(file)


label = "A"

print("Press A/B/C to change label")
print("Press S to save data")
print("Press ESC to exit")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        
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

        
        cv2.putText(img, f"Label: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

   
    if key == ord('a'):
        label = "A"
        print("Label changed to A")

    if key == ord('b'):
        label = "B"
        print("Label changed to B")

    if key == ord('c'):
        label = "C"
        print("Label changed to C")

    if key == ord('d'):
        label = "D"
        print("Label changed to D")

    if key == ord('e'):
        label = "E"
        print("Label changed to E")
        
    if key == ord('f'):
       label = "F"
       print("Label changed to F")

    if key == ord('g'):
       label = "G"
       print("Label changed to G")

    if key == ord('h'):
       label = "H"
       print("Label changed to H")

    if key == ord('i'):
       label = "I"
       print("Label changed to I")

    if key == ord('j'):
       label = "J"
       print("Label changed to J")
    
    if key == ord('k'):
       label = "K"
       print("Label changed to K")

    if key == ord('l'):
       label = "L"
       print("Label changed to L")

    if key == ord('m'):
       label = "M"
       print("Label changed to M")
    
    if key == ord('n'):
       label = "N"
       print("Label changed to N")

    if key == ord('o'):
         label = "O"
         print("Label changed to O")
    
    if key == ord('p'):
         label = "P"
         print("Label changed to P")
    
    if key == ord('q'):
         label = "Q"
         print("Label changed to Q")  
    
    if key == ord('r'):
         label = "R"
         print("Label changed to R")        

    if key == ord('$'):
         label = "S"
         print("Label changed to S")

    if key == ord('t'):
         label = "T"
         print("Label changed to T")

    if key == ord('u'):
         label = "U"
         print("Label changed to U")

    if key == ord('v'):
         label = "V"
         print("Label changed to V")
   
    if key == ord('w'):
         label = "W"
         print("Label changed to W")

    if key == ord('x'):
         label = "X"
         print("Label changed to X")

    if key == ord('y'):
         label = "Y"
         print("Label changed to Y")


   
    if key == ord('s') and hands:
        writer.writerow(data + [label])
        print("Saved one sample")

  
    if key == 27:
        break

file.close()
cap.release()
cv2.destroyAllWindows()