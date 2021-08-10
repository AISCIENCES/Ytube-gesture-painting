import numpy as np
import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
imgCanvas = np.zeros((500,650,3),np.uint8)

while True:
    status, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow("HandDrawing", img)
    cv2.waitKey(1)



    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    canDraw = False

    
    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(idx,cx,cy)
                handPoints.append((cx,cy))

        if handPoints[12][1] < handPoints[10][1]:
            cv2.circle(img, handPoints[8], 15, (255, 255, 0))
            # print("both fingers are open")
            canDraw = False
        elif handPoints[8][1] < handPoints[6][1]:
            # print("only finger 1 is open")
            cv2.circle(img, handPoints[8], 15, (255, 255, 0), cv2.FILLED)
            canDraw = True

        if canDraw:

            cv2.circle(img, handPoints[8], 15, (255, 255, 0), cv2.FILLED)
            cv2.circle(imgCanvas, handPoints[8], 15, (255, 255, 0), cv2.FILLED)


    cv2.imshow("HandDrawing", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)

