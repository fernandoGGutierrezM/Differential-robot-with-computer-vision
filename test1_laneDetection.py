import cv2
import numpy as np
import time
video_path = 'mp4TestLaneDetection1.mp4'
cap = cv2.VideoCapture(video_path)


while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_frame, 100, 255, 0)

    if not ret:
        break
    time.sleep(0.05)
    cv2.imshow('Line Detection', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
