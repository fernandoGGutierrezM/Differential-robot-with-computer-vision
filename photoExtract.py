
import cv2
import os

cap = cv2.VideoCapture(0)


ret, img = cap.read()


filename  = 'save.jpg'

cv2.imwrite(filename, img)

