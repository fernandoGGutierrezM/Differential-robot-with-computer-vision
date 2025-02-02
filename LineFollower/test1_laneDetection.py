#Open cv2 testing for an image (comparisson between grayscale,
#color and smoothed image. Canny edge detection and Hough lines
#testing)

import cv2
import numpy as np
import math

video_path = 'footage/imgTestLaneDetection3.jpeg'

gray_raw_image= cv2.imread(video_path, cv2.IMREAD_GRAYSCALE)
gray_vertical_flip = cv2.flip(gray_raw_image, 1)
gray_flip_repaired = cv2.flip(gray_vertical_flip, 0)
ret, thresh = cv2.threshold(gray_flip_repaired, 127, 255, 0)
smooth_image = cv2.GaussianBlur(thresh, (5,7), 3)

dst = cv2.Canny(smooth_image, 50, 200, None, 3)

cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(dst, 1, np.pi/180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    
cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

 
linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("normal", gray_flip_repaired)
cv2.imshow("blur", smooth_image)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
cv2.waitKey()