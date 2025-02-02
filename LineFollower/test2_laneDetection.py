#opencv2 testing for video or image: Probabilistic hough
#Transform, gaussian blur and a crop from gaussian blur
#Display a graph of the results and a regession of them

import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

#video_path = 'imgTestLaneDetection3.jpeg'
video_path = 'footage/mp4TestLaneDetection2.mp4'

cap  = cv2.VideoCapture(video_path)

# Crop properly so we get the line
# Find mean of the lines
# 
meanArr = []
while (True):
    ret, frame = cap.read()
    if not ret:
        print("valio orto")
        break
    #print(cap.isOpened())
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray_raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray_raw_image[5][2])
    gray_vertical_flip = cv2.flip(gray_raw_image, 1)
    gray_flip_repaired = cv2.flip(gray_vertical_flip, 0)
     
    #       y0
    #       |
    #  x0---|---xf
    #       |
    #       yf

    ret, thresh = cv2.threshold(gray_flip_repaired, 127, 255, 0)
    crop = thresh[200:478,0:650]   #crop the image to only see the line (y0,y1,x0,x1)
    #smooth_image = cv2.GaussianBlur(thresh, (5,7), 3)
    dst = cv2.Canny(crop, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, np.pi/90, 50, None, 50, 10) #lines p returns an array with x1,y1,x2,y2
    

    #puntos = linesP[:,0,:]    #[a][b][c] a is the x1,x2,x3,x4 index; b is the line and c is the entire column, this part prints the entire list for the iteration
    arrayx1 = linesP[:,0,0]
    meanval = np.mean(arrayx1)
    #print(meanval)
    meanArr.append(meanval)
    cv2.line(cdstP, (int(meanval), 0), (int(meanval), 400), (0,0,255), 3, cv2.LINE_AA)
    print(meanArr)
    
    
    #cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
       #if linesP is not None:
        #for i in range(0, len(linesP)): 
            #l = linesP[i][0] #This is the array with the lines actually
            #print(0.5)
            #arrx1 = linesP[3][0][0] #realmente este es un punto de el array, por eso el mean no funcionaba
            #arrx2 = linesP[i][0][2] 
            #print(arrx1)
            #arrAux = np.array(arrx1)
            #np.mean(arrAux)
            #print(arrx1)
            #print('--------------')
            #print(arrAux)
            #cv2.line(cdstP, (arrx1,0),(arrx1,400), (0,0,255), 3, cv2.LINE_AA)
            #print(linesP[i][0][1])
            #cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)  #what is LINE_AA
    
    #cv2.imshow("normal", gray_flip_repaired)
    #time.sleep(0.01)
    #print(linesP)
    cv2.imshow("blur", thresh)
    cv2.imshow("crop", crop)
    cv2.imshow("Probabilistic Houg  Transform", cdstP)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

timeArr = list(range(0, len(meanArr)))
model = np.poly1d(np.polyfit(timeArr, meanArr, 5))
print(meanArr)
print("/////////////////////////")
print(len(meanArr))
print(len(timeArr))
print(len(model))
plt.plot(meanArr)
plt.plot(timeArr, model(timeArr))
plt.show()

cap.release()
cv2.destroyAllWindows()

#https://www.w3schools.com/python/python_ml_polynomial_regression.asp
#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
#