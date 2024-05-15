#Reference filtered

import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

video_path = 'mp4TestLaneDetection2.mp4'

cap  = cv2.VideoCapture(video_path)

count = 0
#FALTA CALCULAR OFFSET O ERROR COMO SEA XD JAJA ODIO FUZZY LOGIC
lastMeans = [207, 207, 207, 207, 207]

#temporary arrays for comparing the reuslts of the "filter"
filteredInfo = [] #array for the filtered data index 1
filteredInfo2 = [] #index 1.5
filteredInfo3 = [] #index 2
filteredInfo4 = [] #index 2.5
filteredInfo5 = [] #index 3
filteredInfo6 = [] #index 3.5
filteredInfo7 = [] #index 4


meanArr = [] #array for all of the means at the moment

setPoint = 225 #set point that is the center of the camera, calculated below with the crop info
while (True):
    ret, frame = cap.read()
    if not ret:
        print("valio orto")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray_raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray_raw_image, 127, 255, 0)
    crop = thresh[0:478,200:650]   #crop the image to only see the line (y0,y1,x0,x1)
    #650px-200px = 450 then /2 it is 225px + crop = 425 px as SP
    
    dst = cv2.Canny(crop, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, np.pi/90, 50, None, 50, 10) #lines p returns an array with x1,y1,x2,y2

    arrayx1 = linesP[:,0,0]
    meanval = np.mean(arrayx1) #the mean of the points at x axis
    #print(meanval)
    meanArr.append(meanval)

    #get the regresion with the last five values
    lastMeans[4] = lastMeans[3]
    lastMeans[3] = lastMeans[2]    
    lastMeans[2] = lastMeans[1]
    lastMeans[1] = lastMeans[0]
    lastMeans[0] = meanval

    timeArr = list(range(0, len(lastMeans)))  
    regression = np.poly1d(np.polyfit(timeArr, lastMeans, 3))
    #print(regression(1))
    filteredInfo.append(regression(1)) #append info to the "filtered info"
    filteredInfo2.append(regression(1.5))
    filteredInfo3.append(regression(2))
    filteredInfo4.append(regression(2.5))
    filteredInfo5.append(regression(3))
    filteredInfo6.append(regression(3.5))
    filteredInfo7.append(regression(4))

    #draw the set point at that time 
    cv2.line(cdstP, (int(regression(2.5)), 0), (int(regression(2.5)), 400), (0,0,255), 3, cv2.LINE_AA)
    error = setPoint-regression(2.5)
    print(error)
    #print the error 
    #time.sleep(0.25)

    cv2.imshow("blur", thresh)
    cv2.imshow("crop", crop)
    cv2.imshow("Probabilistic Houg  Transform", cdstP)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



plt.plot(meanArr)
#plt.plot(filteredInfo, label = 'index 1')
#plt.plot(filteredInfo2, label = 'index 1.5')
#plt.plot(filteredInfo3, label = 'index 2')
plt.plot(filteredInfo4, label = 'index 2.5')
plt.plot(filteredInfo5, label = 'index 3')
plt.plot(filteredInfo6, label = 'index 3.5')
plt.plot(filteredInfo7, label = 'index 4')
plt.legend()
plt.show()

cap.release()
cv2.destroyAllWindows()