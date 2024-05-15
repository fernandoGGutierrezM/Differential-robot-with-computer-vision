import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

video_path = 'mp4TestLaneDetection2.mp4'

cap  = cv2.VideoCapture(video_path)

count = 0
meanArr = []

while (True):
    ret, frame = cap.read()
    if not ret:
        print("valio orto")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray_raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #       y0
    #       |
    #  x0---|---xf
    #       |
    #       yf

    ret, thresh = cv2.threshold(gray_raw_image, 127, 255, 0)
    crop = thresh[0:478,200:650]   #crop the image to only see the line (y0,y1,x0,x1)
    dst = cv2.Canny(crop, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, np.pi/90, 50, None, 50, 10) #lines p returns an array with x1,y1,x2,y2

    arrayx1 = linesP[:,0,0]
    meanval = np.mean(arrayx1)
    #print(meanval)
    meanArr.append(meanval)
    cv2.line(cdstP, (int(meanval), 0), (int(meanval), 400), (0,0,255), 3, cv2.LINE_AA)
    print(meanArr)

    if(count == 5):        
        #model = np.poly1d(np.polyfit(timeArr, meanArr, 3))
        #MAKE REGRESION WITH THE 5 PREVIOUS VALUES
        #meanArr.clear()
        count = 0

    cv2.imshow("blur", thresh)
    cv2.imshow("crop", crop)
    cv2.imshow("Probabilistic Houg  Transform", cdstP)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fiveMean = []
for i in range (0, 5):
    fiveMean.append(meanArr[i])
print("-----------------------------")
timeArr = list(range(0, len(fiveMean)))
print(len(fiveMean))
print(len(timeArr))
model = np.poly1d(np.polyfit(timeArr, fiveMean, 2))
plt.plot(fiveMean)
plt.plot(timeArr, model(timeArr))
plt.show()
    

#timeArr = list(range(0, len(meanArr)))
#model = np.poly1d(np.polyfit(timeArr, meanArr, 5))
#plt.plot(meanArr)
#plt.plot(timeArr, model(timeArr))
#plt.show()

cap.release()
cv2.destroyAllWindows()
