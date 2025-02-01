#Reference filtered
#open cv2 testing code for the line follower  WITH A MASK
#Team Fernando G. Gutierrez M. , Pierre JB Hantson, Valentin Belardi

import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

video_path = 'footage/mp4TestLaneDetection5.mp4'

cap  = cv2.VideoCapture(video_path)

count = 0
#FALTA CALCULAR OFFSET O ERROR COMO SEA XD JAJA ODIO FUZZY LOGIC
lastMeans = [0, 0, 0, 0, 0]

#temporary arrays for comparing the reuslts of the "filter"
filteredInfo = [] #array for the filtered data index 1
filteredInfo2 = [] #index 1.5
filteredInfo3 = [] #index 2
filteredInfo4 = [] #index 2.5
filteredInfo5 = [] #index 3
filteredInfo6 = [] #index 3.5
filteredInfo7 = [] #index 4

#array for all of the means at the moment
meanArr = [] 

#set point that is the center of the camera, calculated below with the crop info
setPoint = 225

while (True):
    #The method read is used at the object "cap" to extract the frame from the given source

    ret, frame = cap.read()
    if not ret:
        print("valio orto")
        break
    
    #cvtColor is the method to convert the frames extracted to another color space 
    #Its parameters here are src (the frame it receives) and the color space conversion code (in this case to BGR)
    #The method will return the frame converted in the specified color space
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #brightImage = cv2.addWeighted(frame, -2, np.zeros(frame.shape, frame.dtype), 0, 1)
    alow = image.min()
    ahigh = image.max()
    print("MIN: ", alow)
    print("MAX: ", ahigh)
    amax = 255
    amin = 70

    alpha = ((amax-amin)/(ahigh-alow))
    print("alpha: ", image.mean())
    #alpha = ((ahigh-alow/(amax-amin)))
    beta = amin - alow * alpha 

    image2 = np.copy(image)
    brightImage = cv2.convertScaleAbs(image2, alpha=alpha, beta=beta)
    
    gray_raw_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    #glow = gray_raw_image.min()
    #ghigh = gray_raw_image.max()

    #print("MIN: ", glow)
    #print("MAX: ", ghigh)
    #alpha = ((amax-amin)/(ghigh-glow))


    #flip is the method to "forgive the redundancy" flip the image either vertically or horizontally 
    #its parameters are source(it must be an array or well image) and the flip code (1 for x, 0 for y and -1 for both)
    #It will return the flipped image
    #gray_bright = cv2.convertScaleAbs(gray_raw_image, alpha=alpha)
    
    #gray_vertical_flip = cv2.flip(gray_raw_image, 1)
    #gray_flip_repaired = cv2.flip(gray_vertical_flip, 0)

    #blur is the method to smooth the image by averaging the pixel values in a neighborhood defined by a kernel
    #Its paremeters are src (the input image) and k size (size of the kernel)
    #The method will return the blurred image
    blurred = cv2.blur(gray_raw_image, (5,7))
    crop = blurred[400:720, 300:980]
    #threshold is the method to apply a fixed value thresholding to a grayscale image
    #Its parameters are src (input image), thresh (threshold value), maxval(the maximum value to use with) and the type of threshold applied (if set to 0, it will be the "THRESH_BINARY" type)
    #The method will return the thresholded image (IN THIS CASE THE BINARY OF THE THRESHOLDED IMAGE)
    ret, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)

    #Here the interest area (where the center line is located will be extracted)
    #[y0:y1,x0:x1], y0 starts from above and x0 starts from the left
    #crop = thresh[400:720,450:950]  
    #crop = thresh[400:720, 300:980]

    #brightImage = cv2.equalizeHist(crop) #DID NOT WORK AT ALL SHT

    #Canny is the method to detect edges in an image using the Canny algorithm
    #Its parameters are image (the input image), threshold1 (the first threshold for the hysteresis procedure), threshold2 (the second threshold for the hysteresis procedure), and optionally apertureSize (size of the Sobel kernel used to find image gradients)
    #The method will return an image of the edges found
    highThresh = ret
    lowThresh = ret*0.5
    print(highThresh)
    dst = cv2.Canny(thresh, float(highThresh), float(lowThresh), None, 5)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    cdstP = np.copy(cdst)
    
    #HoughLines is the method to detect lines in an edge image using the Hough Line Transform
    #Its parameters are image (the edge image), rho (the distance resolution of the accumulator in pixels), theta (the angle resolution of the accumulator in radians), and threshold (the accumulator threshold parameter. Only those lines are returned that get enough votes)
    #The method will return an output vector of lines
    linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 50, None, 50, 20) #lines p returns an array with x1,y1,x2,y2
    cut = dst[150:320, 0:680]
    arrayx1 = linesP[:,0,0]
    arrayx2 = linesP[:,0,2]
    arrayy1 = linesP[:,0,1]
    arrayy2 = linesP[:,0,3]

    meanval = np.mean(arrayx1) #the mean of the points at x axis
    meanx2 = np.mean(arrayx2)
    meany1 = np.mean(arrayy1)
    meany2 = np.mean(arrayy2)
    #print(meanval)
    #meanArr.append(meanval)
    
    #get the regresion with the last five values
    lastMeans[4] = lastMeans[3]
    lastMeans[3] = lastMeans[2]    
    lastMeans[2] = lastMeans[1]
    lastMeans[1] = lastMeans[0]
    lastMeans[0] = meanval

    timeArr = list(range(0, len(lastMeans)))  
    regression = np.poly1d(np.polyfit(timeArr, lastMeans, 1))
    #print(regression(1))
    filteredInfo.append(regression(1)) #append info to the "filtered info"
    filteredInfo2.append(regression(1.5))
    filteredInfo3.append(regression(2))
    filteredInfo4.append(regression(2.5))
    filteredInfo5.append(regression(3))
    filteredInfo6.append(regression(3.5))
    filteredInfo7.append(regression(4))
    
    slope = (meany2-meany1)/(meanx2-meanval)
    generalYmean = meany2+meany1/2
    y_coords, x_coords = np.where(dst == 255)  # Get the coordinates where the edges are white
    yCannyMean = np.mean(y_coords)
    meanArr.append(yCannyMean)
    print("Slope: ", slope)
    #draw the set point at that time 
    #cv2.line(cdstP, (int(meanx2), 0), (int(meanval), 400), (0,0,255), 3, cv2.LINE_AA)
    cv2.line(cdstP, (0, int(yCannyMean)), (400, int(yCannyMean)), (0,0,255), 3, cv2.LINE_AA)
    cv2.line(cdstP, (0, int(generalYmean)), (400, int(generalYmean)), (0,255,0), 3, cv2.LINE_AA)
    error = setPoint-regression(2)
    #print(regression(2))
    #print the error 
    time.sleep(0.15)

    cv2.imshow("normal", frame)
    cv2.imshow("modified bright", brightImage)
    cv2.imshow("blur", thresh)
    cv2.imshow("crop", crop)
    cv2.imshow("cut", cut)
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