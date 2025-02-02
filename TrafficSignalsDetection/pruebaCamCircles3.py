#Detecting the color of the traffic light with only the color intensity at the image
import cv2
import numpy as np

def detect_traffic_light_color(frame):
    # Convert frame to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Define color ranges in RGB
    red_lower = np.array([67, 60, 71])
    red_upper = np.array([96, 91, 83])

    green_lower = np.array([72, 47, 56])
    green_upper = np.array([115, 77, 68])
    
    yellow_lower = np.array([60, 54, 57])
    yellow_upper = np.array([96, 88, 72])

#    red_lower = np.array([136, 87, 111])
#    red_upper = np.array([180, 255, 255])
#    green_lower = np.array([0, 100, 0])
#    green_upper = np.array([150, 255, 178])
#    yellow_lower = np.array([20, 100, 100])
#    yellow_upper = np.array([30, 255, 255])


    # Create masks for different colors
    red_mask = cv2.inRange(rgb_image, red_lower, red_upper)
    green_mask = cv2.inRange(rgb_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(rgb_image, yellow_lower, yellow_upper)

    # Determine the color by checking the sum of white pixels in masks
    red_sum = np.sum(red_mask)
    green_sum = np.sum(green_mask)
    yellow_sum = np.sum(yellow_mask)

    # Detect the color with the maximum pixel count
    if red_sum > green_sum and red_sum > yellow_sum:
        return 'Red'
    elif green_sum > red_sum and green_sum > yellow_sum:
        return 'Green'
    elif yellow_sum > red_sum and yellow_sum > green_sum:
        return 'Yellow'
    else:
        return 'No Traffic Light Detected'

def main():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        (B, G, R) = cv2.split(frame)

        alowB = B.min()
        ahighB = B.max()
        alowG = G.min()
        ahighG = G.max()
        alowR = R.min()
        ahighR = R.max()

        amaxB = 200
        aminB = 0
        amaxG = 200
        aminG = 0
        amaxR = 200
        aminR = 0

        alphaB = ((amaxB-aminB)/(ahighB-alowB))
        betaB = aminB - alowB * alphaB
        
        alphaG = ((amaxG-aminG)/(ahighG-alowG))
        betaG = aminG - alowG * alphaG
        
        alphaR = ((amaxR-aminR)/(ahighR-alowR))
        betaR = aminR - alowR * alphaR

        brightImageB = cv2.convertScaleAbs(frame[:,:,0], alpha=alphaB, beta=betaB)
        brightImageG = cv2.convertScaleAbs(frame[:,:,1], alpha=alphaG, beta=betaG)
        brightImageR = cv2.convertScaleAbs(frame[:,:,2], alpha=alphaR, beta=betaR)
        modImage = cv2.merge([brightImageB, brightImageG, brightImageR])

        # Detect the color of the traffic light
        traffic_light_color = detect_traffic_light_color(modImage)
        print(f"Detected Traffic Light Color: {traffic_light_color}")

        # Display the frame
        cv2.imshow('Traffic Light Detection', modImage)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
