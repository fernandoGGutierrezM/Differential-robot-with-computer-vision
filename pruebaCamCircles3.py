#Detecting the color of the traffic light with only the color intensity at the image
import cv2
import numpy as np

def detect_traffic_light_color(frame):
    # Convert frame to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Define color ranges in RGB
    red_lower = np.array([120, 0, 0])
    red_upper = np.array([255, 80, 80])
    green_lower = np.array([0, 120, 0])
    green_upper = np.array([80, 255, 80])
    yellow_lower = np.array([120, 120, 0])
    yellow_upper = np.array([255, 255, 80])

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
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect the color of the traffic light
        traffic_light_color = detect_traffic_light_color(frame)
        print(f"Detected Traffic Light Color: {traffic_light_color}")

        # Display the frame
        cv2.imshow('Traffic Light Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
