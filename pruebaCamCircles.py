import cv2
import numpy as np

def detect_circles(image, color_mask, color_name, detected_colors):
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(color_mask, 5)

    # Apply Canny edge detector
    edged = cv2.Canny(blurred, 75, 250)

    # Applying Hough Circle Transform
    circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        detected_colors[color_name] += len(circles[0, :])
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # Draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (255, 255, 255), 3)
    return image

def main():
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_colors = {'red': 0, 'yellow': 0, 'green': 0}

        # Define color ranges and apply mask
        # Red color  0 0 153  255 153 153
        lower_red = np.array([136, 87, 111]) 
        upper_red = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # Yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Green color
        lower_green = np.array([0, 100, 0])
        upper_green = np.array([150, 255, 178])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Detect circles for each color
        frame = detect_circles(frame, mask_red, 'red', detected_colors)
        frame = detect_circles(frame, mask_yellow, 'yellow', detected_colors)
        frame = detect_circles(frame, mask_green, 'green', detected_colors)

        # Determine which color has the most circles
        if max(detected_colors.values()) > 0:  # Ensure there's at least one circle detected
            traffic_state = max(detected_colors, key=detected_colors.get)
            print(f"Traffic Light is {traffic_state.upper()}")
        else:
            print("No traffic light detected.")

        # Display the resulting frame
        cv2.imshow('Traffic Light Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
