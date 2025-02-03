# PUZZLEBOT
This robot was part of an undergrad robotics class. The challenge was to achieve an autonomous driving of the differential robot using computer vision with control and AI, following a line through a circuit and having different traffic signals such as a semaphore, stop signals and crosswalks.

![Screenshot 2025-02-02 183655](https://github.com/user-attachments/assets/44949494-2700-4365-9c57-0ebd4aa047c0)

Hardware was provided by ManchesterRobotics, including the acrylic laser cutted plates, motors and the controller for the motors. A Nvidia jetson nano with a raspberry pi cam was used for the image processing and as the ROS host, this jetson nano was connected to the controller board by SPI, controller board used an esp32 microprocessor to run micro ros and control the motors.

The codes at this repository include the proof of concepts and algorithm developement in order to achieve the following features as well as the functional ROS2 scripts in python:
- Image filtering and resizing 
- Line following using PD controller
- Traffic signals detection and routines for them 

The folders from this repository are the following:

## LineFollower
This folder includes some footage used to develop the line detection algorithm, testing different morphological operations, filters and parameters for edge detection.
It uses Canny edge detection and Hough Lines to calculate the position of the line on camera, based on the given feedback from the live feed of the camera then it uses a linear regression moving average filter. Finally this number is compared to a set point established when the robot was aligned with the black line to follow and that is how error was calculated.
With this scripts it is possible to visualize the behavior of the line detection, which later needed some tunning because of the latency and refresh rate of the camera. 

## TrafficSignalsDetection
(For this feature the jetson nano was connected to a computer due to performance and latency issues)
The signals to detect are:
-

## ROS_Scripts

