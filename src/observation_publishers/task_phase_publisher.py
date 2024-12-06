#!/usr/bin/env python3

import numpy as np
import rospy
from kortex_driver.msg import *
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Int32

import cv2
from cv_bridge import CvBridge, CvBridgeError

# Global variables to store the gripper state and number of objects
gripper_state = "unknown"
num_objects = 0

# Thresholds for gripper positions (adjust these values based on your gripper's specifications)
open_threshold = 0.8
closed_threshold = 0.2

# Initialize the CvBridge class
bridge = CvBridge()

def gripper_pos(data):
    global gripper_state
    gripper_pos_value = data.position[-1]
    # print("gripper pos value", gripper_pos_value )
    if gripper_pos_value >= open_threshold:
        gripper_state = "open"
    elif gripper_pos_value <= closed_threshold:
        gripper_state = "fully_closed"
    else:
        gripper_state = "semi_closed"

def Image_data(data):
    global blue_detected, red_detected, num_objects
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        # Convert image to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define stricter range for blue color in HSV
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        # Define stricter range for red color in HSV
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Threshold the HSV image to get only red colors with stricter ranges
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Optional: Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        # Check if blue is detected
        blue_detected = np.any(mask_blue > 0)
        # Check if red is detected
        red_detected = np.any(mask_red > 0)

        if blue_detected and not red_detected:
            num_objects = 1
        elif red_detected and not blue_detected:
            num_objects = 1
        elif red_detected and blue_detected:
            num_objects = 2
        else:
            num_objects = 0

        # For visualization, create contours around the detected areas
        # Blue contours
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours_blue, -1, (255, 0, 0), 2)  # Blue contours in BGR

        # Red contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours_red, -1, (0, 0, 255), 2)  # Red contours in BGR

        # Display the image with contours
        cv2.imshow("Detected Colors", cv_image)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print(e)

def observation_publisher():
    rospy.init_node("observation_pub", anonymous=True)

    # Publisher to publish the task phase
    pub = rospy.Publisher("task_phase", Int32, queue_size=10)

    # Subscribers to get gripper position and image data
    rospy.Subscriber("/my_gen3_lite/base_feedback/joint_state", JointState, callback=gripper_pos)
    rospy.Subscriber("/camera_obs__dev_video0", Image, callback=Image_data)

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        try:
            # Determine the task phase based on gripper state and number of objects
            phase = None
            # print("num of objs", num_objects)
            # print("gripper pos", gripper_state)
            if num_objects == 1:
                if gripper_state == "open" or gripper_state == "fully_closed":
                    phase = 0
                elif gripper_state == "semi_closed":
                    phase = 1
            elif num_objects == 2:
                if gripper_state == "open" or gripper_state == "fully_closed":
                    phase = 3
                else:
                    phase = -2
            else:
                phase = -1  # Undefined phase

            if phase is not None:
                pub.publish(phase)
                print("Published phase:", phase)
        except Exception as e:
            print(e)
        rate.sleep()

if __name__ == '__main__':
    try:
        print("Publishing observations")
        observation_publisher()
    except rospy.ROSInterruptException:
        pass
