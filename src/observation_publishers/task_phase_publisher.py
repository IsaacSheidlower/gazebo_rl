#!/usr/bin/env python3

import numpy as np
import rospy
from kortex_driver.msg import *
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Int32

import cv2
from cv_bridge import CvBridge, CvBridgeError

# task phase:
# 0: the can is not grasped
# 1: the can is grasped
# 2: the can is grasped and the can is placed on the shelf
# 3: the can is not grasped and the can is placed on the shelf
# 4: the can is placed at the right position

# Global variables to store the gripper state and number of objects
gripper_state = "unknown"
num_objects = 0
target_reached = False

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
    global blue_detected, red_detected, num_objects, target_reached
    try:
        # Convert ROS Image message to OpenCV image in RGB format
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # Convert RGB image to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        # Define HSV range for blue color (unchanged)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        # Define HSV ranges for red color (unchanged)
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([175, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        # Threshold the HSV image for blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Threshold the HSV image for red colors
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Apply morphological operations to clean up the masks
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        # Check for blue detection
        blue_detected = np.any(mask_blue > 0)
        if blue_detected:
            print("blue detected")

        # Check for red detection
        red_detected = np.any(mask_red > 0)
        if red_detected:
            print("red detected")
            num_objects = 1
        else:
            num_objects = 0

        # Find and draw blue contours
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours_blue, -1, (0, 0, 255), 2)  # Drawing blue contours in RGB (red line)

        # Find and draw red contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image, contours_red, -1, (255, 0, 0), 2)  # Drawing red contours in RGB (blue line)

        # Determine if the red object is at the left bottom corner
        target_reached = False
        height, width, _ = cv_image.shape
        center_left = int(width / 3)
        center_right = int(2 * width / 3)
        center_top = int(height / 3)
        center_bottom = int(2 * height / 3)

        # Draw a rectangle representing the center zone
        cv2.rectangle(
            cv_image,
            (center_left, center_top),
            (center_right, center_bottom),
            (0, 255, 0),  # green rectangle in RGB
            2
)
        if len(contours_red) > 0:
            # Find the largest red contour by area
            largest_red_contour = max(contours_red, key=cv2.contourArea)
            M = cv2.moments(largest_red_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
        # Check if centroid (cx, cy) lies within the center boundaries
                if center_left < cx < center_right and center_top < cy < center_bottom:
                    target_reached = True
                    print("Target reached: Red object is at the center.")
                else:
                    target_reached = False
            else:
                target_reached = False

        # Create a named window and show the image
        cv2.namedWindow("Detected Colors", cv2.WINDOW_NORMAL)
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
    rospy.Subscriber("/camera_obs__dev_video0_96x96", Image, callback=Image_data)

    rate = rospy.Rate(40)  # 10 Hz

    while not rospy.is_shutdown():
        try:
            # Determine the task phase based on gripper state and number of objects
            phase = None
            # print("num of objs", num_objects)
            # print("gripper pos", gripper_state)
            if num_objects == 0 and gripper_state == "unknown":
                phase = -1
            elif num_objects == 0 and gripper_state == "open":
                phase = 0
            elif num_objects == 0 and gripper_state == "fully_closed":
                phase = 0
            elif num_objects == 0 and gripper_state == "semi_closed":
                phase = 1
            elif num_objects == 1 and gripper_state == "semi_closed":
                phase = 2
            elif num_objects == 1 and not target_reached:
                phase = 3
            elif target_reached:
                phase = 4
            else:
                phase = -1
            print("Task phase:", phase)
        except Exception as e:
            print(e)
        rate.sleep()

if __name__ == '__main__':
    try:
        print("Publishing observations")
        observation_publisher()
    except rospy.ROSInterruptException:
        pass
