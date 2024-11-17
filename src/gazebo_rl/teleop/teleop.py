#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np

# Import the UIInterface and the specific UI class you want to use
from ui_interface import UIInterface
from mouse_keyboard_expert import MouseKeyboardExpert

class RobotControlNode:
    def __init__(self, ui: UIInterface):
        self.ui = ui
        self.pub = rospy.Publisher('robot_control', Twist, queue_size=10)
        rospy.init_node('robot_control_node', anonymous=True)
        self.rate = rospy.Rate(10)  # 10 Hz

    def run(self):
        while not rospy.is_shutdown():
            action, buttons = self.ui.get_action()
            # Process the action and buttons as needed
            twist_msg = self.process_action(action, buttons)
            # Publish the Twist message to control the robot
            self.pub.publish(twist_msg)
            self.rate.sleep()

    def process_action(self, action: np.ndarray, buttons: dict) -> Twist:
        # Convert action to Twist message
        twist = Twist()
        # Map the action to linear and angular velocities
        # Adjust scaling factors as necessary
        twist.linear.x = action[0] * 0.01
        twist.linear.y = action[1] * 0.01
        twist.linear.z = action[2] * 0.01
        twist.angular.x = action[3] * 0.01
        twist.angular.y = action[4] * 0.01
        twist.angular.z = action[5] * 0.01
        return twist

def main():
    # Instantiate the UI object (can be swapped with other UI implementations)
    ui = MouseKeyboardExpert()
    robot_control_node = RobotControlNode(ui)
    try:
        robot_control_node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
