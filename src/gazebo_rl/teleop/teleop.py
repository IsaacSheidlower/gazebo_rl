#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
import numpy as np

# Import the UIInterface and the specific UI class you want to use
from ui_interface import UIInterface
from mouse_keyboard_expert import MouseKeyboardExpert

class RobotControlNode:
    def __init__(self, ui: UIInterface):
        self.ui = ui
        self.pub = rospy.Publisher('joy', Joy, queue_size=10)
        rospy.init_node('robot_control_node', anonymous=True)
        self.rate = rospy.Rate(30)  # 10 Hz

    def run(self):
        while not rospy.is_shutdown():
            action, buttons = self.ui.get_action()
            # Process the action and buttons as needed
            joy_msg = self.process_action(action, buttons)
            # Publish the Joy message
            self.pub.publish(joy_msg)
            self.rate.sleep()

    def process_action(self, action: np.ndarray, buttons: dict) -> Joy:
        # Convert action and buttons to Joy message
        joy = Joy()
        # Map the action array to the axes field
        joy.axes = action.tolist()

        # Map the buttons dictionary to the buttons field
        # Assuming buttons is a dictionary with button names as keys and boolean values
        # Define the order of buttons to maintain consistency
        button_order = ['mouse_left', 'mouse_button9', 'mouse_button8']  # Adjust based on actual buttons
        joy.buttons = [int(buttons.get(button, 0)) for button in button_order]

        return joy

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
