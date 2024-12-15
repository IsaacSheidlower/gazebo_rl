#!/home/j/workspace/lerobot/venv/bin/python
import rospy
import rosbag
import os, sys, resource
from pathlib import Path
import time
import cv2
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image, Joy
from collections import deque
from armpy import kortex_arm
from armpy import kortex_arm_sketchy
import std_msgs.msg



    

class BagVideoPublisher():
    def __init__(self, path):
        # dirname is also the name of the camera topic live
        rospy.init_node('bagvideopublisher', anonymous=True)
        self.arm = kortex_arm_sketchy.Arm()

        # arm = armpy.initialize('gen3_lite')

        bag = rosbag.Bag(str(path / 'trial_data.bag'))
# read joint states from bag
        joint_states = []
        self.all_joint_states = []
        for topic, msg, t in bag.read_messages(topics=['/my_gen3_lite/joint_states', '/joy']):

            if 'joy' in topic and sum(msg.buttons[4:8]) > 0:
                    # none active, skip it
                if joint_states:
                    # Save the joint states before the gripper command
                    self.all_joint_states.append(('joints', joint_states))
                    joint_states = []
                    
                if msg.buttons[4] == 1 or msg.buttons[6] == 1: 
                    self.all_joint_states.append(('gripper', 0.1))
                elif msg.buttons[5] == 1 or msg.buttons[7] == 1:
                    self.all_joint_states.append(('gripper', -1.0))


            elif 'joint_states' in topic:
                joint_states.append(msg.position if args.sketchy else args.position[:6]) # NOTE: for non-sketchy
                # joint_states.append(msg.position) # NOTE: for sketchy

        if joint_states: self.all_joint_states.append(('joints', joint_states))

        print(f"Replay Sequence: ")
        for type, entry in self.all_joint_states:
            if type == 'joints':
                print(f"Joints: {len(entry)}")
            elif type == 'gripper':
                print(f"Gripper: {entry}")
    
    def replay(self):
        # self.arm.home_arm()
        self.arm.open_gripper(); rospy.sleep(1)
        for type, entry in self.all_joint_states:
            if type == 'joints':
                waypoints = []
                # # waypoints.append(entry.position[:6])
                for i in range(0, len(entry), 10): # subsample
                    waypoints.append(entry[i])
                # # waypoints.append(entry.position)

                if args.sketchy:
                    self.arm.goto_joint_gripper_waypoints(waypoints, block=True) # NOTE: for sketchy
                else:
                    self.arm.goto_joint_waypoints(waypoints, block=True) # NOTE: for non-sketchy
            elif type == 'gripper':
                self.arm.send_gripper_command(entry, mode = 'speed', duration = 200, relative=True, block=False)
                rospy.sleep(0.2)
            #print([f'{entry:.2f}' for entry in  waypoints[-1]])
        # self.arm.goto_joint_gripper_waypoints(waypoints)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d0', '--dir0', type=str, required=True)
    # parser.add_argument('-d1', '--dir1', type=str, required=True)
    parser.add_argument('-p', '--root', type=str, default='~/')
    parser.add_argument('-dir', '--directory', type=str, required=True)
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--sketchy', action='store_true')
    args = parser.parse_args()

    path = Path(args.root).expanduser() / args.directory

    print(f"Playing back from {path}")
    replayer = BagVideoPublisher(path)
    if args.test:
        print(f"Test mode")
    else:
        replayer.replay()
    
