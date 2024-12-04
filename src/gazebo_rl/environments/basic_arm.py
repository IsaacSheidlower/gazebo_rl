#!/usr/bin/env python3
'''
This implements a wrapper for the arm that takes actions an controls the arm, but does not implement a gym or reinforcement learning environment. It's intended for use with lerobot, but generally gives a little extra use-ability on top of armpy.
'''

import copy
import numpy as np
import rospy 
import time
import armpy
from kortex_driver.srv import *
from kortex_driver.msg import *
from collections import defaultdict, deque
import cv2

zero = lambda x: (x[0] + x[1]) / 2
nrange = lambda x: x[1] - x[0]
ybounds = -0.5, 0.5; yzero = zero(ybounds); yrange = nrange(ybounds)
xbounds = 0.0, 1.0; xzero = zero(xbounds); xrange = nrange(xbounds)
zbounds = 0.0, 0.5; zzero = zero(zbounds); zrange = nrange(zbounds)
z_flower_thresh = 0.365 # for flowerpot
y_flower_thresh = -0.04 # for flowerpot
x_flower_thresh = 0.52 # for flowerpot

import logging
import threading

class BasicArm():
    def __init__(self, max_action=.1, min_action=-.1, n_actions=2, input_size=4, action_duration=.5, reset_pose=None, velocity_control=False,
        stack_size=4, home_arm=True, max_vel=.3, cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, discrete_actions=False, robot_name='gen3'):
        
        """
            Generic point reaching class for the Gen3 robot.
            Args:
                max_action (float): maximum action value
                min_action (float): minimum action value
                n_actions (int): number of actions
                action_duration (float): duration of each action
                reset_pose (list): list of floats for the reset pose
                home_arm (bool): whether to home the arm at the beginning of each episode
                max_vel (float): maximum velocity
                cartesian_control (bool): whether to use cartesian control
                relative_commands (bool): whether to use relative commands or absolute commands
                sim (bool): whether to use the simulation
                workspace_limits (list): list of floats for the workspace limits (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        super().__init__()
        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.reset_pose = reset_pose
        self.home_arm = home_arm
        self.max_vel = max_vel
        #self.action_timout = action_timout
        self.cartesian_control = cartesian_control
        self.relative_commands = relative_commands
        self.sim = sim
        self.velocity_control = velocity_control

        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing arm", end='...')
        self.arm = armpy.initialize(robot_name.replace('my_', ''))
        self.logger.info("Initialized arm")
        
        if workspace_limits is None:
            self.workspace_limits = [*xbounds, *ybounds, *zbounds]
        else:
            self.workspace_limits = workspace_limits

        rospy.Subscriber(f"/{robot_name}/base_feedback", BaseCyclic_Feedback, self._base_feedback_callback)
        self.SAFETY_MODE = False
        self.safety_histories = {
            "x_tool_torque": deque(maxlen=10),
            "joint_1_torque": deque(maxlen=10), # the first bend, pressing down relieves the torque here.
        }
        self.x_tool_thresh, self.joint_1_thresh = 2.0, -0.01

        self.gripper_state = None
        self.current_step = 0
        self.prev_eef = None; self._eef = None
        self._eef_lock = threading.Lock()
        self._eef_time = time.time()
        self.LITE = 'lite' in robot_name
        if self.LITE: self.logger.info(f"Using gen3_lite")
        else: self.logger.info(f"Using gen3")

    def _base_feedback_callback(self, msg: BaseCyclic_Feedback):
        '''
        NOTE: This is not a generic safety check, and WILL NOT PREVENT MOST COLLISIONS.
        We're looking at specific safety check to prevent moving down in the z-direction when we are experiencing increased torques from vertical collision.
        '''
        self.safety_histories['x_tool_torque'].append(msg.base.tool_external_wrench_torque_x)
        self.safety_histories['joint_1_torque'].append(msg.actuators[1].torque)
        if len(self.safety_histories['x_tool_torque']) == 10:
            xtool_mean = np.mean(self.safety_histories['x_tool_torque']); joint1_mean = np.mean(self.safety_histories['joint_1_torque'])
            if (xtool_mean > self.x_tool_thresh) or (joint1_mean <= self.joint_1_thresh):
                self.SAFETY_MODE = True
                self.logger.info(f"SAFETY MODE ENGAGED: {np.mean(self.safety_histories['x_tool_torque'])} {np.mean(self.safety_histories['joint_1_torque'])}")
            else:
                self.SAFETY_MODE = False

        try:
            gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        except:
            self.logger.info("ERROR: no gripper feedback received")
            gripper_pos = 0.0

        with self._eef_lock:
            self._eef = np.array([msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, gripper_pos])
            dt = time.time() - self._eef_time
            if dt > 5: self.logger.info(f"WARN: EEF time: {dt} seconds.")
            self._eef_time = time.time()

    def sync_copy_eef(self):
        with self._eef_lock:
            return self._eef.copy()
    
    def stop_motion(self):
        self.logger.info("Stopping motion")
        self.arm.stop_arm()
        rospy.sleep(0.25)
        self.arm.cartesian_velocity_command([0 for _ in range(7)], duration=self.action_duration, radians=True)
        rospy.sleep(0.25)
        self.logger.info("\tHopefully stopped motion")

    def reset(self):
        self.current_step = 0
        if not self.sim:
            self.arm.stop_arm()
            self.logger.info(f"RESET {'- sim' if self.sim else ''}")
            while rospy.get_param("/pause", False):
                self.logger.info("Waiting for pause to be lifted before resetting.")
                rospy.sleep(1)
            self.arm.clear_faults()
            rospy.sleep(.25)
            self.arm.open_gripper()
            rospy.sleep(0.5)
        if self.reset_pose is None:
            self.arm.home_arm()
        else:
            if self.sim:
                self.arm.goto_joint_pose_sim(self.reset_pose)
            else:
                # first go up to avoid collisions
                self.arm.goto_cartesian_pose_old([-0.1, -0.15, 0.05, 0, 0, 0], relative=True, radians=True)
                rospy.sleep(0.5)
                self.arm.goto_joint_pose(self.reset_pose)
            rospy.sleep(1)
        self.prev_eef = self.sync_copy_eef()
        return self.prev_eef

    def step(self, action, orientation_speed=None, translation_speed=None, clip_wrist_action=False):
        if self.prev_eef is None:
            self.logger.info(f"Waiting for initial arm state message.")
            return

        self.current_step += 1

        ## GRIPPER
        if len(action) == 7 and action[6] != 0:
            if self.SAFETY_MODE:
                self.logger.info("Safety mode engaged. Gripper action ignored.")
                pass
            else:
                if self.velocity_control: # send a 0 vel to prevent the arm from moving while grippering
                    self.arm.stop_arm()

                gripper_open = True if (self.gripper_state and self.gripper_state < 0.5) else False # NOTE: the initial motor position of the gripper seems to be variable based on unknown factors.

                if action[6] == 1 and not gripper_open:
                    self.arm.open_gripper()
                elif action[6] == -1 and gripper_open: # prevent gripper faults
                    self.arm.close_gripper()
                self.logger.info(f"{self.current_step:4d} gripper action {action[6]} with state {self.gripper_state:1.2f}")
        else:
            # from IPython import embed as ipshell; ipshell()
            if clip_wrist_action:
                action = np.clip(np.array(action), self.min_action, self.max_action)
            else:
                original_action = copy.copy(action)
                action = np.clip(np.array(action), self.min_action, self.max_action)
                action = [action[0], action[1], action[2], 0, 0, 0]

            if self.velocity_control:
                buffered_move_xyz = [1.5 * (a * self.action_duration) for a in action[:3]]
                prev_xyz = self.prev_eef[:3]
                expected_new_position = newx, newy, newz = [prev_p + dp for prev_p, dp in zip(prev_xyz, buffered_move_xyz)]
            else: expected_new_position = newx, newy, newz = self.prev_eef[:3] + action[:3] # Do not allow an action to take us beyond the workspace limits

            prev_state_str = f"{self.prev_eef[0]:+1.2f} {self.prev_eef[1]:+1.2f} {self.prev_eef[2]:+1.2f}"
            pred_state_str = f"{newx:+1.2f} {newy:+1.2f} {newz:+1.2f}"
            self.logger.info(f"{self.current_step:4d} dp: {prev_state_str} -> {pred_state_str} from action {action[:3]}")

            # NOTE: UGLY HARDCODINGS FOR FLOWERBED TASK
            if (newx < self.workspace_limits[0] and action[0] < 0) or (newx > self.workspace_limits[1] and action[0] > 0): action[0] = 0 # self.logger.info("x out of bounds. stopping.")
            if (newy < self.workspace_limits[2] and action[1] < 0) or (newy > self.workspace_limits[3] and action[1] > 0): action[1] = 0 # self.logger.info("y out of bounds. stopping.")
            
            ### SAFETY CHECK
            if self.SAFETY_MODE:
                # Only allow the arm to move up in the z-direction
                if action[2] <= 0:
                    action = [0, 0, 0, 0, 0, 0, 0]
                    self.logger.info("Safety mode engaged. Stopping non +z-movement.")
            ### SAFETY CHECK

            if self.sim:
                if self.cartesian_control:
                    if not self.relative_commands:
                        self.arm.goto_cartesian_pose_sim(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                    else:
                        self.arm.goto_cartesian_relative_sim(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                        self.arm.stop_arm()
                else:
                    if self.relative_commands:
                        self.arm.goto_joint_pose_sim(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                    else:
                        self.arm.goto_joint_pose(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                        self.arm.stop_arm()
            else:
                if self.cartesian_control:
                    if not self.relative_commands: # NOTE: wtf?
                        self.arm.goto_cartesian_pose_sim(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                    else:
                        if self.velocity_control:
                            self.arm.cartesian_velocity_command(action, duration=self.action_duration, radians=True)
                        else:
                            # self.logger.info("goto_cartesian_pose_old")
                            self.arm.goto_cartesian_pose_old(action, relative=True, radians=True, 
                                                            translation_speed=translation_speed, orientation_speed=orientation_speed)
                            rospy.sleep(self.action_duration)
                            # self.arm.stop_arm()
                else:
                    if self.relative_commands:
                        self.arm.goto_joint_pose_sim(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                    else:
                        self.arm.goto_joint_pose(action, speed=self.max_vel)
                        rospy.sleep(self.action_duration)
                        self.arm.stop_arm()
            
        self.prev_eef = self.sync_copy_eef()
        return True
    
    def close(self):
        self.arm.stop_arm()
        self.arm.home_arm()
        rospy.sleep(.5)

if __name__ == '__main__':
    try:
        rospy.init_node("arm_reacher")
        rospy.sleep(1.0)
        robot_name = rospy.get_param('robot_name', ',my_gen3')
        sim = rospy.get_param('sim', False)
        arm = BasicArm(robot_name=robot_name, velocity_control=True, sim=sim)
        arm.reset()
        action = [0 for _ in range(7)]
        
        vx = 0.1
        for i in range(20):
            if i % 4 == 0: vx *= -1

            action[0] = vx

            arm.step(action)
            
            rospy.sleep(0.1)
            self.logger.info(i)
        arm.close()
    except rospy.ROSInterruptException as E:
        self.logger.info(E)


        