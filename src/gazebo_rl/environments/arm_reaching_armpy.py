#!/usr/bin/env python3

import copy
import numpy as np
import rospy 
import time
import armpy
from gazebo_rl.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
import gymnasium as gym
from gymnasium.spaces import Box
from gym import spaces
from sensor_msgs.msg import Image
from collections import defaultdict, deque
import cv2

class ArmReacher(gym.Env):
    def __init__(self, max_action=.1, min_action=-.1, n_actions=2, input_size=4, action_duration=.5, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.1, home_arm=True, with_pixels=False, max_vel=.3, 
        cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="rl_observation",
        goal_dimensions=3, max_steps=200, discrete_actions=False):
        
        """
            Generic point reaching class for the Gen3 robot.
            Args:
                max_action (float): maximum action value
                min_action (float): minimum action value
                n_actions (int): number of actions
                action_duration (float): duration of each action
                reset_pose (list): list of floats for the reset pose
                episode_time (float): duration of each episode
                stack_size (int): size of the stack
                sparse_rewards (bool): whether to use sparse rewards
                success_threshold (float): threshold for success
                home_arm (bool): whether to home the arm at the beginning of each episode
                with_pixels (bool): whether to use pixels
                max_vel (float): maximum velocity
                cartesian_control (bool): whether to use cartesian control
                relative_commands (bool): whether to use relative commands or absolute commands
                sim (bool): whether to use the simulation
                workspace_limits (list): list of floats for the workspace limits (x_min, x_max, y_min, y_max, z_min, z_max)
                observation_topic (str): topic to subscribe to for observations
                goal_dimensions (int): number of dimensions for the goal (assumes goal position is the last n dimensions)
        """
        super().__init__()
        self.max_action = max_action
        self.min_action = min_action
        self.action_duration = action_duration
        self.n_actions = n_actions
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(n_actions)
        else:
            self.action_space = spaces.Box(low=min_action, high=max_action, shape=(n_actions,), dtype=np.float32)
        self.reset_pose = reset_pose
        self.episode_time = episode_time
        self.stack_size = stack_size
        self.sparse_rewards = sparse_rewards
        self.success_threshold = success_threshold
        self.home_arm = home_arm
        self.with_pixels = with_pixels
        self.max_vel = max_vel
        #self.action_timout = action_timout
        self.cartesian_control = cartesian_control
        self.relative_commands = relative_commands
        self.sim = sim
        self.prev_obs = None

        # observation space is the size of the observation topic
        self.observation_topic = observation_topic
        self.img_obs = "_img_" in observation_topic
        print(f"{self.observation_topic} {input_size}")
        self.input_size = input_size
        if self.img_obs:
            print(str(__class__), "Using image observations")
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(low=0, high=255, shape=input_size, dtype=np.uint8),
                'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'is_first': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_last': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_terminal': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
            })
        else:
            print(str(__class__), "Using float observations")
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(input_size,), dtype=np.float32)

        self.goal_dimensions = goal_dimensions
        self.max_steps = max_steps
        self.current_step = 0
        self.arm = armpy.initialize("gen3")

        if workspace_limits is None:
            self.workspace_limits = [.25, .7, -.25, .25, .025, .55]
        else:
            self.workspace_limits = workspace_limits

        self.SAFETY_MODE = False
        if not self.sim:
            rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self._base_feedback_callback)
            self.safety_histories = {
                "x_tool_torque": deque(maxlen=10),
                "joint_1_torque": deque(maxlen=10), # the first bend, pressing down relieves the torque here.
            }
            self.x_tool_thresh, self.joint_1_thresh = 2.0, 0

    def _base_feedback_callback(self, msg: BaseCyclic_Feedback):
        '''
        NOTE: This is not a generic safety check, and WILL NOT PREVENT MOST COLLISIONS.
        We're looking at specific safety check to prevent moving down in the z-direction when we are experiencing increased torques from vertical collision.
        '''
        self.safety_histories['x_tool_torque'].append(msg.base.tool_external_wrench_torque_x)
        self.safety_histories['joint_1_torque'].append(msg.actuators[1].torque)
        if len(self.safety_histories['x_tool_torque']) == 10:
            ytool_mean = np.mean(self.safety_histories['x_tool_torque']); joint1_mean = np.mean(self.safety_histories['joint_1_torque'])
            if (ytool_mean > self.x_tool_thresh) or (joint1_mean <= self.joint_1_thresh):
                self.SAFETY_MODE = True
                print(f"SAFETY MODE ENGAGED: {np.mean(self.safety_histories['x_tool_torque'])} {np.mean(self.safety_histories['joint_1_torque'])}")
            else:
                self.SAFETY_MODE = False

    def _get_obs(self, is_first=False):
        if self.img_obs:
            try:
                print("waiting for observation")
                img_msg = rospy.wait_for_message(self.observation_topic, Image, timeout=5)
                state_msg = rospy.wait_for_message("rl_observation", ObsMessage, timeout=5)
            except Exception as e:
                print("No image received. Sending out blank observation.", e)
                # return self._get_obs(is_first=is_first) #oof ugly
                return {
                    "image": np.zeros(self.input_size, dtype=np.uint8),
                    "is_first": is_first,
                    "is_last": False, # never ends
                    "is_terminal": False, # never ends
                    "state": np.array([0, 0, 0, 0])
                }
            
            img_np = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
            
            # reshape to the config size
            # NOTE: this is duplicated inthe observation node. Must unify.
            width, height = img_np.shape[1], img_np.shape[0]
            if width != height: # non square
                if width < height:
                    raise ValueError("Image is taller than it is wide. This is not supported.")
                else: # images are wider than tall
                    # crop the square image from the center, with an additional offset
                    offset = 20
                    bounds = (width//2 - offset) -  height//2, (width//2 - offset) + height//2
                    # print(f"img {img_np.shape}", bounds[0], bounds[1])
                    img_np = img_np[:, bounds[0]:bounds[1]]
                    # print(f"\timg {img_np.shape}")
                    # img_np = img_np[:, width//2 - height//2: width//2 + height//2]
            
            resized_img = cv2.resize(img_np, self.input_size[:2])
            # flip the image vertically
            resized_img = cv2.flip(resized_img, 0)
            # flip the image horizontally
            resized_img = cv2.flip(resized_img, 1)

            if GRAYSCALE:=True:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                resized_img = np.expand_dims(resized_img, axis=-1)
        
            cv2.imshow("image", resized_img)
            cv2.waitKey(1)

            return {
                "image": resized_img,
                "is_first": is_first,
                "is_last": False, # never ends
                "is_terminal": False, # never ends
                "state": np.array(state_msg.obs)
            }
        else:
            return np.array(rospy.wait_for_message(self.observation_topic, ObsMessage).obs)

    def reset(self):
        print(f"RESET {'- sim' if self.sim else ''}")
        if not self.sim:
            rospy.sleep(.1)
            self.arm.stop_arm()
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
                self.arm.goto_joint_pose(self.reset_pose)

        self.prev_obs = self._get_obs(is_first=True)
        return self.prev_obs

    # def get_reward(self, observation):
    #     """
    #         Returns the reward and whether the episode is done
    #     """
    #     raise NotImplementedError
    
    def _get_reward(self, observation, action):
        """
            Returns the reward and whether the episode is done
        """
        raise NotImplementedError
    
    def _get_action(self, action):
        """
            Applies any necessary transformations to the action 
        """
        raise NotImplementedError
    
    def _map_discrete_actions(self, action):
        """
            Maps the discrete actions to continuous actions
        """
        raise NotImplementedError

    def step(self, action, velocity_control=False, orientation_speed=None, translation_speed=None,
             clip_wrist_action=False):
        if self.discrete_actions: action = self._map_discrete_actions(action)
        self.current_step += 1

        ## GRIPPER
        if len(action) == 7 and action[6] != 0:
            if self.SAFETY_MODE:
                print("Safety mode engaged. Gripper action ignored.")
                pass
            else:
                if action[6] == 1:
                    self.arm.open_gripper()
                elif action[6] == -1:
                    self.arm.close_gripper()
                rospy.sleep(self.action_duration)
        else:
            if clip_wrist_action:
                action = np.clip(np.array(action), self.min_action, self.max_action)
            else:
                original_action = copy.copy(action)
                action = np.clip(np.array(action), self.min_action, self.max_action)
                action = [action[0], action[1], action[2], 0, 0, 0]
            # Do not allow an action to take us beyond the workspace limits
            expected_new_position = newx, newy, newz = self.prev_obs["state"][:3] + action[:3]
            prev_state_str = f"{self.prev_obs['state'][0]:1.2f} {self.prev_obs['state'][1]:1.2f} {self.prev_obs['state'][2]:1.2f}"
            pred_state_str = f"{newx:1.2f} {newy:1.2f} {newz:1.2f}"
            print(f"Current position: {prev_state_str} -> {pred_state_str}", end=" ")
            print(f"from action {action[:3]}", end=" ")

            # NOTE: UGLY HARDCODINGS FOR FLOWERBED TASK
            if newx < self.workspace_limits[0] or newx > self.workspace_limits[1]: action[0] = 0 # print("x out of bounds. stopping.")
            if newy < self.workspace_limits[2] or newy > self.workspace_limits[3]: action[1] = 0 # print("y out of bounds. stopping.")
            
            if newx < 0.54: # NOTE: Hardcoded horizontal position of flowerbed
                if newz < self.workspace_limits[4] or newz > self.workspace_limits[5]: action[2] = 0 # print("z out of bounds. stopping.") 
            else: # NOTE: protect the flow bed by introducing a higher z limit
                if newz < 0.37: 
                    action[0] = 0 # don't let us push into the flowerbed
                    action[2] = max(0, action[2]) 
                elif newz > self.workspace_limits[5]: action[2] = 0

            ### SAFETY CHECK
            if self.SAFETY_MODE:
                # Only allow the arm to move up in the z-direction
                if action[2] <= 0:
                    action = [0, 0, 0, 0, 0, 0, 0]
                    print("Safety mode engaged. Stopping non +z-movement.")
                
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
                        if velocity_control:
                            self.arm.cartesian_velocity_command(action, duration=self.action_duration, radians=True)
                        else:
                            print("goto_cartesian_pose_old")
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
        # check if we have reached the goal
        obs = self.prev_obs = self._get_obs()
        # if self.SAFETY_MODE and action[2] <= 0: # give negative reward when in safety mode and not going up.
        #     reward = -.01; done = False
        # else:
        reward, done = self._get_reward(obs, action)

        if self.current_step >= self.max_steps:
            done = True
            self.current_step = 0

        print(f"\tReward: {reward} Done: {done}")
        return obs, reward, done, {}
    
    def render(self):
        pass
    
    def close(self):
        self.arm.stop_arm()
        self.arm.home_arm()
        rospy.sleep(.5)

if __name__ == '__main__':
    try:
        rospy.init_node("arm_reacher")
        ArmReacher()
    except rospy.ROSInterruptException:
        pass

        