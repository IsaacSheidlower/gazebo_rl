#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Bool
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Image
from gazebo_rl.environments.arm_reaching_armpy import ArmReacher, y_flower_thresh, z_flower_thresh, xrange, yrange, zrange, xzero, yzero, zzero
import gymnasium as gym
import cv2

from cv_bridge import CvBridge
from pathlib import Path
import collections, datetime, itertools
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool

#TODO ActionMap = {
#     0: [0, 0, 0, 0, 0, 0, 0],
#     1: [0, 0, 0, 0, 0, 0, 1],
#     2: [0, 0, 0, 0, 0, 0, -1],
import threading
current_observation = np.zeros(5)    
eef_lock = threading.Lock()
eef_time = time.time()
def eef_pose(data):
    global current_observation, eef_time
    # NOTE: ioda has many commented lines that should be referenced when adding state
    # TODO: this should just be a pose message.
    # augmented with velocity:
    x_pose = data.base.tool_pose_x 
    y_pose = data.base.tool_pose_y 
    z_pose = data.base.tool_pose_z

    with eef_lock:
        current_observation[0] = x_pose
        current_observation[1] = y_pose
        current_observation[2] = z_pose
        try:
            current_observation[3] = data.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        except:
            current_observation[3] = 0.0
            print("ERROR: No gripper feedback received.")
        current_observation[4] = 0.0 # REWARD
        
        dt = time.time() - eef_time
        if dt > 5: print(f"WARN: EEF time: {dt} seconds.")
        eef_time = time.time()

image_lock = threading.Lock()
current_image = np.zeros((128, 128, 1), dtype=np.uint8)
image_time = time.time()
def img_cb(data):
    with image_lock:
        global current_image, image_time
        current_image = CvBridge().imgmsg_to_cv2(data, "8UC1")
        # add the grayscale channel
        current_image = np.expand_dims(current_image, axis=-1)
        dt = time.time() - image_time
        if dt > 5: print(f"WARN: image time: {dt} seconds.")
        image_time = time.time()

side_image_lock = threading.Lock()
current_side_image = np.zeros((64, 64), dtype=np.uint8)
side_image_time = time.time()
def side_img_cb(data):
    with side_image_lock:
        global current_side_image, side_image_time
        current_side_image = CvBridge().imgmsg_to_cv2(data, "8UC1")
        # add the grayscale channel
        current_side_image = np.expand_dims(current_side_image, axis=-1)
        dt = time.time() - side_image_time
        if dt > 5: print(f"WARN: side image time: {dt} seconds.")
        side_image_time = time.time()


weights_lock = threading.Lock()
current_weights = np.zeros(5)
weights_time = time.time()
def weights_cb(data):
    with weights_lock:
        global current_weights, weights_time
        current_weights = data.data
        dt = time.time() - weights_time
        if dt > 5: print(f"WARN: weights time: {dt} seconds.")
        weights_time = time.time()


def sync_copy_eef():
    with eef_lock:
        return current_observation.copy()
    
def sync_copy_image():
    global current_image
    with image_lock:
        img_np = current_image.copy()
    return img_np

def sync_copy_side_image():
    global current_side_image
    with side_image_lock:
        img_np = current_side_image.copy()
    return img_np
    
def sync_copy_weights():
    with weights_lock:
        return list(current_weights)
    
class ArmReacher3D(ArmReacher):
    # inherits from ArmReacher
    def __init__(self, max_action=.1, min_action=-.1, n_actions=3, action_duration=.2, reset_pose=None, episode_time=60, 
        stack_size=4, sparse_rewards=False, success_threshold=.08, wrist_rotate_limit=.3,home_arm=True, with_pixels=False, max_vel=.3,
        velocity_control=False, cartesian_control=True, relative_commands=True, sim=True, workspace_limits=None, observation_topic="rl_observation",
        goal_dimensions=3, goal_pose=None, action_movement_threshold=.01,input_size=5, discrete_actions=False):
        
        if discrete_actions:
            n_actions = 13 # 8 discrete planar actions, 2 discrete vertical actions, 2 discrete gripper actions

        ArmReacher.__init__(self, max_action=max_action, min_action=min_action, n_actions=n_actions, input_size=input_size,
            action_duration=action_duration, reset_pose=reset_pose, episode_time=episode_time,
            stack_size=stack_size, sparse_rewards=sparse_rewards, success_threshold=success_threshold, home_arm=home_arm, with_pixels=with_pixels, max_vel=max_vel,
            cartesian_control=cartesian_control, relative_commands=relative_commands, sim=sim, workspace_limits=workspace_limits, observation_topic=observation_topic,
            goal_dimensions=goal_dimensions, discrete_actions=discrete_actions)
        
        if goal_pose is None:
            if workspace_limits is None:
                self.goal_pose = np.array([.5, .5, .5])
            else:
                self.goal_pose = [np.random.uniform(workspace_limits[0], workspace_limits[1]),
                                  np.random.uniform(workspace_limits[2], workspace_limits[3]),
                                  np.random.uniform(workspace_limits[4], workspace_limits[5])]
        else:
            self.goal_pose = goal_pose
        self.goal_pose = np.array(self.goal_pose)

        self.reward_received_pub = rospy.Publisher("/reset_signal", Bool, queue_size=1)
        self.velocity_control = velocity_control

        rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, callback=eef_pose)
        rospy.Subscriber("/rl_img_observation", Image, callback=img_cb)
        # rospy.Subscriber("/rl_side_img_observation", Image, callback=side_img_cb)
        rospy.Subscriber("/weights", Float32MultiArray, callback=weights_cb)

        self.tracked_weights = []; self.alpha = 0.1; self.TARGET_IDXS = [1,2,3] # which scales
        self.WAITING_FOR_RESET = 0; self.WFR = 15
        self.executed_episodes = 0
        self.debug = rospy.get_param("/debug", False)

    def check_for_reward(self):
        give_reward = False
        if self.WAITING_FOR_RESET > 0:
            # publish the reward
            self.WAITING_FOR_RESET -= 1
            print(f"\t\t{self.WAITING_FOR_RESET} {datetime.datetime.now().strftime('%H:%M:%S')}")
            give_reward = True
        else:
            weights = sync_copy_weights()
            if not self.tracked_weights: self.tracked_weights = [collections.deque(maxlen=25) for w in weights]
            for idx, w in enumerate(weights):
                self.tracked_weights[idx].append(w)

            if len(self.tracked_weights[0]) >= 25:
                THRESH = 1000; MAJOR_THRESH = 10000; short_idx = len(self.tracked_weights[0]) - 2
                long_means = [np.mean(list(itertools.islice(self.tracked_weights[tidx], 0, short_idx))) for tidx in self.TARGET_IDXS]
                short_means = [np.mean(list(itertools.islice(self.tracked_weights[tidx], short_idx, len(self.tracked_weights[tidx])))) for tidx in self.TARGET_IDXS]
                # ema_weights = [self.alpha * weights.data[tidx] + (1-self.alpha) * self.tracked_weights[tidx][-1] for tidx in self.TARGET_IDXS]
                # diff = [abs(ema_weight - mean) for ema_weight, mean in zip(ema_weights, long_means)]
                diff = [abs(sm - lm) for sm, lm in zip(short_means, long_means)]
                trigger = [dw > THRESH for dw in diff]
                major_trigger = False # any([dw > MAJOR_THRESH for dw in diff])
                if sum(trigger) >= 2: # two of three sensors agree
                    print(f"REWARD: {trigger} at {datetime.datetime.now().strftime('%H:%M:%S')} from {['{:.2f}'.format(d) for d in diff]}")
                    self.WAITING_FOR_RESET = self.WFR; print(f"\t Waiting for reset: {self.WAITING_FOR_RESET}")
                    self.tracked_weights = []
                    give_reward = True
                elif major_trigger:
                    print(f"REWARD from major trigger at {datetime.datetime.now().strftime('%H:%M:%S')} from {['{:.2f}'.format(d) for d in diff]}")
                    self.WAITING_FOR_RESET = self.WFR; print(f"\t Waiting for reset: {self.WAITING_FOR_RESET}")
                    self.tracked_weights = []
                    give_reward = True
                elif sum(trigger) > 0:
                    print(f"False positive from one weight. diffs: {diff} at {datetime.datetime.now().strftime('%H:%M:%S')}")
                    # self.tracked_weights = []


        return give_reward

    def _get_obs(self, is_first=False):
        if self.img_obs:
            try:
                img_np = sync_copy_image()
                # side_img = sync_copy_side_image()
                state = sync_copy_eef()
            except Exception as e:
                print("No image received. Sending out blank observation.", e)
                # return self._get_obs(is_first=is_first) #oof ugly
                return {
                    "image": np.zeros(self.input_size, dtype=np.uint8),
                    # "side_img": np.zeroes((64, 64, 1), dtype=np.uint8), # placeholder (should be 64x64)
                    "is_first": is_first,
                    "is_last": False, # never ends
                    "is_terminal": False, # never ends
                    "state": np.array([0, 0, 0, 0, 0]), # x, y, z, gripper_state, reward
                    "rel_eef": np.array([0, 0, 0, 0])  # normalized x, y, z, unnormalized gripper_state
                }
            
            # img_np = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
            
            # reshape to the config size
            # NOTE: this is duplicated inthe observation node. Must unify.
            resized_img = img_np
        
            cv2.imshow("env_image", resized_img)
            cv2.waitKey(1)

            # normalize the post based on the bounds. 0 mean
            # x_pose, y_pose, z_pose, gripper_state = state_msg.obs[:4]
            x_pose, y_pose, z_pose, gripper_state = state[:4]
            x_pose = (x_pose - xzero) / xrange
            y_pose = (y_pose - yzero) / yrange
            z_pose = (z_pose - zzero) / zrange
            gripper_state = gripper_state / 50 - 1 # normalize to range [-1, 1]

            self.gripper_state = gripper_state

            if self.debug: print(f"State: {[f'{s:.2f}' for s in state]}")

            return {
                "image": resized_img,
                # "side_image": side_img,
                "is_first": is_first,
                "is_last": False, # never ends
                "is_terminal": False, # never ends
                "state": np.array(state),
                # "state": np.array(state_msg.obs),
                "rel_eef": np.array((x_pose, y_pose, z_pose, gripper_state))
            }
        else:
            return np.array(rospy.wait_for_message(self.observation_topic, ObsMessage).obs)
    
    def reset(self):
        self.executed_episodes += 1

        if self.executed_episodes % rospy.get_param('/episode_interval', 5) == 0:
            print(f"PAUSING FOR ENVIRONMENT CLEANUP")
            rospy.set_param("/pause", True)
        return super().reset()
    
    def step(self, action):
        return super().step(action, velocity_control=self.velocity_control)

    def get_reward(self, observation):
        # using the presence of red color as the goal state
        done = False
        state = observation["state"]

        try:
            state[-1] = 1 if self.check_for_reward() else 0 # check the scales
            reward = state[-1]
        except Exception as e:
            print(f"Error from checking weights for reward: {e}")
            reward = 0

        # NOTE: since we keep getting false alarms, we're going to gate reward on a z value. A cup cannot possibly be in the basket if the tool z is not above the basket.
        flower_thresh = (state[0] > y_flower_thresh) and (state[2] > 0.35)

        if reward == 1:
            if not flower_thresh:
                print(f"False positive due to z value: {state[2]}")
                reward = 0
            else: 
                done = True; self.reward_received_pub.publish(Bool(True));
        
        return reward, done
    
    def _get_reward(self, observation, action=None):
        return self.get_reward(observation)
    
    def _map_discrete_actions(self, action):
        """
            Maps the discrete actions to continuous actions
        """
        
        if self.velocity_control:
            action_distance = ad = 0.25; had = ad # NOTE: half action on diagonal
            zad = 0.04
        else:
            action_distance = ad = 0.025; had = ad
            zad = 0.025


        gripper, dx, dy, dz = 0, 0, 0, 0
        if action == 0: pass # noop
        elif action == 1: dx, dy = 0, -ad
        elif action == 2: dx, dy = had, -had
        elif action == 3: dx, dy = ad, 0
        elif action == 4: dx, dy = had, had
        elif action == 5: dx, dy = 0, ad
        elif action == 6: dx, dy = -had, had
        elif action == 7: dx, dy = -ad, 0
        elif action == 8: dx, dy = -had, -had
        elif action == 9: dz = zad # up
        elif action == 10: dz = -zad # down    
        elif action == 11: gripper = 1
        elif action == 12: gripper = -1 
        else: dx, dy, dz, gripper = 0, 0, 0, 0
        return np.array([dx, dy, dz, 0, 0, 0, gripper])
        
    def get_action(self, action):
        if self.discrete_actions:
            return action
        else:    
            raise NotImplementedError("Continuous actions not implemented yet")
    
        