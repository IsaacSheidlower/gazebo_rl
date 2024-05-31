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

class Camera:
    # Make width and height static parameters
    width = 480
    height = 270
    def __init__(self, width=None, height=None, savepath=None):
        self.bridge = CvBridge()
        self.image = None
        self.width = Camera.width if width is None else width
        self.height = Camera.height if height is None else height
        self.frame_num = 0

        if savepath is not None:
            savepath = Path(savepath).expanduser()
            if savepath.parent.exists():
                import shutil
                # delete the old files
                print(f"Deleting old files in {savepath.parent}")
                shutil.rmtree(savepath.parent)
            print(f"Creating directory {savepath}")
            savepath.mkdir(parents=True)
        self.savepath = savepath

    def start(self, dev_id=0):
        self.c = cv2.VideoCapture(dev_id)
        self.c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.c.isOpened():
            print("Error: Could not open camera.")
            return False
        else:
            self.c.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            rospy.loginfo(f"Camera initialized with res {str(self.width)} x : {str(self.height)}")
            return True
        
    def read(self, get_msg=True, show=False):
        ret, frame = self.c.read()

        if show:
            cv2.imshow("frame", frame)
            key = cv2.waitKey(10)
            if key == 27:
                cv2.destroyAllWindows()
                self.c.release()
                print("Camera released")
                exit(0)


        if self.savepath is not None:
            # check if the savepath exists
            cv2.imwrite(str(self.savepath / f"{self.frame_num:04d}.png"), frame)

        self.frame_num += 1
        if get_msg:
            # return ret, self.bridge.cv2_to_imgmsg(frame, encoding="passthrough")
            return ret, self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        else:
            return ret, frame
        
    def close(self):
        self.c.release()
        cv2.destroyAllWindows()
        print("Camera released")
        

#TODO ActionMap = {
#     0: [0, 0, 0, 0, 0, 0, 0],
#     1: [0, 0, 0, 0, 0, 0, 1],
#     2: [0, 0, 0, 0, 0, 0, -1],
import threading
current_observation = np.zeros(5)    
lock = threading.Lock()
def eef_pose(data):
    # NOTE: ioda has many commented lines that should be referenced when adding state
    # TODO: this should just be a pose message.
    # augmented with velocity:
    x_pose = data.base.tool_pose_x 
    y_pose = data.base.tool_pose_y 
    z_pose = data.base.tool_pose_z

    # normalize the post based on the bounds. 0 mean
    # x_pose = (x_pose - xzero) / xrange
    # y_pose = (y_pose - yzero) / yrange
    # z_pose = (z_pose - zzero) / zrange

    # y_twist = np.deg2rad(data.base.tool_pose_theta_y)
    current_observation[0] = x_pose
    current_observation[1] = y_pose
    current_observation[2] = z_pose
    try:
        current_observation[3] = data.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
    except:
        current_observation[3] = 0.0
        print("ERROR: No gripper feedback received.")

    current_observation[4] = 0.0 # REWARD

def sync_copy():
    with lock:
        return current_observation.copy()
    
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

        # bring observation node directly into the environment to keep us as close to live operation as possible.
        self.executed_episodes = 0
        self.camera = Camera(savepath=None) # A simulated self.camera should get an image published from gazebo.
        self.camera.start(); print(f"Camera started.")
        rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, callback=eef_pose)

        self.tracked_weights = []; self.alpha = 0.1; self.TARGET_IDXS = [1,2,3] # which scales
        self.WAITING_FOR_RESET = 0; self.WFR = 15

    def check_for_reward(self):
        give_reward = False
        if self.WAITING_FOR_RESET > 0:
            # publish the reward
            self.WAITING_FOR_RESET -= 1
            print(f"\t\t{self.WAITING_FOR_RESET} {datetime.datetime.now().strftime('%H:%M:%S')}")
            give_reward = True
        else:
            # weights = rospy.wait_for_message("/tracked_weights", Int32MultiArray, timeout=2)
                
            weights = rospy.wait_for_message("/weights", Float32MultiArray, timeout=2)
            if not self.tracked_weights: self.tracked_weights = [collections.deque(maxlen=25) for w in weights.data]
            for idx, w in enumerate(weights.data):
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
                # img_msg = rospy.wait_for_message(self.observation_topic, Image, timeout=5)
                # state_msg = rospy.wait_for_message("rl_observation", ObsMessage, timeout=5)
                ret, img_np = self.camera.read(show=True, get_msg=False)
                state = sync_copy()
            except Exception as e:
                print("No image received. Sending out blank observation.", e)
                # return self._get_obs(is_first=is_first) #oof ugly
                return {
                    "image": np.zeros(self.input_size, dtype=np.uint8),
                    "is_first": is_first,
                    "is_last": False, # never ends
                    "is_terminal": False, # never ends
                    "state": np.array([0, 0, 0, 0, 0]), # x, y, z, gripper_state, reward
                    "rel_eef": np.array([0, 0, 0, 0])  # normalized x, y, z, unnormalized gripper_state
                }
            
            # img_np = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
            
            # reshape to the config size
            # NOTE: this is duplicated inthe observation node. Must unify.
            width, height = img_np.shape[1], img_np.shape[0]
            if width != height: # non square
                if width < height:
                    raise ValueError("Image is taller than it is wide. This is not supported.")
                else: # images are wider than tall
                    # crop the square image from the center, with an additional offset
                    # offset = rospy.get_param("/image_offset", 0)
                    offset = 50
                    bounds = (width//2 - offset) -  height//2, (width//2 - offset) + height//2
                    img_np = img_np[:, bounds[0]:bounds[1]]
            
            resized_img = cv2.resize(img_np, self.input_size[:2])
            # flip the image horizontally
            resized_img = cv2.flip(resized_img, 1)

            if GRAYSCALE:=True:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                resized_img = np.expand_dims(resized_img, axis=-1)
        
            cv2.imshow("image", resized_img)
            cv2.waitKey(1)

            # normalize the post based on the bounds. 0 mean
            # x_pose, y_pose, z_pose, gripper_state = state_msg.obs[:4]
            x_pose, y_pose, z_pose, gripper_state = state[:4]
            x_pose = (x_pose - xzero) / xrange
            y_pose = (y_pose - yzero) / yrange
            z_pose = (z_pose - zzero) / zrange
            gripper_state = gripper_state / 50 - 1

            self.gripper_state = gripper_state

            return {
                "image": resized_img,
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
    
        