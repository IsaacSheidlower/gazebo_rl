#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Bool
import time
from gen3_testing.gen3_movement_utils import Arm
from gazebo_rl.msg import ObsMessage
from kortex_driver.srv import *
from kortex_driver.msg import *
from gazebo_rl.environments.arm_reaching_armpy import ArmReacher, x_flower_thresh, z_flower_thresh
import gymnasium as gym


#TODO ActionMap = {
#     0: [0, 0, 0, 0, 0, 0, 0],
#     1: [0, 0, 0, 0, 0, 0, 1],
#     2: [0, 0, 0, 0, 0, 0, -1],

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

        self.executed_episodes = 0

    def get_obs(self):
        # append the goal pose to the observation
        obs = super()._get_obs()
        return obs
    
    def reset(self):
        self.executed_episodes += 1

        if self.executed_episodes % rospy.get_param('/episode_interval', 5) == 0:
            print(f"PAUSING FOR ENVIRONMENT CLEANUP")
            rospy.set_param("/pause", True)
        return super().reset()
    
    def step(self, action):
        return super().step(action, velocity_control=self.velocity_control)

    def get_reward(self, observation):
        # NOTE: temporary, arbitrary, goal state
        # goal_state = np.array([0.6, -0.2]) # sim arm, upper right corner 2D
        # goal_state = np.array([0.5, 0.15]) # real arm, lower left corner 2D
        # current_state = observation["state"][:2]
        # distance = np.linalg.norm(current_state - goal_state)
        # if distance < 0.05:
        #     return 1, True
        # else:
        #     return 0, False

        # using the presence of red color as the goal state
        state = observation["state"]
        reward = state[-1]

        # NOTE: since we keep getting false alarms, we're going to gate reward on a z value. A cup cannot possibly be in the basket if the tool z is not above the basket.
        flower_thresh = (state[0] > x_flower_thresh) and (state[2] > z_flower_thresh)

        if reward == 1 and flower_thresh: 
            done = True; self.reward_received_pub.publish(Bool(True));
        else: 
            done = False
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
    
        