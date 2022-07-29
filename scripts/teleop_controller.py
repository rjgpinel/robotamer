import functools
import gym
import os
import pickle
import rospy
import time
import robotamer.envs

import numpy as np
from math import pi
from datetime import datetime

from robotamer.envs.pick import PickEnv
from sensor_msgs.msg import Joy


PUSHING_START_CONFIG = [
        # 0.9326981650333579, -1.752163298993259, 1.7692008154315744, -1.070960116650423, 2.19026060548725, 2.3614391975469964
        0.9326981650333579, -1.752163298993259, 1.7692008154315744, -1.070960116650423, 2.19026060548725, 0
]


class Dataset:

    def __init__(self, path):
        self.path = path
        self.episodes = []
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

    def reset(self, obs):
        # if self.episodes:
        #     for k in self.episodes[-1]:
        #         self.episodes[-1][k] = np.array(self.episodes[-1][k])
        print('Starting episode', len(self.episodes) + 1)
        self.episodes.append({'observations': [obs], 'actions': []})

    def append(self, act, next_obs):
        self.episodes[-1]['actions'].append(act)
        self.episodes[-1]['observations'].append(next_obs)

    def discard_episode(self):
        if self.episodes:
            self.episodes = self.episodes[:-1]
            print('Discarded episode', len(self.episodes) + 1)
        else:
            print('No episodes to discard')

    def save(self):
        episodes = self.episodes
        # if episodes and not episodes[-1]['actions']:
        #     # Leave out last empty episode.
        #     episodes = episodes[:-1]
        with open(self.path, 'wb') as f:
            pickle.dump(episodes, f)


def callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    joy_left = data.axes[0]
    joy_up = data.axes[1]
    done = data.buttons[2]
    discard = data.buttons[1]
    vx = y_scale * joy_up
    vy = x_scale * joy_left
    action = {
        "linear_velocity": np.array([vx, vy, 0.0]),
        "angular_velocity": np.array([0.0, 0.0, 0.0]),
        "grip_open": 1,
    }
    action_2d = [vx, vy]
    print('Sending', action)
    if done:
        dataset.save()
        print('Finished episode; Resetting arm')
        obs = reset_env(env)
        dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    elif discard:
        dataset.discard_episode()
        print('Resetting arm')
        obs = reset_env(env)
        dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
        real_obs = env.step(action)
        dataset.append(action_2d, real_obs)

def reset_arm(env):
    if env.arm_name == 'left':
        gripper_pos = [-0.40, 0, 0.1]
        gripper_orn = [pi, 0, pi / 2]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
    obs = env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)
    return obs


def reset_joints(env):
    obs = env.reset(joints=PUSHING_START_CONFIG)
    return obs


def reset_to_home(env):
    obs = env.reset(home_only=True)
    return obs


def reset_env(env):
   obs = reset_joints(env)
   return obs


def main():
    try:
        pick_env = gym.make("RealRobot-Pick-v0", cam_list=[], arm='right')
        real_obs = reset_env(pick_env)
        # real_obs = reset_arm(pick_env)
        # real_obs = reset_joints(pick_env)
        # print('pose before:', pick_env.robot.eef_pose())
        # for _ in range(4):
        #     action = {
        #         'linear_velocity': np.array([1, 0, 0]),
        #         'angular_velocity': np.array([0, 0, 0]),
        #         'grip_open': 1
        #     }
        #     pick_env.step(action)
        # print('pose after:', pick_env.robot.eef_pose())

        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        dataset_path = os.path.join(os.environ['TOP_DATA_DIR'],
                                    f'rrlfd/pushing_demos_sim_dev_{timestamp}.pkl')
        dataset = Dataset(dataset_path)
        dataset.reset(real_obs)
        env_step_callback = functools.partial(
            callback, env=pick_env, dataset=dataset, x_scale=0.05, y_scale=0.05)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')

        rospy.spin()
    except rospy.ROSInterruptException:
        dataset.save()


if __name__ == "__main__":
    main()


