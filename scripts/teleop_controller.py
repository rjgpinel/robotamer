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


class Dataset:

    def __init__(self, path):
        self.path = path
        self.episodes = []
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

    def reset(self, obs):
        if self.episodes:
            for k in self.episodes[-1]:
                self.episodes[-1][k] = np.array(self.episodes[-1][k])
        self.episodes.append({'observations': [obs], 'actions': []})

    def append(self, act, next_obs):
        self.episodes[-1]['actions'].append(act)
        self.episodes[-1]['observations'].append(next_obs)

    def save(self):
        episodes = self.episodes
        if episodes and not episodes[-1]['actions']:
            # Leave out last empty episode.
            episodes = episodes[:-1]
        with open(self.path, 'wb') as f:
            pickle.dump(episodes, f)


def callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    joy_left = data.axes[0]
    joy_up = data.axes[1]
    done = data.buttons[2]
    vx = y_scale * joy_up
    vy = x_scale * joy_left
    action = {
        "linear_velocity": np.array([vx, vy, 0.0]),
        "angular_velocity": np.array([0.0, 0.0, 0.0]),
        "grip_open": 1,
    }
    action_2d = [vx, vy]
    # print('Sending', action)
    if done:
        print('Finished episode; Resetting arm')
        obs = reset_arm(env)
        dataset.reset(obs)
        dataset.save()
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


def main():
    pick_env = gym.make("RealRobot-Pick-v0", cam_list=[], arm='right')
    real_obs = reset_arm(pick_env)

    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    dataset_path = os.path.join(os.environ['TOP_DATA_DIR'],
                                f'rrlfd/pushing_demos_sim_dev_{timestamp}.pkl')
    dataset = Dataset(dataset_path)
    dataset.reset(real_obs)
    env_step_callback = functools.partial(
        callback, env=pick_env, dataset=dataset, x_scale=0.05, y_scale=0.05)
    try:
        rospy.Subscriber('joy', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')

        rospy.spin()
    except rospy.ROSInterruptException:
        dataset.save()


if __name__ == "__main__":
    main()


