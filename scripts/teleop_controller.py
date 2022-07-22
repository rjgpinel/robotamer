import functools
import time
import robotamer.envs
import gym
import rospy
import pickle

import numpy as np
from math import pi

from PIL import Image
from robotamer.core.constants import SIM_DT
from robotamer.envs.pick import PickEnv
from sensor_msgs.msg import Joy


class Dataset:
    episodes = []

    def reset(self, obs):
        if self.episodes:
            for k in self.episodes[-1]:
                self.episodes[k] = np.array(self.episodes[k])
        self.episodes.append({'observations': [obs], 'actions': []})

    def append(self, act, next_obs):
        self.episodes[-1]['actions'].append(act)
        self.episodes[-1]['observations'].append(next_obs)

    def save(self, path):
        if self.episodes and not self.episodes[-1]['actions']:
            # Remove last empty episode.
            self.episodes = self.episodes[:-1]
        with open(path, 'wb') as f:
            pickle.dump(self.episodes, f)


def callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    joy_left = data.axes[0]
    joy_up = data.axes[1]
    done = data.buttons[0]
    vx = x_scale * joy_up
    vy = y_scale * joy_left
    action = {
        "linear_velocity": np.array([vx, vy, 0.0]),
        "angular_velocity": np.array([0.0, 0.0, 0.0]),
        "grip_open": 1,
    }
    # print('Sending', action)
    if done:
        print('Finished episode; Resetting arm')
        obs = reset_arm(env)
        dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
        real_obs = env.step(action)
        dataset.append(action, real_obs)


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

    dataset = Dataset()
    dataset.reset(real_obs)
    env_step_callback = functools.partial(callback, env=pick_env, dataset=dataset)
    try:
        rospy.Subscriber('joy_teleop', Joy, env_step_callback)
        print('Ready to receive joystick controls')

        rospy.spin()
    except rospy.ROSInterruptException:
        dataset.save('/home/rgarciap/catkin_ws/src/robotamer/pushing_demos.pkl')


if __name__ == "__main__":
    main()


