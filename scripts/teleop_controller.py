import functools
import time
import robotamer.envs
import gym
import rospy

import numpy as np
from math import pi

from PIL import Image
from robotamer.core.constants import SIM_DT
from robotamer.envs.pick import PickEnv
from sensor_msgs.msg import Joy


def callback(data, env, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    joy_left = data.axes[2]
    joy_up = data.axes[3]
    vx = x_scale * joy_up
    vy = y_scale * joy_left
    action = {
        "linear_velocity": np.array([vx, vy, 0.0]),
        "angular_velocity": np.array([0.0, 0.0, 0.0]),
        "grip_open": 1,
    }
    print('Sending', action)
    real_obs = env.step(action)


def main():
    pick_env = gym.make("RealRobot-Pick-v0", cam_list=[], arm='right')

    if pick_env.arm_name == 'left':
        gripper_pos = [-0.40, 0, 0.1]
        gripper_orn = [pi, 0, pi / 2]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
    real_obs = pick_env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)

    # rospy.init_node('teleop_controller')
    env_step_callback = functools.partial(callback, env=pick_env)
    rospy.Subscriber('joy_teleop', Joy, env_step_callback)
    print('Ready to receive joystick controls')

    rospy.spin()


if __name__ == "__main__":
    main()
