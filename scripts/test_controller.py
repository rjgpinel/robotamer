import time
import robotamer.envs
import gym
import rospy
import numpy as np
from math import pi

from numpy import pi

import matplotlib.pyplot as plt

from PIL import Image
from robotamer.core.constants import SIM_DT


def main():

    pick_env = gym.make("RealRobot-Pick-v0", cam_list=["charlie_camera", "bravo_camera"], arm='left')

    if pick_env.arm_name == 'left':
        gripper_pos = [-0.60, 0.15, 0.05]
        gripper_orn = [pi, 0, 0]
        # angular_velocity = [0., 0., 3.14 / 10]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
        # angular_velocity = [0., 0., -3.14 / 10]
    # real_obs = pick_env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)
    real_obs = pick_env.reset()

    error_nb = 0
    time_prev = time.time()
    time_list = []

    # # input("Gripper should be good...")

    step_size = 0.1
    n_step = 100 / step_size
    speed = 0.01

    theta_obs = []

    for i in range(int(n_step)):
        # print("step ", i)
        real_obs = pick_env.step(
            {
                "linear_velocity": np.array([speed, -speed, speed]),
                "angular_velocity": np.array([0.00, 0.0, 0.0]),
                "grip_open": 1,
            }
        )
    
    print("Done")


if __name__ == "__main__":
    main()
