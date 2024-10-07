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

    real_obs = pick_env.reset()

    error_nb = 0
    time_prev = time.time()
    time_list = []


    # step_size = 0.1
    # n_step = 100 / step_size
    # speed = 0.01

    # theta_obs = []

    # for i in range(int(n_step)):
    #     # print("step ", i)
    #     real_obs = pick_env.step(
    #         {
    #             "linear_velocity": np.array([speed, -speed, speed]),
    #             "angular_velocity": np.array([0.00, 0.0, 0.0]),
    #             "grip_open": 1,
    #         }
    #     )

    pick_env.move(np.array([-0.63477445, -0.0380112,0.085603]),  open_gripper=True)
    
    print("Done")


if __name__ == "__main__":
    main()
