import time
import robotamer.envs
import gym

import numpy as np
from math import pi

from PIL import Image
from robotamer.core.constants import SIM_DT
from robotamer.envs.pick import PickEnv


def main():
    pick_env = gym.make("RealRobot-Pick-v0", cam_list=[], arm='right')
    # pickenv = PickEnv([])

    if pick_env.arm_name == 'left':
        gripper_pos = [-0.40, 0, 0.1]
        gripper_orn = [pi, 0, pi / 2]
        # angular_velocity = [0., 0., 3.14 / 10]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
        # angular_velocity = [0., 0., -3.14 / 10]
    real_obs = pick_env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)

    error_nb = 0
    time_prev = time.time()
    time_list = []

    # input("Gripper should be good...")

    step_size = 0.1
    n_step = 10 / step_size
    for i in range(int(n_step / 1.5)):
        real_obs = pick_env.step(
            {
                "linear_velocity": np.array([-0.01, 0, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "grip_open": 1,
            }
        )
    for i in range(int(n_step / 1.5)):
        real_obs = pick_env.step(
            {
                "linear_velocity": np.array([0.0, 0.01, 0.0]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "grip_open": 1,
            }
        )

    for i in range(int(n_step / 1.5)):
        real_obs = pick_env.step(
            {
                "linear_velocity": np.array([0.0, 0, 0.01]),
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
                "grip_open": 1,
            }
        )

    # print(f"Gripper should have moved {n_step*step_size*SIM_DT} m")
    print("Done")


if __name__ == "__main__":
    main()
