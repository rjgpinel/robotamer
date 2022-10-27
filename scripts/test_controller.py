import time
import robotamer.envs
import gym
import rospy
import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from PIL import Image
from robotamer.core.constants import SIM_DT


def main():
    env = gym.make("RealRobot-Sweep-v0", cam_list=["charlie_camera", "bravo_camera"])

    gripper_pos = [-0.60, 0.15, 0.05]
    real_obs = env.reset()

    error_nb = 0
    time_prev = time.time()
    time_list = []

    # # input("Gripper should be good...")

    step_size = 0.1
    n_step = 100 / step_size

    theta_obs = []
    for i in range(int(n_step)):
        real_obs = env.step(
            {
                "linear_velocity": np.array([0.02, -0.02, -0.02]),
                "theta_velocity": np.array([3.14 / 10]),
                "grip_open": 1,
            }
        )

        theta_obs.append(real_obs[0]["gripper_theta"])
        if rospy.is_shutdown():
            break


    for i in range(int(n_step)):
        real_obs = env.step(
            {
                "linear_velocity": np.array([0.02, -0.02, -0.02]),
                "angular_velocity": np.array([0.0, 0.0, -3.14 / 10]),
                "grip_open": 1,
            }
        )

        theta_obs.append(real_obs[0]["gripper_theta"])
        if rospy.is_shutdown():
            break

    plt.plot(theta_obs)
    plt.show()
    print(theta_obs)
    print(f"Gripper should have moved {n_step*step_size*SIM_DT} m")
    print("Done")


if __name__ == "__main__":
    main()
