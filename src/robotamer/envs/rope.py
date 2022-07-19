import rospy
import numpy as np

import matplotlib.pyplot as plt
from numpy import pi
from robotamer.envs.base import BaseEnv


class RopeEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(RopeEnv, self).__init__(cam_list=cam_list, depth=depth)

    def set_scene(self, sim_env):
        raise NotImplementedError("Not tested yet!")

    def clean_scene(self, sim_env):
        raise NotImplementedError("Not tested yet!")


    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = -new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)


