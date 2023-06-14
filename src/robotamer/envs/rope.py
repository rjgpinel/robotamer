import rospy
import numpy as np

import matplotlib.pyplot as plt
from numpy import pi
from robotamer.envs.base import BaseEnv


class RopeEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        workspace = np.array([[-0.695, -0.175, 0.02], [-0.295, 0.175, 0.2]])
        super(RopeEnv, self).__init__(cam_list=cam_list, depth=depth, workspace=workspace)
        self.left_home_config=[
            -0.9773843811168246,
            -1.7627825445142729,
            -2.321287905152458,
            -1.1344640137963142,
            -2.199114857512855,
            0.785398,
        ]

    def _reset(self, **kwargs):
        super()._reset(gripper_orn=[pi, 0, -pi/2], open_gripper=True)


    def set_scene(self, sim_env):
        raise NotImplementedError("Not tested yet!")

    def clean_scene(self, sim_env):
        raise NotImplementedError("Not tested yet!")


    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = -new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)


