import numpy as np

from numpy import pi

from robotamer.envs.base import BaseEnv

class SweepEnv(BaseEnv):

    def __init__(self, cam_list, depth=False):
        super(SweepEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.gripper_workspace[0, 2] = 0.05

    def _reset(self, **kwargs):
        super()._reset(gripper_orn=[pi, 0, 0], open_gripper=False)


    def set_scene(self, sim_env):
        raise NotImplementedError("Not implemented")

    def clean_scene(self, sim_env):
        raise NotImplementedError("Not implemented")

    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = -new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)


