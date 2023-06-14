import rospy
import numpy as np

from copy import copy
from math import pi
from robotamer.envs.base import BaseEnv


class PushEnv(BaseEnv):
    def __init__(self, cam_list, *args, depth=False, **kwargs):
        super(PushEnv, self).__init__(*args, cam_list=cam_list, depth=depth, **kwargs)

        workspace_x_center = (
            1 * (self.workspace[1][0] - self.workspace[0][0]) / 8
        ) + self.workspace[0][0]

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace[0, 0] = workspace_x_center
        self.obj_workspace += np.array([[0.25, 0.08, 0.01], [-0.05, -0.08, -0.01]])

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[1, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.0, 0.08, 0.01], [-0.01, -0.08, -0.01]])

        self.default_gripper_height = 0.035
        self.min_x_distance = 0.0255
        self.min_y_distance = 0.06

        self._initial_gripper_pos = np.array([self.workspace[0][1], 0.0, 0.0])
        self._initial_gripper_pos[-1] = self.default_gripper_height


    def _reset(self, **kwargs):
        super()._reset(gripper_pos=self._initial_gripper_pos, open_gripper=False)


    def step(self, action):
        action = copy(action)
        linear_velocity = np.zeros(3)
        linear_velocity[:2] = action.pop("xy_linear_velocity")
        action["linear_velocity"] = linear_velocity
        return super().step(action)

