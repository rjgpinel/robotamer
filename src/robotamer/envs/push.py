import rospy
import numpy as np

from math import pi
from robotamer.envs.base import BaseEnv


class PushEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(PushEnv, self).__init__(cam_list=cam_list, depth=depth)

        workspace_x_center = (
            1 * (self.workspace[1][0] - self.workspace[0][0]) / 8
        ) + self.workspace[0][0]

        self.obj_workspace = self.workspace.copy()
        self.obj_workspace[0, 0] = workspace_x_center
        self.obj_workspace += np.array([[0.25, 0.08, 0.01], [-0.05, -0.08, -0.01]])

        self.goal_workspace = self.workspace.copy()
        self.goal_workspace[1, 0] = workspace_x_center
        self.goal_workspace += np.array([[0.0, 0.08, 0.01], [-0.01, -0.08, -0.01]])
