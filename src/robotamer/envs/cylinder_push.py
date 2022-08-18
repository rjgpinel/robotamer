import gym
import numpy as np

from robotamer.envs.base import BaseEnv
from robotamer.core import constants 


class PushEnv(BaseEnv):

    def __init__(self, cam_list, *args, arm='right', depth=False, version='v0',
                 open_gripper=False, **kwargs):
        super().__init__(
            *args, cam_list=cam_list, arm=arm, depth=depth, version=version,
            open_gripper=open_gripper, **kwargs)
        if arm in constants.TASK_START_CONFIG:
            self.start_config = constants.TASK_START_CONFIG[arm][self.version]
        else:
            self.start_config = None
        self.start_eef_pos = (
            [0.4, 0, 0.04] if arm == 'right' else [-0.4, 0, 0.06])

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-0.05, -0.05]),
                              high=np.array([0.05, 0.05]))
    
    def reset_joints_and_eef(self):
        obs = self.reset(joints=self.start_config,
                         gripper_pos=self.start_eef_pos,
                         open_gripper=False)
        return obs
    
    def reset_eef(self):
        obs = super().reset(gripper_pos=self.start_eef_pos, open_gripper=False)
        return obs
    
    def reset(self):
        if self.start_config is not None:
            obs = super().reset(joints=self.start_config, open_gripper=False)
        else:
            obs = self.reset_eef()
        return obs
