from numpy import pi

from robotamer.envs.base import BaseEnv
from robotamer.core import constants 


class PushEnv(BaseEnv):

    def __init__(self, cam_list, *args, arm='right', depth=False, version='v0',
                 **kwargs):
        super().__init__(
            *args, cam_list=cam_list, arm=arm, depth=depth, version=version,
            **kwargs)
        if arm in constants.TASK_START_CONFIG:
            self.start_config = constants.TASK_START_CONFIG[arm][self.version]
        else:
            self.start_config = None
        self.start_eef_pos = [0.4, 0, 0.04] if arm == 'right' else [-0.4, 0, 0.3]
    
    def reset_joints_and_eef(self):
        obs = self.reset(joints=self.start_config, gripper_pos=self.start_eef_pos)
        return obs
    
    def reset_eef(self):
        obs = super().reset(gripper_pos=self.start_eef_pos, open_gripper=False)
        return obs
    
    def reset(self):
        if self.start_config is not None:
            obs = super().reset(joints=self.start_config)
        else:
            obs = self.reset_eef()
        return obs

    # def set_scene(self, sim_self):
    #     sim_obs = sim_self.observe()
    #     default_orn = [pi, 0, pi / 2]
    #     cube_pos = sim_obs[f"cube0_pos"]
    #     put_success = self.put([cube_pos[0], cube_pos[1], self.CUBE_HEIGHT/2], default_orn)
    #     if not put_success:
    #         raise RuntimeError("Setting scene failed")

    #     return True

    # def clean_scene(self, sim_self):
    #     sim_obs = sim_self.observe()
    #     x_get, y_get, _ = sim_obs["cube0_pos"]
    #     if self.clean_i % 2 == 0:
    #         default_orn = [pi, 0, pi / 2]
    #     else:
    #         default_orn = [pi, 0, 0]
    #     self.clean_i += 1

    #     pick_success = self.pick([x_get, y_get, self.CUBE_HEIGHT / 2], default_orn)
    #     if not pick_success:
    #         raise RuntimeError("Cleaning scene failed")

    #     return True
