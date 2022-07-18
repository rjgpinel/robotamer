from numpy import pi

from robotamer.envs.base import BaseEnv

class PickEnv(BaseEnv):
    INITIAL_XY = [-0.1125, 0.0]
    CUBE_HEIGHT = 0.0225 * 2


    def __init__(self, cam_list, depth=False):
        super(PickEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.safe_height = 0.1
        self.clean_i = 0

    def set_scene(self, sim_env):
        sim_obs = sim_env.observe()
        default_orn = [pi, 0, pi / 2]
        cube_pos = sim_obs[f"cube0_pos"]
        put_success = self.put([cube_pos[0], cube_pos[1], self.CUBE_HEIGHT/2], default_orn)
        if not put_success:
            raise RuntimeError("Setting scene failed")

        return True

    def clean_scene(self, sim_env):
        sim_obs = sim_env.observe()
        x_get, y_get, _ = sim_obs["cube0_pos"]
        if self.clean_i % 2 == 0:
            default_orn = [pi, 0, pi / 2]
        else:
            default_orn = [pi, 0, 0]
        self.clean_i += 1

        pick_success = self.pick([x_get, y_get, self.CUBE_HEIGHT / 2], default_orn)
        if not pick_success:
            raise RuntimeError("Cleaning scene failed")

        return True
