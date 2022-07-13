from numpy import pi

from robotamer.envs.base import BaseEnv

INITIAL_XY = [-0.1125, 0.0]
CUBE_HEIGHT = 0.0225 * 2

class PushAndPickEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(PushAndPickEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.safe_height = 0.1

    def set_scene(self, sim_env):
        sim_obs = sim_env.observe()
        default_orn = [pi, 0, pi / 2]

        x_get, y_get = INITIAL_XY
        center_success = self.center_object_pos([x_get, y_get, CUBE_HEIGHT / 2])
        cube_pos = sim_obs[f"cube0_pos"]

        put_success = self.put([cube_pos[0], cube_pos[1], CUBE_HEIGHT/2], default_orn)
        if not put_success:
            raise RuntimeError("Setting scene failed")

        return True

    def clean_scene(self, sim_env):
        sim_obs = sim_env.observe()
        x_put, y_put = INITIAL_XY
        default_orn = [pi, 0, pi / 2]
        put_success = self.put([x_put, y_put, CUBE_HEIGHT / 2], default_orn)
        if not put_success:
            raise RuntimeError("Cleaning scene failed")

        return True
