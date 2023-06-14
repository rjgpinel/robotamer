import rospy
import time

from numpy import pi
from robotamer.envs.base import BaseEnv

INIT_X = 0.085
INIT_Y = 0.24
CUBES_HEIGHT = 0.025*2
CUBES_DIST = 0.17


class CupboardEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(CupboardEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.safe_height = 0.12
        self.gripper_workspace[:, 2] = self.safe_height
        self.set_counter = 0

    def set_scene(self, sim_env, initial_xy=[[-0.745, 0.225], [-0.745, 0.0], [-0.745, -0.225]]):
        sim_obs = sim_env.observe()
        num_cubes = 1 + sim_env.num_distractors
        orn=[pi, 0, pi / 2]
        for i in range(num_cubes):
            x_get, y_get = initial_xy[i]
            if self.set_counter == 0:
                center_success = self.center_object_pos([x_get, y_get, CUBES_HEIGHT/2])
            elif self.set_counter % 2 == 0:
                    orn=[pi, 0, pi / 2]
                    pick_success = self.pick([x_get, y_get, CUBES_HEIGHT / 2], orn)
            else:
                 orn = [pi, 0, 0]
                 pick_success = self.pick([x_get, y_get,CUBES_HEIGHT / 2], orn)

            cube_pos = sim_obs[f"cube{i}_pos"]
            put_success = self.put(cube_pos, orn)
            if not put_success:
                raise RuntimeError("Setting scene failed")
            if rospy.is_shutdown():
                break
        self.set_counter+=1
        return True

    def clean_scene(self, sim_env):
        return True
