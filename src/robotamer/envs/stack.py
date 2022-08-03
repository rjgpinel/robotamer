import rospy
import time

from numpy import pi
from robotamer.envs.base import BaseEnv

INIT_X = 0.085
INIT_Y = 0.24
CUBES_HEIGHT = 0.025*2
CUBES_DIST = 0.17


class StackEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(StackEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.safe_height = 0.14

    def set_scene(self, sim_env):
        sim_obs = sim_env.observe()
        # num_cubes = sim_env.num_cubes + sim_env.num_distractors
        num_cubes = 2

        self.safe_height = 0.1

        # initial_xy = [[INIT_X + (-CUBES_DIST * i), INIT_Y] for i in range(num_cubes)]
        initial_xy = [[-0.745, 0.225], [-0.745, 0.0], [-0.745, -0.225], [-0.81, 0.115], [-0.81, -0.115]]
        default_orn = [pi, 0, pi / 2]

        for i in range(num_cubes):
            x_get, y_get = initial_xy[i]
            center_success = self.center_object_pos([x_get, y_get, CUBES_HEIGHT/2])
            cube_pos = sim_obs[f"cube{i}_pos"]
            put_success = self.put(cube_pos, default_orn)
            if not put_success:
                raise RuntimeError("Setting scene failed")
            if rospy.is_shutdown():
                break

        self.safe_height = 0.14

        return True

    def clean_scene(self, sim_env):
        sim_obs = sim_env.observe()
        # eef_safe_pos, eef_orn = self.robot.eef_pose()
        # eef_safe_pos[-1] = self.safe_height
        # success_safe_pos = self.robot.go_to_pose(eef_safe_pos, eef_orn)
        # num_cubes = sim_env.num_cubes + sim_env.num_distractors
        num_cubes = 2
        # initial_xy = [[INIT_X + (-CUBES_DIST * i), INIT_Y] for i in range(num_cubes)]
        initial_xy = [[-0.745, 0.225], [-0.745, 0.0], [-0.745, -0.225], [-0.81, 0.115], [-0.81, -0.115]]
        default_orn = [pi, 0, pi / 2]
        for i in sorted(list(range(num_cubes)), reverse=True):
            time.sleep(2)
            cube_pos = sim_obs[f"cube{i}_pos"]
            get_success = self.pick(cube_pos, default_orn)
            if not get_success:
                raise RuntimeError("Cleaning scene failed during picking")
            if rospy.is_shutdown():
                break

            x_put, y_put = initial_xy[i]
            put_success = self.put([x_put, y_put, CUBES_HEIGHT/2], default_orn)
            if not put_success:
                raise RuntimeError("Cleaning scene failed during putting")
            if rospy.is_shutdown():
                break

        return True
