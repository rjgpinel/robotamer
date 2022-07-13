import rospy
import numpy as np

import matplotlib.pyplot as plt
from numpy import pi
from robotamer.envs.base import BaseEnv
from robotamer.envs.utils import quat_to_euler

INIT_X = 0.085
SCREW_HEIGHT = 0.105
NUT_HANDLE_OFFSET = 0.075
NUT_HANDLE_HEIGHT = 0.02


class NutAssemblyEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(NutAssemblyEnv, self).__init__(cam_list=cam_list, depth=depth)
        self.safe_height = 0.28

    def set_scene(self, sim_env):
        sim_obs = sim_env.observe()
        screw_block_size = SCREW_HEIGHT / 3
        initial_screw_pos = [-0.745, 0.225, screw_block_size * 2.5]
        default_orn = [pi, 0, pi / 2]

        screw_pos = sim_obs[f"screw0_pos"]
        screw_pos[-1] = screw_block_size * 2.5
        pick_success = self.pick(initial_screw_pos, default_orn)
        if not pick_success:
            raise RuntimeError("Setting scene failed")
        put_success = self.put(screw_pos, default_orn)
        if not put_success:
            raise RuntimeError("Setting scene failed")
        if rospy.is_shutdown():
            return False

        initial_handle_pos = [-0.745, NUT_HANDLE_OFFSET, NUT_HANDLE_HEIGHT]
        handle_default_orn = [pi, 0, 0]
        pick_success = self.pick(initial_handle_pos, handle_default_orn)

        nut_handle_pos = sim_env.scene.get_geom_pos("nut0_handle")
        nut_handle_quat = sim_env.scene.get_geom_quat("nut0_handle")

        nut_handle_pos[-1] = NUT_HANDLE_HEIGHT
        nut_handle_orn = quat_to_euler(nut_handle_quat, False)
        print(nut_handle_orn)
        handle_orn = np.array([pi, 0, -pi/2])
        handle_orn[-1] -= nut_handle_orn[0]

        put_success = self.put(nut_handle_pos, handle_orn)
        plt.imshow(sim_obs["rgb_charlie_camera"])
        plt.show()

        if not pick_success:
            raise RuntimeError("Setting scene failed")
        if not put_success:
            raise RuntimeError("Setting scene failed")
        if rospy.is_shutdown():
            return False

        return True

    def clean_scene(self, sim_env):
        sim_obs = sim_env.observe()
        eef_safe_pos, eef_orn = self.robot.eef_pose()
        eef_safe_pos[-1] = self.safe_height
        success_safe_pos = self.robot.go_to_pose(eef_safe_pos, eef_orn)
        return True


    def step(self, action):
        new_action = action.copy()
        new_action["angular_velocity"] = np.zeros(3)
        new_action["angular_velocity"][-1] = -new_action.pop("theta_velocity", 0.0)
        return super().step(new_action)


