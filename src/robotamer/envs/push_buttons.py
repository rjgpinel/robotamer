from numpy import pi

from robotamer.envs.base import BaseEnv

BUTTONS_DIST = 0.135
INIT_X = -0.1125
BUTTONS_HEIGHT = 0.02


class PushButtonsEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(PushButtonsEnv, self).__init__(cam_list=cam_list, depth=depth)

    def set_scene(self, sim_env):
        sim_obs = sim_env.observe()

        num_buttons = sim_envs.num_buttons + sim_envs.num_distractors
        initial_xy = [
            [INIT_X, BUTTONS_DIST + (-BUTTONS_DIST * i)] for i in range(num_buttons)
        ]
        default_orn = [pi, 0, pi / 2]

        for i in range(num_buttons):
            x_get, y_get = initial_xy[i]
            center_success = self.center_object_pos([x_get, y_get, BUTTONS_HEIGHT / 2])
            button_pos = sim_obs[f"button{i}_pos"]
            put_success = self.put(button_pos, default_orn)
            if not put_success:
                raise RuntimeError("Setting scene failed")
            if rospy.is_shutdown():
                break

        return True

    def clean_scene(self, sim_env):
        sim_obs = sim_env.observe()
        eef_safe_pos, eef_orn = self.robot.eef_pose()
        eef_safe_pos[-1] = GRIPPER_HEIGHT_INIT
        success_safe_pos = self.robot.go_to_pose(eef_safe_pos, eef_orn)
        num_buttons = sim_envs.num_buttons + sim_envs.num_distractors
        default_orn = [pi, 0, pi / 2]
        initial_xy = [
            [INIT_X, BUTTONS_DIST + (-BUTTONS_DIST * i)] for i in range(num_buttons)
        ]
        for i in range(num_buttons):
            button_pos = sim_obs[f"button{i}_pos"]
            get_success = self.get([
                button_pos[0],
                button_pos[1],
                BUTTONS_HEIGHT / 2], default_orn
            )
            if not get_success:
                raise RuntimeError("Cleaning scene failed")

            x_put, y_put = initial_xy[i]
            put_success = self.put([
                x_put,
                y_put,
                BUTTONS_HEIGHT / 2], default_orn
            )
            if not put_success:
                raise RuntimeError("Cleaning scene failed")
            if rospy.is_shutdown():
                break

        return True
