import gym
import rospy

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from math import pi
from gym.utils import seeding
from robotamer.core.robot import Robot
from robotamer.core.constants import REAL_DT, SIM_DT, CAM_INFO

WORKSPACE = np.array([[-0.695, -0.175, 0.00], [-0.295, 0.175, 0.2]])

GRIPPER_HEIGHT_INIT = np.array([0.06, 0.10])

LEFT_DEFAULT_CONF = [
    -0.73303829,
    -1.6406095,
    -1.98967535,
    -1.72787596,
    -2.05948852,
    -2.565634,
]


class BaseEnv(gym.Env):
    def __init__(
        self, cam_list=["bravo_camera", "charlie_camera"], depth=False, cam_info=None
    ):
        rospy.init_node("env_node", log_level=rospy.INFO)

        # Workspace definition
        self.workspace = WORKSPACE
        self.gripper_workspace = self.workspace.copy()
        self.gripper_workspace[:, 2] = GRIPPER_HEIGHT_INIT

        self.left_home_config = LEFT_DEFAULT_CONF

        self.rate = rospy.Rate(1.0 / REAL_DT)
        self.cam_list = cam_list

        # Controller
        self.robot = Robot(self.gripper_workspace, cam_list, depth=depth)

        # Depth flag
        self._depth = depth
        self._grip_history = deque(maxlen=5)

        # Cam info
        self.cam_info = {CAM_INFO[cam_name] for cam_name in cam_list}
        self._np_random = np.random

    def seed(self, seed):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random

    def sample_random_gripper_pos(self):
        position = self._np_random.uniform(
            self.gripper_workspace[0], self.gripper_workspace[1]
        )
        return position

    def _reset(self, gripper_pos=None, gripper_orn=None, open_gripper=True, **kwargs):
        self.robot.set_config(self.left_home_config)
        if gripper_pos is None:
            gripper_pos = self.sample_random_gripper_pos()

        if gripper_orn is None:
            gripper_orn = [pi, 0, pi / 2]

        success = self.robot.go_to_pose(gripper_pos, gripper_orn, cartesian=True)

        if not success:
            print("Moving the robot to default position failed")
            exit()

        self.robot.reset(open_gripper=open_gripper)

    def reset(self, **kwargs):
        self._reset(**kwargs)
        return self.render()

    def step(self, action):
        processed_action = {}

        # Move the gripper
        grip_open = action["grip_open"]
        self._grip_history.append(grip_open)
        grip_open_mean = np.mean(self._grip_history)

        if grip_open_mean > 0:
            self.robot.move_gripper("open", wait=False)
        else:
            self.robot.move_gripper("close", wait=False)

        processed_action["linear_velocity"] = (
            action["linear_velocity"] * SIM_DT / REAL_DT
        )
        processed_action["angular_velocity"] = (
            action["angular_velocity"] * SIM_DT / REAL_DT
        )

        # Move the arm
        move_success, msg = self.robot.move_relative(
            REAL_DT,
            processed_action["linear_velocity"],
            processed_action["angular_velocity"],
        )

        self.rate.sleep()
        obs = self.render()

        # Default real robot outputs
        success = False
        reward = 0

        return (
            obs,
            reward,
            success,
            {"success": success, "failure_message": msg},
        )

    def render(self):
        obs = {}

        for cam_name in self.robot.cam_list:
            cam = self.robot.cameras[cam_name]
            obs[f"rgb_{cam_name}"] = cam.record_image(dtype=np.uint8)

            if self._depth:
                depth_cam = self.robot.depth_cameras[cam_name]
                obs[f"depth_{cam_name}"] = (
                    depth_cam.record_image(dtype=np.uint16)
                    .astype(np.float32)
                    .squeeze(-1)
                    / 1000
                )

            obs[f"info_{cam_name}"] = self.cam_info[f"info_{cam_name}"]

        gripper_pose = self.robot.eef_pose()
        obs["gripper_pos"] = np.array(gripper_pose[0])
        obs["gripper_quat"] = np.array(gripper_pose[1])
        obs["grip_velocity"] = self.robot._grip_velocity
        obs["gripper_state"] = self.robot._grasped

        # TODO: Add joints state

        return obs

    def close(self):
        pass
