import gym
import rospy

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from math import pi
from gym.utils import seeding
from prl_ur5_demos.utils import make_pose
from robotamer.core.robot import Robot
from robotamer.core.constants import (
    REAL_DT,
    SIM_DT,
    CAM_INFO,
    ROBOT_BASE_FRAME,
    EEF_STEPS,
    JUMP_THRESHOLD,
)

WORKSPACE = {
    "left": np.array([[-0.695, -0.175, 0.00], [-0.295, 0.175, 0.2]]),
    "right": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]]),
}

GRIPPER_HEIGHT_INIT = np.array([0.06, 0.10])

DEFAULT_CONF = {
    "left": [
        -0.9773843811168246,
        -1.7627825445142729,
        -2.321287905152458,
        -1.1344640137963142,
        -2.199114857512855,
        -2.3387411976724017,
    ],
    "right":  # [
    #     1.2217304763960306,
    #     -1.4486232791552935,
    #     1.4835298641951802,
    #     1.9024088846738192,
    #     -2.2863813201125716,
    #     -2.670353755551324,
    # ]
    [
        1.2915436464758039,
        -1.6929693744344996,
        1.5533430342749532,
        -1.1344640137963142,
        2.303834612632515,
        0  # for cylinder
        # -1.064650843716541,  # for gripper
    ],
    # Pushing start position.
    # [0.9326989669308903, -1.752094800707198, 1.7692552040365719, -1.070880584564021, 2.1902111646944924, 2.3613882129912653]
    # [-1.544117350934758, -1.545942555282938, -1.54736573102487, 0.0038076134003528495, 1.5741106580621542, -0.785054080293274],
    # [
}


class BaseEnv(gym.Env):
    def __init__(
        self,
        cam_list=["bravo_camera", "charlie_camera"],
        depth=False,
        cam_info=None,
        arm="left",
    ):
        rospy.init_node("env_node", log_level=rospy.INFO)

        # Workspace definition
        self.workspace = WORKSPACE[arm]
        self.gripper_workspace = self.workspace.copy()
        self.gripper_workspace[:, 2] = GRIPPER_HEIGHT_INIT

        self.home_config = DEFAULT_CONF[arm]

        self.rate = rospy.Rate(1.0 / REAL_DT)
        self.cam_list = cam_list

        # Controller
        self.robot = Robot(self.workspace, cam_list, depth=depth, arm=arm)
        self.arm_name = arm

        # Depth flag
        self._depth = depth
        self._grip_history = deque(maxlen=5)
        # import pudb; pudb.set_trace()

        # Cam info
        self.cam_info = {
            f"info_{cam_name}": CAM_INFO[cam_name] for cam_name in cam_list
        }
        self._np_random = np.random

        self.safe_height = GRIPPER_HEIGHT_INIT[-1]

    def seed(self, seed):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random

    def sample_random_gripper_pos(self):
        position = self._np_random.uniform(
            self.gripper_workspace[0], self.gripper_workspace[1]
        )
        return position

    def _get_current_config(self):
        variables = self.robot.commander.get_current_variable_values()
        config_joints = [f'{self.arm_name}_{k}' for k in [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint']]
        config = [variables[k] for k in config_joints]
        return config

    def _reset(self, gripper_pos=None, gripper_orn=None, open_gripper=True, joints=None, home_only=False, **kwargs):
        print('Returning to home config')
        success = self.robot.set_config(self.home_config)
        if success:
            print('Done')
        else:
            print('Failed to return')
        config = self._get_current_config()
        diff = np.subtract(config, self.home_config)

        print('Current config vs. home; difference')
        np.set_printoptions(suppress=True, linewidth=90)
        print(np.array2string(
            np.stack([config, self.home_config, diff]), separator=', '))
        np.set_printoptions(suppress=False, linewidth=75)
        if home_only:
            return

        if joints is not None:
            print('Setting to custom config')
            success = self.robot.set_config(joints)
            config = self._get_current_config()
            diff = np.subtract(config, joints)

            print('Current config vs. target; difference')
            np.set_printoptions(suppress=True, linewidth=90)
            print(np.array2string(
                np.stack([config, joints, diff]), separator=', '))
            np.set_printoptions(suppress=False, linewidth=75)
        if gripper_pos is not None or gripper_orn is not None:
            if gripper_pos is None:
                gripper_pos = self.robot.eef_pose()[0]
                # gripper_pos = self.sample_random_gripper_pos()

            if gripper_orn is None:
                gripper_orn = self.robot.eef_pose()[1]
                # gripper_orn = [pi, 0, pi / 2]

            print('Moving to cartesian pose', gripper_pos, gripper_orn)
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

        if "grip_open" in action.keys():
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

    def safe_move_cartesian(self, pos, orn):
        current_pose = self.robot.eef_pose()
        current_pos, current_orn = current_pose
        waypoints = [
            make_pose(current_pos, current_orn, frame_id=ROBOT_BASE_FRAME).pose,
            make_pose(
                [current_pos[0], current_pos[1], self.safe_height],
                current_orn,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
            make_pose(
                [pos[0], pos[1], self.safe_height],
                current_orn,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
            make_pose(
                [pos[0], pos[1], self.safe_height], orn, frame_id=ROBOT_BASE_FRAME
            ).pose,
            make_pose(
                pos,
                orn,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
        ]

        # Move in cartesian space through the waypoints
        path, fraction = self.robot.arm.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=EEF_STEPS,
            jump_threshold=JUMP_THRESHOLD,
        )

        if fraction < 1:
            rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
            return False

        state = self.robot.commander.get_current_state()
        path = self.robot.arm.retime_trajectory(
            state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
        )

        success = self.robot.arm.execute(path, wait=True)
        return success

    def pick(self, pick_pos, pick_orn):
        move_success = self.safe_move_cartesian(pick_pos, pick_orn)

        if not move_success:
            return move_success

        # Allow collision between the table and the cube during the pick and place
        self.robot.arm.set_support_surface_name("table")
        touch_links = self.robot.commander.get_link_names(
            group="{self.arm_name}_gripper"
        )
        touch_links.append("{self.arm_name}_camera_link")

        self.robot.gripper.set_named_target("close")
        success = self.robot.gripper.go(wait=True)
        return success

    def put(self, put_pos, put_orn):
        move_success = self.safe_move_cartesian(put_pos, put_orn)

        if not move_success:
            return move_success

        self.robot.arm.set_support_surface_name("table")
        self.robot.gripper.set_named_target("open")
        success = self.robot.gripper.go(wait=True)
        return success

    def center_object_pos(self, pos):
        current_pose = self.robot.eef_pose()
        current_pos, current_orn = current_pose

        # Pick and open
        success = self.pick(pos, [pi, 0, pi / 2])
        self.robot.gripper.set_named_target("open")
        success = self.robot.gripper.go(wait=True)

        waypoints = [
            make_pose(pos, [pi, 0, pi / 2], frame_id=ROBOT_BASE_FRAME).pose,
            make_pose(pos, [pi, 0, 0], frame_id=ROBOT_BASE_FRAME).pose,
        ]
        # Move in cartesian space through the waypoints
        path, fraction = self.robot.arm.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=EEF_STEPS,
            jump_threshold=JUMP_THRESHOLD,
        )
        if fraction < 1:
            rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
            return False

        state = self.robot.commander.get_current_state()
        path = self.robot.arm.retime_trajectory(
            state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
        )
        success = self.robot.arm.execute(path, wait=True)
        self.robot.gripper.set_named_target("close")
        success = self.robot.gripper.go(wait=True)
        return success

    def move(self, gripper_pos, gripper_quat=None, open_gripper=True):

        gripper_pos = gripper_pos.astype(np.float)
        gripper_quat = None

        if gripper_quat is not None:
            gripper_orn = quat_to_euler(gripper_quat, False)
        else:
            gripper_orn = [pi, 0, pi / 2]
        success = self.robot.go_to_pose(gripper_pos, gripper_orn, cartesian=True)

        if open_gripper:
            self.robot.move_gripper("open", wait=True)
            pass
        else:
            self.robot.move_gripper("close", wait=True)
            pass

        obs = self.render()
        return (
            obs,
            None,
            False,
            {"success": success},
        )
