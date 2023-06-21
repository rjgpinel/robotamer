import functools
import numpy as np

from gym import spaces
from scipy.spatial.transform import Rotation


def safe_random_pose(self):
    current_pose = self.real_robot.robot.left_arm.get_current_pose().pose
    current_pos = [
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
    ]
    current_ori = [
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w,
    ]
    ori = [pi, 0, pi / 2]

    random_x, random_y, random_z = self.sample_random_gripper_pos()

    waypoints = [
        current_pose,
        make_pose(
            [current_pos[0], current_pos[1], self.safe_height[1]],
            current_ori,
            frame_id="prl_ur5_base",
        ).pose,
        make_pose(
            [random_x, random_y, random_z],
            ori,
            frame_id="prl_ur5_base",
        ).pose,
    ]

    # Move in cartesian space through the waypoints
    path, fraction = self.real_robot.robot.left_arm.compute_cartesian_path(
        waypoints=waypoints,
        eef_step=0.001,
        jump_threshold=0.0,
    )

    if fraction < 0.1:
        rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
        return False

    state = self.real_robot.robot.get_current_state()
    path = self.real_robot.robot.left_arm.retime_trajectory(
        state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
    )

    success = self.real_robot.robot.left_arm.execute(path, wait=True)

    return success


def euler_to_quat(euler, degrees):
    rotation = Rotation.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = Rotation.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)

def convert_to_spec(obs):
    Box = functools.partial(spaces.Box, low=-np.inf, high=np.inf)
    Img = functools.partial(spaces.Box, low=0, high=255, dtype=np.uint8)
    if isinstance(obs, dict):
        return spaces.Dict({k: convert_to_spec(v) for k, v in obs.items()})
    elif isinstance(obs, np.ndarray):
        if obs.dtype == np.uint8:
            return Img(shape=obs.shape)
        else:
            return Box(shape=obs.shape)
    elif isinstance(obs, bool):
        return spaces.Discrete(2)
    elif isinstance(obs, int) or isinstance(obs, float):
        return Box(shape=(), dtype=type(obs))
    else:
        raise TypeError(
            f'Could not convert observation {obs} of type {type(obs)} to spec')
