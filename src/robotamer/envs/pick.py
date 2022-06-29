import rospy
import numpy as np

from math import pi
from robotamer.envs.base import BaseEnv
from prl_ur5_demos.utils import make_pose
from robotamer.core.constants import ROBOT_BASE_FRAME, EEF_STEPS, JUMP_THRESHOLD


class PickEnv(BaseEnv):
    def __init__(self, cam_list, depth=False):
        super(PickEnv, self).__init__(cam_list=cam_list, depth=depth)

    def safe_move_cartesian(self, pos, orn):
        current_pose = self.robot.eef_pose()
        current_pos, current_orn = current_pose
        waypoints = [
            current_pose,
            make_pose(
                [current_pos[0], current_pos[1], GRIPPER_HEIGHT_INIT[1]],
                current_orn,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
            make_pose(
                [pos[0], pos[1], GRIPPER_HEIGHT_INIT[1]],
                current_ori,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
            make_pose(
                [pos[0], pos[1], self.safe_height[1]], orn, frame_id=ROBOT_BASE_FRAME
            ).pose,
            make_pose(
                [pos[0], pos[1], pos[2]],
                orn,
                frame_id=ROBOT_BASE_FRAME,
            ).pose,
        ]

        # Move in cartesian space through the waypoints
        path, fraction = self.real_robot.robot.left_arm.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=EEF_STEPS,
            jump_threshold=JUMP_THRESHOLD,
        )

        if fraction < 1:
            rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
            return False

        state = self.robot.commander.get_current_state()
        path = self.real_robot.robot.left_arm.retime_trajectory(
            state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
        )

        success = self.robot.commander.left_arm.execute(path, wait=True)
        return success

    def pick(self, pick_pos, pick_orn):
        move_success = self.safe_move_cartesian(pick_pos, pick_orn)

        if not move_success:
            return move_success

        # Allow collision between the table and the cube during the pick and place
        self.real_robot.robot.left_arm.set_support_surface_name("table")
        touch_links = self.real_robot.robot.get_link_names(group="left_gripper")
        touch_links.append("left_camera_link")

        self.robot.commander.left_gripper.set_named_target("close")
        success = self.robot.commander.left_gripper.go(wait=True)
        return success

    def put(self, put_pos, put_orn):
        move_success = self.safe_move_cartesian(put_pos, put_orn)

        if not move_success:
            return move_success

        self.robot.commander.left_arm.set_support_surface_name("table")
        self.robot.commander.left_gripper.set_named_target("open")
        success = self.robot.commander.left_gripper.go(wait=True)
        return success

    def center_object_pos(self, pos):
        current_pose = self.robot.eef_pose()
        current_pos, current_orn = current_pose

        # Pick and open
        success = self.pick(pos, [pi, 0, pi / 2])
        self.robot.commander.left_gripper.set_named_target("open")
        success = self.robot.commander.left_gripper.go(wait=True)

        waypoints = [
            current_pose,
            make_pose(current_pos, [pi, 0, 0], frame_id=ROBOT_BASE_FRAME).pose,
        ]
        # Move in cartesian space through the waypoints
        path, fraction = self.robot.commander.left_arm.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=EEF_STEPS,
            jump_threshold=JUMP_THRESHOLD,
        )
        if fraction < 1:
            rospy.logerr(f"Failed to plan cartesian path. Fraction Path: {fraction}")
            return False

        state = self.robot.commander.get_current_state()
        path = self.robot.commander.left_arm.retime_trajectory(
            state, path, 0.5, 0.5, "time_optimal_trajectory_generation"
        )
        success = self.robot.commander.left_arm.execute(path, wait=True)
        self.robot.commander.left_gripper.set_named_target("close")
        success = self.robot.commander.left_gripper.go(wait=True)
        return success
