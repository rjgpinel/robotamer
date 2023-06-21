import pinocchio as pin
import moveit_commander
import rospy
import sys
import tf

import numpy as np

from moveit_msgs.msg import RobotState
from prl_ur5_demos.utils import make_pose
from robotamer.core.observer import Camera, CameraAsync, TFRecorder, JointStateRecorder
from robotamer.core.constants import (
    EEF_FRAME,
    OVERSHOOT_FACTOR,
    N_SAMPLES_OVERSHOOT,
    MAX_VELOCITY_SCALING_FACTOR,
    MAX_ACCELERATION_SCALING_FACTOR,
    PLANNING_TIME,
    COMMAND_ROS_TOPIC,
    EEF_STEPS,
    JUMP_THRESHOLD,
    JOINTS_STATE_TOPIC,
    Q_VEL_THRESHOLD,
    ROBOT_BASE_FRAME,
)
from robotamer.core.utils import compute_goal_pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class Robot:
    def __init__(self, workspace, cam_list, depth=False, cam_async=False, arm='left', open_gripper=True):
        # Create ros node
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize the robot
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.arm_name = arm
        self.arm = getattr(self.commander, f'{arm}_arm')
        self.gripper = getattr(self.commander, f'{arm}_gripper')
        self.eef_frame = EEF_FRAME[arm]

        # Configure the planning pipeline
        self.arm.set_max_velocity_scaling_factor(
            MAX_VELOCITY_SCALING_FACTOR
        )
        self.arm.set_max_acceleration_scaling_factor(
            MAX_ACCELERATION_SCALING_FACTOR
        )
        self.arm.set_planning_time(PLANNING_TIME)
        self.arm.set_planner_id("RRTstar")

        # Set eef link
        self.arm.set_end_effector_link(self.eef_frame)

        # limits
        self.workspace = workspace

        # Trajectory Publisher
        self._traj_publisher = rospy.Publisher(
            COMMAND_ROS_TOPIC[arm],
            JointTrajectory,
            queue_size=1,
        )

        # Transformations
        self.tf_listener = tf.TransformListener()
        self.tf_brodcaster = tf.TransformBroadcaster()
        # EEF transformation recorder
        self._eef_tf_recorder = TFRecorder(ROBOT_BASE_FRAME, self.eef_frame)
        # Joints state recorder
        self.joints_state_recorder = JointStateRecorder()

        # Cameras
        self.cam_list = cam_list
        self.cameras = {}
        self._depth = depth
        if self._depth:
            self.depth_cameras = {}

        for cam_name in cam_list:
            if cam_async:
                self.cameras[cam_name] = CameraAsync(f"/{cam_name}/color/image_raw")
                if self._depth:
                    self.depth_cameras[cam_name] = CameraAsync(
                        f"{cam_name}/aligned_depth_to_color/image_raw"
                    )
            else:
                self.cameras[cam_name] = Camera(f"/{cam_name}/color/image_raw")
                if self._depth:
                    self.depth_cameras[cam_name] = Camera(
                        f"{cam_name}/aligned_depth_to_color/image_raw"
                    )
        # self.scene.add_box('pick_box', make_pose([0.02, 0.0, 0.075],[0, 0, 0, 1],frame_id=ROBOT_BASE_FRAME), size=[0.2 , 0.55, 0.15])
        # Grasped flag
        self._grasped = False
        self._grip_velocity = 2 if open_gripper else -2
        self.reset(open_gripper=open_gripper)

    def reset(self, open_gripper=True):
        self._is_goal_init = False

        self.i_avg = 0
        self.v_avg_record = [np.zeros(6) for _ in range(N_SAMPLES_OVERSHOOT)]

        if open_gripper:
            self._grasped = True
            self.move_gripper("open")
            self._grip_velocity = 2
        else:
            self._grasped = False
            self.move_gripper("close")
            self._grip_velocity = -2

    def eef_pose(self):
        eef_tf = self._eef_tf_recorder.record_tf().transform
        eef_pose = [np.array([
                    eef_tf.translation.x,
                    eef_tf.translation.y,
                    eef_tf.translation.z
                ]),
                np.array([
                    eef_tf.rotation.x,
                    eef_tf.rotation.y,
                    eef_tf.rotation.z,
                    eef_tf.rotation.w
                ])
        ]
        return eef_pose

    def joints_state(self):
        return self.joints_state_recorder.record_state()

    def _limit_pos(self, pos):
        new_position = []
        for i, coord in enumerate(position):
            new_coord = min(max(coord, self.workspace[0][i]), self.workspace[1][i])
            new_position.append(new_coord)
        return new_position

    def move_relative(self, dt, v_xyz, v_rpy):
        # FIXME: https://github.com/ros-planning/moveit/issues/773
        """
        Move the robot relatively to its current position.

        return false if the robot was unable to achieve the motion.
        """

        # Create a left_goal_frame in charge to 'keep' the goal pose relative to the world frame
        # This goal frame prevent any position drift
        if not self._is_goal_init:
            # Robot current pos
            latest_t = self.tf_listener.getLatestCommonTime(ROBOT_BASE_FRAME, self.eef_frame)
            latest_pose = self.tf_listener.lookupTransform(
                ROBOT_BASE_FRAME, self.eef_frame, latest_t
            )
            self.goal_pose = pin.XYZQUATToSE3(latest_pose[0] + latest_pose[1])
            self._is_goal_init = True

        # Update the goal frame (relatively to the world frame)
        self.goal_pose = compute_goal_pose(dt, v_xyz, v_rpy, self.goal_pose)

        # Apply workspace limits
        self.goal_pose.translation = np.array(
            self._limit_position(self.goal_pose.translation)
        )

        # Compute an overshoot of the motion in case of missed deadline
        overshoot_pose = compute_goal_pose(
            OVERSHOOT_FACTOR * dt, v_xyz, v_rpy, self.goal_pose
        )
        overshoot_pose.translation = np.array(
            self._limit_position(overshoot_pose.translation)
        )

        # Transform the poses in world frame
        list_goal_pose = pin.SE3ToXYZQUAT(self.goal_pose)
        list_goal_pose = [list_goal_pose[:3], list_goal_pose[3:]]
        moveit_goal_pose = make_pose(*list_goal_pose)

        list_overshoot_pose = pin.SE3ToXYZQUAT(overshoot_pose)
        list_overshoot_pose = [list_overshoot_pose[:3], list_overshoot_pose[3:]]
        moveit_overshoot_pose = make_pose(*list_overshoot_pose)

        # Compute path
        path, fraction = self.arm.compute_cartesian_path(
            [moveit_overshoot_pose], eef_step=EEF_STEPS, jump_threshold=JUMP_THRESHOLD
        )

        if fraction < 1.0:
            return False, "No cartesian path found"

        # Extract trajectory from planning
        trajectory = path.joint_trajectory
        valid = self.check_jumps(trajectory)

        if not valid:
            raise RuntimeError("No cartesian path found. Exceeded joint velocity threshold")

        delta_t = dt * (1.0 + OVERSHOOT_FACTOR)

        # Keep only first and last point
        trajectory.points = [trajectory.points[0], trajectory.points[-1]]
        trajectory.points[0].time_from_start = rospy.Duration(0)
        trajectory.points[0].velocities = [
            (trajectory.points[1].positions[i] - trajectory.points[0].positions[i])
            / delta_t
            for i in range(len(trajectory.points[0].positions))
        ]
        trajectory.points[0].accelerations = []

        trajectory.points[1].time_from_start = rospy.Duration(delta_t)
        trajectory.points[1].velocities = [
            (trajectory.points[1].positions[i] - trajectory.points[0].positions[i])
            / delta_t
            for i in range(len(trajectory.points[0].positions))
        ]
        trajectory.points[1].accelerations = []

        valid = self.check_jumps(trajectory)

        if not valid:
            raise RuntimeError("No cartesian path found. Exceeded joint velocity threshold")

        # Execute
        self._traj_publisher.publish(trajectory)
        return True, ""

    def check_jumps(self, trajectory):
        num_joints = len(trajectory.points[0].positions)
        for i in range(len(trajectory.points) - 1):
            for j in range(num_joints):
                diff_q = np.abs(
                    trajectory.points[i].positions[j]
                    - trajectory.points[i + 1].positions[j]
                )
                diff_t = (
                    trajectory.points[i + 1].time_from_start.to_sec()
                    - trajectory.points[i].time_from_start.to_sec()
                )
                if diff_q / diff_t > Q_VEL_THRESHOLD:
                    return False
        return True

    def move_gripper(self, state, wait=False):
        if state == "open":
            self._grip_velocity = 2
        else:
            self._grip_velocity = -2

        if self._grasped and state == "close":
            # Return if it was already closed and command to close
            return
        elif not self._grasped and state == "open":
            # Return if it was already open and command to open
            return
        elif self._grasped and state == "open":
            self._grasped = False
        elif not self._grasped and state == "close":
            self._grasped = True

        self.gripper.set_named_target(state)
        self.gripper.go(wait=wait)

    def swap_state(self, wait=True):
        if self._grasped:
            self._grip_velocity = -2
            self._grasped = False
            next_state = "open"
        else:
            self._grasped = True
            self._grip_velocity = 2
            next_state = "close"
        self.gripper.set_named_target(next_state)
        self.gripper.go(wait=wait)

    def set_config(self, q):
        success = self.arm.go(q, wait=True)
        return success

    def _limit_position(self, position):
        new_position = []
        for i, coord in enumerate(position):
            new_coord = min(max(coord, self.workspace[0][i]), self.workspace[1][i])
            new_position.append(new_coord)
        return new_position

    def go_to_pose(self, gripper_pos, gripper_orn, cartesian=True):
        gripper_pos = self._limit_position(gripper_pos)
        gripper_pose = make_pose(gripper_pos, gripper_orn)
        success = False
        if cartesian:
            path, fraction = self.arm.compute_cartesian_path(
                [gripper_pose], eef_step=EEF_STEPS, jump_threshold=JUMP_THRESHOLD
            )

            trajectory = path.joint_trajectory
            valid = self.check_jumps(trajectory)
            if not valid:
                raise Exception("There is a jump in the path!")

            if fraction >= 1.0:
                self.commander.left_arm.execute(path, wait=True)
                success = True
        else:
            self.arm.set_pose_target(gripper_pose)
            success = self.arm.go(wait=True)

        return success
