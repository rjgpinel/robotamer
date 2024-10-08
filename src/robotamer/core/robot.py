import moveit_commander
import pinocchio as pin
import rospy
import sys
import tf

import numpy as np

from moveit_msgs.msg import RobotState
from prl_ur5_demos.utils import make_pose
from robotamer.core.observer import CameraPose, CameraAsyncPose, TFRecorder, JointStateRecorder
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
    CAM_TF_TOPIC,
    ROBOT_LINKS
)
from robotamer.core.utils import compute_goal_pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import  GripperCommandActionGoal
from geometry_msgs.msg import PoseStamped


class Robot:
    def __init__(self, workspace, cam_list, cam_async=True, depth=False, arm='left', open_gripper=True):
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


        # Gripper Publisher
        self._gripper_publisher = rospy.Publisher(
            f"/{arm}_gripper/gripper_controller/gripper_cmd/goal",
            GripperCommandActionGoal,
            queue_size=1,
        )

        # Transformations
        self.tf_listener = tf.TransformListener()
        self.tf_brodcaster = tf.TransformBroadcaster()
        # EEF transformation recorder
        self._eef_tf_recorder = TFRecorder(ROBOT_BASE_FRAME, self.eef_frame)
        # Links transformation recorder
        self.robot_links = ROBOT_LINKS[arm]
        self._links_tf_recorder = {link: TFRecorder(ROBOT_BASE_FRAME, link) for link in self.robot_links}

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
                self.cameras[cam_name] = CameraAsyncPose(f"/{cam_name}/color/image_raw", CAM_TF_TOPIC[cam_name])
                if self._depth:
                    self.depth_cameras[cam_name] = CameraAsyncPose(
                        f"{cam_name}/aligned_depth_to_color/image_raw", CAM_TF_TOPIC[cam_name]
                    )
            else:
                self.cameras[cam_name] = CameraPose(f"/{cam_name}/color/image_raw", CAM_TF_TOPIC[cam_name])
                if self._depth:
                    self.depth_cameras[cam_name] = CameraPose(
                        f"{cam_name}/aligned_depth_to_color/image_raw", CAM_TF_TOPIC[cam_name]
                    )

        # Grasped flag
        self.box_name = None
        self.box_name_def = "box"
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
            self.remove_gripper_box()
        else:
            self._grasped = False
            self.move_gripper("close")
            self._grip_velocity = -2
            self.add_gripper_box()

    def add_gripper_box(self):
        return
        print("Add box")
        if not self.box_name or self.box_name_def not in self.scene.get_known_object_names():
            box_pose = PoseStamped()
            box_pose.header.frame_id = self.eef_frame
            box_pose.pose.orientation.w = 1.0
            self.box_name = self.box_name_def
            self.scene.add_box(self.box_name, box_pose, size=(0.025, 0.025, 0.025))
            start = rospy.get_time()
            seconds = rospy.get_time()
            timeout = 2
            while (seconds - start < timeout) and not rospy.is_shutdown():
                # Test if the box is in attached objects
                attached_objects = self.scene.get_attached_objects([self.box_name])
                is_attached = len(attached_objects.keys()) > 0
                # Test if the box is in the scene.
                # Note that attaching the box will remove it from known_objects
                is_known = self.box_name in self.scene.get_known_object_names()

                # Test if we are in the expected state
                if is_attached and is_known:
                    break

                # Sleep so that we give other threads time on the processor
                rospy.sleep(0.05)
                seconds = rospy.get_time()


            touch_links = [
                        #    f"{self.arm_name}_gripper_finger_1_origin", 
                           f"{self.arm_name}_gripper_finger_1_truss_arm", 
                           f"{self.arm_name}_gripper_finger_1_finger_tip", 
                           f"{self.arm_name}_gripper_finger_1_flex_finger", 
                           f"{self.arm_name}_gripper_finger_1_safety_shield",
                           f"{self.arm_name}_gripper_finger_1_moment_arm",
                           f"{self.arm_name}_gripper_finger_2_origin", 
                           f"{self.arm_name}_gripper_finger_2_truss_arm", 
                           f"{self.arm_name}_gripper_finger_2_finger_tip", 
                           f"{self.arm_name}_gripper_finger_2_flex_finger", 
                           f"{self.arm_name}_gripper_finger_2_safety_shield",
                           f"{self.arm_name}_gripper_finger_2_moment_arm",
                           f"{self.arm_name}_gripper_grasp_frame"
                           ]
            print(touch_links)
            self.scene.attach_box(self.eef_frame, self.box_name, touch_links=touch_links)

    
    def remove_gripper_box(self):
        return
        if self.box_name or self.box_name_def in self.scene.get_known_object_names():
            print("Remove box")
            self.scene.remove_attached_object(self.eef_frame, name=self.box_name_def)
            self.scene.remove_world_object(self.box_name_def)
            self.box_name = None

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
    
    def links_pose(self):
        links_poses = {}
        for link_name, link_tf_recorder in self._links_tf_recorder.items():
            link_tf = link_tf_recorder.record_tf().transform
            link_pose = np.array([
                link_tf.translation.x,
                link_tf.translation.y,
                link_tf.translation.z,
                link_tf.rotation.x,
                link_tf.rotation.y,
                link_tf.rotation.z,
                link_tf.rotation.w
            ])
            
            links_poses[link_name] = link_pose
        return links_poses
        

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

    def move_gripper(self, state, wait=True):

        command = GripperCommandActionGoal()

        if self._grasped and state == "close":
            # Return if it was already closed and command to close
            return
        elif not self._grasped and state == "open":
            # Return if it was already open and command to open
            return
        elif self._grasped and state == "open":
            self._grasped = False
            command.goal.command.position = 0.0
            self.remove_gripper_box()
        elif not self._grasped and state == "close":
            self._grasped = True
            command.goal.command.position = 1.0
            self.add_gripper_box()
        # self._gripper_publisher.publish(command)
        self.gripper.set_named_target(state)
        success = self.gripper.go(wait=wait)
        if success:
            if state == "open":
                self._grip_velocity = 2
            else:
                self._grip_velocity = -2
        # rospy.sleep(6)

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
        success = self.gripper.go(wait=wait)
        if success:
            if self._grasped:
                self._grip_velocity = -2
                self._grasped = False
            else:
                self._grasped = True
                self._grip_velocity = 2

            

    def set_config(self, q):
        success = self.arm.go(q, wait=True)
        self._is_goal_init = False
        return success

    def _limit_position(self, position):
        new_position = []
        for i, coord in enumerate(position):
            new_coord = min(max(coord, self.workspace[0][i]), self.workspace[1][i])
            new_position.append(new_coord)
        return new_position

    def go_to_pose(self, gripper_pos, gripper_orn, cartesian=True, only_cartesian=True):
        gripper_pos = self._limit_position(gripper_pos)
        gripper_pose = make_pose(gripper_pos, gripper_orn)
        success = False
        if cartesian:
            for i in range(10):
                print(f"Trying cartesian path {i}")
                path, fraction = self.arm.compute_cartesian_path(
                    [gripper_pose], eef_step=EEF_STEPS, jump_threshold=JUMP_THRESHOLD
                )

                trajectory = path.joint_trajectory
                valid = self.check_jumps(trajectory)
                if not valid:
                    continue

                if fraction >= 1.0:
                    self.commander.left_arm.execute(path, wait=True)
                    success = True
                    break

            if (not valid or fraction < 1.0) and not only_cartesian:
                no_cartesian = input("sure to run non cartesian?")
                if not no_cartesian:
                    return False
                for i in range(2):
                    self.arm.set_pose_target(gripper_pose)
                    success = self.arm.go(wait=True)
                    if success: 
                        break
        else:
            for i in range(10):
                self.arm.set_pose_target(gripper_pose)
                success = self.arm.go(wait=True)
                if success:
                    break

        self._is_goal_init = False

        return success
