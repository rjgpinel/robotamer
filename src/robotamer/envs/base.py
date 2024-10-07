import functools
import gym
import rospy
import tf2_ros

import numpy as np

from gym import spaces
from collections import deque
from geometry_msgs.msg import Vector3
from math import pi
from gym.utils import seeding
from prl_ur5_demos.utils import make_pose
from robotamer.core.robot import Robot
from robotamer.envs.utils import quat_to_euler
from robotamer.core.constants import (
    REAL_DT,
    SIM_DT,
    CAM_INFO,
    ROBOT_BASE_FRAME,
    EEF_STEPS,
    JUMP_THRESHOLD,
    WORKSPACE,
)
from robotamer.core.tf import depth_to_pcd, pos_euler_to_hom, transform_pcd, project

GRIPPER_HEIGHT_INIT = np.array([0.06, 0.10])

DEFAULT_CONF = {
    "left": [
      0.2792526803190927, 
      -1.2217304763960306,
      1.2042771838760873, 
      -1.5533430342749535, 
      -1.5707963267948966, 
      -1.2915436464758039],
    "right": [
      # Home config at a height of 4.0cm
      1.1697157621383667,
      -1.6209071318255823,
      1.3057317733764648,
      -0.8799679915057581,
      2.284698247909546,
      0.5466184616088867
    ]
}

class BaseEnv(gym.Env):
    def __init__(
        self,
        cam_list=["bravo_camera", "charlie_camera"],
        cam_async=True,
        depth=False,
        pcd=False,
        gripper_attn=False,
        cam_info=None,
        arm="left",
        version="legacy",
        open_gripper=True,
        grip_history_len = 5,
    ):
        rospy.init_node("env_node", log_level=rospy.INFO)

        # Workspace definition
        self.version = version
        self.workspace = WORKSPACE[arm][version]

        self.gripper_workspace = self.workspace.copy()
        self.gripper_workspace[:, 2] = GRIPPER_HEIGHT_INIT

        self.home_config = DEFAULT_CONF[arm]

        self.rate = rospy.Rate(1.0 / REAL_DT)
        self.cam_list = cam_list

        # Controller
        self.robot = Robot(self.workspace, cam_list, cam_async=cam_async, depth=depth, arm=arm,
                           open_gripper=open_gripper)
        self.arm_name = arm

        # Depth flag
        self._depth = depth
        self._pcd = pcd
        self._gripper_attn = gripper_attn
        self._grip_history = deque(maxlen=grip_history_len)

        # Cam info
        self.cam_info = {}
        for cam_name in cam_list:
            if cam_name in CAM_INFO:
                self.cam_info[f"info_{cam_name}"] = CAM_INFO[cam_name]
            self.cam_info[f"intrinsics_{cam_name}"] = (
                self.robot.cameras[cam_name].intrinsics)
        self._np_random = np.random

        self.neutral_gripper_orn = [pi, 0, 0] if arm == "right" else [pi, 0, pi]
        self.safe_height = GRIPPER_HEIGHT_INIT[-1]
        
        self.velocity_subscriber = rospy.Subscriber(
            f"{arm}_eef_velocity", Vector3, self.get_eef_velocity, queue_size=1)
        self.eef_velocity = np.array([0., 0., 0.])

    def seed(self, seed):
        np_random, seed = seeding.np_random(seed)
        self._np_random = np_random

    def sample_random_gripper_pos(self):
        position = self._np_random.uniform(
            self.gripper_workspace[0], self.gripper_workspace[1]
        )
        return position

    def get_current_config(self):
        variables = self.robot.commander.get_current_variable_values()
        config_joints = [f"{self.arm_name}_{k}" for k in [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"]]
        config = [variables[k] for k in config_joints]
        return config
    

    def set_configuration(self, config):
        success = self.robot.set_config(config)
        return success

    def _reset(self, gripper_pos=None, gripper_orn=None, open_gripper=True, joints=None, home_only=False, **kwargs):
        print("Returning to home config")
        config = self.get_current_config()
        diff = np.sum(np.subtract(config, self.home_config))
   
        success = self.robot.set_config(self.home_config) if diff >= 0.001 else True
        if success:
            print("Done")
        else:
            print("Failed to return")

        if home_only:
            return

        if joints is not None:
            print("Setting to custom config")
            success = self.robot.set_config(joints)
            config = self._get_current_config()
            diff = np.subtract(config, joints)

            print("Current config vs. target; difference")
            np.set_printoptions(suppress=True, linewidth=90)
            print(np.array2string(
                np.stack([config, joints, diff]), separator=", "))
            print("EEF pose", self.robot.eef_pose())
            np.set_printoptions(suppress=False, linewidth=75)


        print(quat_to_euler(np.array(self.robot.eef_pose()[1]), False))
        if gripper_pos is not None or gripper_orn is not None:
            if gripper_pos is None:
                gripper_pos = self.robot.eef_pose()[0]
                # gripper_pos = self.sample_random_gripper_pos()

            if gripper_orn is None:
                gripper_orn = self.neutral_gripper_orn

            print("Moving to cartesian pose", gripper_pos, gripper_orn)
            success = self.robot.go_to_pose(gripper_pos, gripper_orn, cartesian=True)
        
        # Random gripper pose
        # else:
        #    gripper_pos = self.sample_random_gripper_pos()
        #    gripper_orn = self.neutral_gripper_orn
        #    success = self.robot.go_to_pose(gripper_pos, gripper_orn, cartesian=True)



        if not success:
            raise RuntimeError("Moving the robot to default position failed")

        self.robot.reset(open_gripper=open_gripper)

    def stop_current_movement(self):
        try:
            for _ in range(3):
                self.step({"linear_velocity": np.array([0., 0., 0.]),
                           "angular_velocity": np.array([0., 0., 0.])})
        except tf2_ros.TransformException:
            pass

    def reset(self, **kwargs):
        self.stop_current_movement()
        self._reset(**kwargs)
        return self.render()

    def step(self, action):
        processed_action = {}

        if "grip_open" in action.keys():
            # Move the gripper
            grip_open = action["grip_open"]
            self._grip_history.append(grip_open)
            grip_open_mean = np.mean(self._grip_history)

            if grip_open_mean >= 0:
                self.robot.move_gripper("open", wait=True)
            else:
                self.robot.move_gripper("close", wait=True)

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

        no_render = True
        self.rate.sleep()
        if not no_render:
            obs = self.render()
        else:
            obs = None

        # Default real robot outputs
        success = False
        reward = 0

        return (
            obs,
            reward,
            success,
            {"success": success, "failure_message": msg},
        )

    def get_eef_velocity(self, vel):
        self.eef_velocity = np.array([vel.x, vel.y, vel.z])

    def render(self, sync_record=False, **unused_kwargs):
        obs = {}

        gripper_pose = self.robot.eef_pose()
        obs["gripper_pos"] = np.array(gripper_pose[0])
        obs["gripper_quat"] = np.array(gripper_pose[1])
        obs["gripper_trans_velocity"] = np.array(self.eef_velocity)
        obs["gripper_theta"] = quat_to_euler(np.array(gripper_pose[1]), False)[-1]
        obs["grip_velocity"] = self.robot._grip_velocity
        obs["gripper_state"] = self.robot._grasped

        # Sensors
        for cam_name in self.robot.cam_list:
            cam = self.robot.cameras[cam_name]

            if not sync_record:
                obs[f"rgb_{cam_name}"], cam_pose = cam.record_image(dtype=np.uint8)
            else:
                obs[f"rgb_{cam_name}"], cam_pose = cam.record_image_sync(dtype=np.uint8)

            if f"info_{cam_name}" in self.cam_info:
                obs[f"info_{cam_name}"] = self.cam_info[f"info_{cam_name}"]
                obs[f"info_{cam_name}"]["pos"] = CAM_INFO[cam_name]["pos"]
                obs[f"info_{cam_name}"]["euler"] = CAM_INFO[cam_name]["euler"]
            else:
                obs[f"info_{cam_name}"] = dict()
                obs[f"info_{cam_name}"]["pos"] = cam_pose[0]
                obs[f"info_{cam_name}"]["euler"] = cam_pose[1]

            if self._depth:
                depth_cam = self.robot.depth_cameras[cam_name]
                if not sync_record:
                    depth, _ = depth_cam.record_image(dtype=np.uint16)
                else:
                    depth, _ = depth_cam.record_image_sync(dtype=np.uint16)
                depth = depth.astype(np.float32).squeeze(-1) / 1000
                obs[f"depth_{cam_name}"] = depth

                if self._pcd:
                    info_cam = obs[f"info_{cam_name}"]
                    cam_pos = info_cam["pos"]
                    cam_euler = info_cam["euler"]
                    world_T_cam = pos_euler_to_hom(cam_pos, cam_euler)
                    intrinsics = self.cam_info[f"intrinsics_{cam_name}"]
                    pcd = depth_to_pcd(depth, intrinsics)
                    pcd = transform_pcd(pcd, world_T_cam)
                    obs[f"pcd_{cam_name}"] = pcd

            if self._gripper_attn:
                K = self.cam_info[f"intrinsics_{cam_name}"]["K"]
                gr_px = project(gripper_pose, np.linalg.inv(world_T_cam), K)
                gr_x, gr_y = gr_px[0], gr_px[1]
                obs[f"gripper_uv_{cam_name}"] = [gr_x, gr_y]

            obs["robot_info"] = self.robot.links_pose()
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

    def move(self, gripper_pos, gripper_quat=None, open_gripper=True, cartesian=True, only_cartesian=True):

        gripper_pos = gripper_pos.astype(np.double)
        # gripper_quat = None

        if gripper_quat is not None:
            gripper_orn = quat_to_euler(gripper_quat, False)
        else:
            gripper_orn = self.neutral_gripper_orn
        success = self.robot.go_to_pose(gripper_pos, gripper_orn, cartesian=cartesian, only_cartesian=only_cartesian)

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
