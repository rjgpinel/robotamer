import numpy as np


# Speed integration period
REAL_DT = 0.2
SIM_DT = 0.2

# Cameras configuration
BRAVO_INFO = {
    "pos": np.array([-0.486534, 0.70895, 0.398287]),
    "euler": np.array([-1.931, 0.014, -3.121]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}
CHARLIE_INFO = {
    "pos": np.array([ -1.199518, 0.003458, 0.400626]),
    "euler": np.array([-2.095, -0.006, -1.550]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}

CAM_INFO = {"bravo_camera": BRAVO_INFO, "charlie_camera": CHARLIE_INFO}

# CAM_TF_TOPIC = {"bravo_camera": "bravo_camera_calibration_pose", "charlie_camera": "charlie_camera_calibration_pose", "left_camera": "left_camera_color_optical_frame"}
CAM_TF_TOPIC = {"bravo_camera": "bravo_camera_color_optical_frame", "charlie_camera": "charlie_camera_color_optical_frame", "left_camera": "left_camera_color_optical_frame"}

# Controller definition
EEF_FRAME = {"left": "left_gripper_grasp_frame", "right": "right_gripper_grasp_frame"}
ROBOT_BASE_FRAME = "prl_ur5_base"
OVERSHOOT_FACTOR = 1.0  # Set to 0.0 for no overshoot
N_SAMPLES_OVERSHOOT = 1
MAX_VELOCITY_SCALING_FACTOR = 0.2
MAX_ACCELERATION_SCALING_FACTOR = 0.2
# PLANNING_TIME = 20.0
PLANNING_TIME = 2.0
# PLANNING_TIME = 10.0
COMMAND_ROS_TOPIC = {"left": "/left_arm/scaled_pos_joint_traj_controller/command", "right": "/right_arm/scaled_pos_joint_traj_controller/command"}
# COMMAND_ROS_TOPIC = "/left_arm/scaled_vel_joint_traj_controller/command" # Velocity controller
EEF_STEPS = 0.01
JUMP_THRESHOLD = 0.0
JOINTS_STATE_TOPIC = "/joint_states"
Q_VEL_THRESHOLD = 2  # rad/s


TASK_START_CONFIG = {
    "right": {
        # Center-right of the table. (near 0.4, 0, 0.04)
        # Obtained from the real robot, 4cm from the table.
        "v0": [
            0.8707625865936279, -1.7185638586627405, 1.6314215660095215, -0.9090965429889124, 2.146097183227539, 0.8783294558525085
        ],
        # "v1": [
        #     0.8707625865936279, -1.7185638586627405, 1.6314215660095215, -0.9090965429889124, 2.146097183227539, 0.8783294558525085
        # ]
    },
    "left": {
        "v0": [
            -1.0338895956622522, -1.3494389692889612, -1.8950207869159144, -1.932540241871969, -2.2173264662372034, -2.266449276600973
        ]
    }
}

WORKSPACE = {
    # TODO: Define overlapping workspaces in v1.
    "left": {"v0": np.array([[-0.695, -0.16, 0.01], [-0.295, 0.175, 0.2]]),
             "v1": np.array([[-0.695, -0.16, 0.01], [-0.295, 0.175, 0.2]]),
             "legacy": np.array([[-0.695, -0.175, 0.02], [-0.295, 0.175, 0.75]]),
             "wide": np.array([[-1, -0.175, 0.01], [0.0, 0.40, 0.75]]),
             "legacy-s2r": np.array([[-0.695, -0.175, 0.01], [-0.295, 0.175, 0.2]])},
    "right": {"v0": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]]),
              "v1": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]])},
}

