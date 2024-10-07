import numpy as np


# Speed integration period
REAL_DT = 0.2
SIM_DT = 0.2

# Cameras configuration
BRAVO_INFO = {
    "pos": np.array([-0.192736, 0.835309,0.700992]),
    "euler": np.array([-2.227, 0.011, 3.142]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}

CHARLIE_INFO = {
    "pos": np.array([-0.670188, -0.558114, 0.597616]),
    "euler": np.array([-2.355, -0.100,  -0.517]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}

ALPHA_INFO = {
    "pos": np.array([0.131421, -0.626980, 0.818875]),
    "euler": np.array([-2.467, 0.026, 0.475]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}

CAM_INFO = {"bravo_camera": BRAVO_INFO, "charlie_camera": CHARLIE_INFO, "alpha_camera": ALPHA_INFO}

# CAM_TF_TOPIC = {"bravo_camera": "bravo_camera_calibration_pose", "charlie_camera": "charlie_camera_calibration_pose", "left_camera": "left_camera_color_optical_frame"}
CAM_TF_TOPIC = {"bravo_camera": "bravo_camera_color_optical_frame", "charlie_camera": "charlie_camera_color_optical_frame", "alpha_camera": "alpha_camera_color_optical_frame"}

# Controller definition
EEF_FRAME = {"left": "left_gripper_grasp_frame", "right": "right_gripper_grasp_frame"}
ROBOT_LINKS = {"left": ["left_base_link", 
                        "left_shoulder_link", 
                        "left_upper_arm_link", 
                        "left_forearm_link", 
                        "left_wrist_1_link", 
                        "left_wrist_2_link", 
                        "left_wrist_3_link", 
                        "left_ft300_mounting_plate", 
                        "left_ft300_sensor", 
                        "left_camera_link", 
                        "left_gripper_body", 
                        "left_gripper_bracket", 
                        "left_gripper_finger_1_finger_tip", 
                        "left_gripper_finger_1_flex_finger", 
                        "left_gripper_finger_1_safety_shield",
                        "left_gripper_finger_1_truss_arm",
                        "left_gripper_finger_1_moment_arm",
                        "left_gripper_finger_2_finger_tip",
                        "left_gripper_finger_2_flex_finger", 
                        "left_gripper_finger_2_safety_shield",
                        "left_gripper_finger_2_truss_arm",
                        "left_gripper_finger_2_moment_arm",
                        ], 
                        "right": ["right_base_link"]}
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
        # "v0": [
        #     -1.0338895956622522, -1.3494389692889612, -1.8950207869159144, -1.932540241871969, -2.2173264662372034, -2.266449276600973
        # ],
        "v0": list(np.pi*np.array([
            18, -69, 67, -87, -94, 17
        ])/180)
    }
}

WORKSPACE = {
    # TODO: Define overlapping workspaces in v1.
    "left": {"v0": np.array([[-0.695, -0.16, 0.01], [-0.295, 0.175, 0.2]]),
             "v1": np.array([[-0.695, -0.16, 0.01], [-0.295, 0.175, 0.2]]),
             "legacy": np.array([[-0.695, -0.175, 0.02], [-0.295, 0.175, 0.75]]),
             "wide": np.array([[-0.50, -0.54, 0.01], [0.0, 0.54, 0.75]]),
             "legacy-s2r": np.array([[-0.695, -0.175, 0.01], [-0.295, 0.175, 0.2]])},
    "right": {"v0": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]]),
              "v1": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]])},
}

