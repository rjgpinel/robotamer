import numpy as np


# Speed integration period
REAL_DT = 0.2
SIM_DT = 0.2

# Cameras configuration
BRAVO_INFO = {
    "pos": np.array([-0.494061, 0.692729, 0.400215]),
    "euler": np.array([1.03091702, 0.00556305, -3.11407431]),
    "fovy": 42.5,
}
CHARLIE_INFO = {
    "pos": np.array([-1.201099, 0.005, 0.403127]),
    "euler": np.array([1.04368278, -0.00250582, -1.56810664]),
    "fovy": 42.5,
}
CAM_INFO = {"bravo_camera": BRAVO_INFO, "charlie_camera": CHARLIE_INFO}

# Controller definition
EEF_FRAME = {"left": "left_gripper_grasp_frame", "right": "right_gripper_grasp_frame"}
ROBOT_BASE_FRAME = "prl_ur5_base"
OVERSHOOT_FACTOR = 1.0  # Set to 0.0 for no overshoot
N_SAMPLES_OVERSHOOT = 1
MAX_VELOCITY_SCALING_FACTOR = 0.2
MAX_ACCELERATION_SCALING_FACTOR = 0.2
PLANNING_TIME = 2.0
COMMAND_ROS_TOPIC = {"left": "/left_arm/scaled_pos_joint_traj_controller/command", "right": "/right_arm/scaled_pos_joint_traj_controller/command"}
# COMMAND_ROS_TOPIC = "/left_arm/scaled_vel_joint_traj_controller/command" # Velocity controller
EEF_STEPS = 0.01
JUMP_THRESHOLD = 0.0
JOINTS_STATE_TOPIC = "/joint_states"
Q_VEL_THRESHOLD = 2  # rad/s

PUSHING_START_CONFIG = [
    # Obtained from the real robot, 4cm from the table.
    0.8707625865936279, -1.7185638586627405, 1.6314215660095215, -0.9090965429889124, 2.146097183227539, 0.8783294558525085
]
