from math import pi
from robotamer.core.constants import PUSHING_START_CONFIG


def reset_arm(env):
    if env.arm_name == 'left':
        gripper_pos = [-0.40, 0, 0.1]
        gripper_orn = [pi, 0, pi / 2]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
    obs = env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)
    return obs


def reset_joints(env):
    obs = env.reset(joints=PUSHING_START_CONFIG)
    return obs


def reset_to_home(env):
    obs = env.reset(home_only=True)
    return obs


def reset_joints_and_eef(env):
    obs = env.reset(joints=PUSHING_START_CONFIG, gripper_pos=[0.4, 0, 0.04])
    return obs


def reset_eef(env):
    obs = env.reset(gripper_pos=[0.4, 0, 0.04])


def reset_env(env):
   obs = reset_joints(env)
   return obs
