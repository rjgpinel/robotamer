import torch
import rospy
import warnings

import tap
import os

import sys

import robotamer.envs

import gym
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
from copy import deepcopy

from robotamer.core.utils import resize, crop_center
from robotamer.envs.utils import quat_to_euler

warnings.filterwarnings("ignore")


def main():
    # Create env
    env = gym.make('RealRobot-Pick-v0',
            cam_list="",
            arm="left",
            version="v0",
            depth=True,
            pcd=True,
            gripper_attn=True)

    print("Current configuraiton:", env.get_current_config())
    print("Current robot pose: ", env.robot.eef_pose())



if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
