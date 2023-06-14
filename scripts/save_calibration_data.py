import argparse
import robotamer.envs
import gym
import rospy
import pickle as pkl

from numpy import pi
from robotamer.core.constants import CAM_INFO

def main():
    # create envs
    cam_list = list(CAM_INFO.keys())

    real_env_name = "RealRobot-Pick-v0"
    real_env = gym.make(real_env_name, cam_list=cam_list)

    workspace = real_env.gripper_workspace
    default_orn = [pi, 0, pi / 2]

    for x in range(2):
        for y in range(2):
            x_pos = workspace[x][0]
            y_pos = workspace[y][1]
            z_pos = 0.1
            real_env.robot.go_to_pose([x_pos, y_pos, z_pos], default_orn)
            obs = real_env.unwrapped.render()
            with open(f"calibration/{x*2 + y}.pkl", "wb") as f:
                pkl.dump(obs, f)


    if rospy.is_shutdown():
        exit()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
