import argparse
import muse.envs
import robotamer.envs
import gym
import rospy
import matplotlib.pyplot as plt
import numpy as np

from numpy import pi
from muse.core import constants
from muse.envs.utils import realsense_resize_crop
from mujoco_py.modder import TextureModder
from robotamer.core.constants import CAM_INFO
from PIL import Image


def set_colors(scene, modder):
    modder.whiten_materials()
    for name in scene.sim.model.geom_names:
        if (
            "gripper" in name
            or "truss_arm" in name
            or "moment_arm" in name
            or "finger" in name
        ):
            modder.set_rgb(name, (255, 0, 0))
        if "table" in name:
            modder.set_rgb(name, (0, 255, 0))
        if "ft300" in name:
            modder.set_rgb(name, (0, 0, 255))


def main():
    cam_list = list(CAM_INFO.keys())

    # create envs
    sim_env_name = "Pick-v0"
    real_env_name = f"RealRobot-{sim_env_name}"
    sim_env = gym.make(sim_env_name, cam_resolution=constants.REALSENSE_RESOLUTION, cam_crop=False, cam_list=cam_list)
    real_env = gym.make(real_env_name, cam_list=cam_list)

    sim_env.seed(0)
    sim_env.reset()
    real_env.reset()
    scene = sim_env.unwrapped.scene
    texture_modder = TextureModder(scene.sim)
    set_colors(scene, texture_modder)

    # compare real positions with sim
    workspace = real_env.gripper_workspace
    default_orn = [pi, 0, pi / 2]

    for x in range(2):
        for y in range(2):
            x_pos = workspace[x][0]
            y_pos = workspace[y][1]
            z_pos = 0.1
            gripper_pos = [x_pos, y_pos, z_pos]
            real_env.robot.go_to_pose(gripper_pos, default_orn)
            scene.reset(
                mocap_pos=dict(left_gripper=gripper_pos),
                mocap_quat=None,
                open_gripper=dict(left_gripper=True),
                workspace=real_env.unwrapped.workspace,
            )
            scene.warmup()
            ims = []
            for cam_name in cam_list:
                sim_rgb = sim_env.observe()[f"rgb_{cam_name}"]
                im0 = Image.fromarray(sim_rgb)

                # im1 = data[f"{cam_name}_img"]
                im1 = real_env.unwrapped.render()[f"rgb_{cam_name}"]
                # im1 = realsense_resize_crop(im1)
                im1 = Image.fromarray(im1)

                res = Image.blend(im0, im1, 0.5)
                # res = np.vstack((im0, im1))
                ims.append(np.asarray(res))
                plt.imshow(res)
                plt.show()
    if rospy.is_shutdown():
        exit()


if __name__ == "__main__":
    main()
