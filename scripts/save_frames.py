import time
import robotamer.envs
import gym
import rospy
import torch
import numpy as np
from torchvision.utils import save_image
import ibc.utils.torch as ptu
from numpy import pi

import matplotlib.pyplot as plt

from PIL import Image
from robotamer.core.constants import SIM_DT
from muse.envs.utils import realsense_resize_crop, realsense_resize_batch_crop
from muse.core.constants import REALSENSE_CROP, REALSENSE_RESOLUTION


def main():
    env = gym.make("RealRobot-NutAssembly-v0", cam_list=["charlie_camera", "bravo_camera"])
    # env.reset()
    obs = env.unwrapped.render()

    cam_list = ["bravo_camera", "charlie_camera"]
    num_streams = len(cam_list)
    max_frame_idx = 0


    render_w, render_h = REALSENSE_RESOLUTION
    W, H = REALSENSE_CROP
    frames = torch.zeros(
        (num_streams, max_frame_idx + 1, H, W, 3), dtype=torch.float32, device=ptu.device
    )
    step_frames = torch.zeros((num_streams, render_h, render_w, 3), dtype=torch.uint8)


    frames[:, :-1] = frames[:, 1:].clone()
    for cam_i, cam_name in enumerate(cam_list):
        step_frames[cam_i, :, :, :] = torch.from_numpy(obs[f"rgb_{cam_name}"])
    frames[:, -1, :, :, :], t = realsense_resize_batch_crop(
        step_frames.to(ptu.device)
    )
    im = Image.fromarray(frames[0, 0].numpy().astype(np.uint8))
    im.save('/home/rgarciap/Desktop/robot_bravo.png')
    im2 = Image.fromarray(frames[1, 0].numpy().astype(np.uint8))
    im2.save('/home/rgarciap/Desktop/robot_charlie.png')

if __name__ == "__main__":
    main()
