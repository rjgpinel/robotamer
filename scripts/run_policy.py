import argparse
import torch
import rospy
import skvideo.io
import warnings

import numpy as np
import ibc.utils.torch as ptu

from collections import deque
from einops import rearrange
from pathlib import Path

from ibc.utils.attention import AttentionHook
from ibc.data.factory import create_transform
from ibc.data.utils import (
    get_im_norm_stats,
    normalize_frames,
    denormalize_frame,
    state_to_frames,
)
from ibc.model.factory import create_vision_model, create_bc_model

from muse.envs.utils import realsense_resize_crop, realsense_resize_batch_crop
from muse.core.constants import REALSENSE_CROP, REALSENSE_RESOLUTION

from prl_ur5_diffsim2real.pick import PickEnv

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("BC policy running script", add_help=False)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--max-steps", default=1200, type=int)
    parser.add_argument("--cam-list", default="", type=str)
    parser.add_argument("--record", action="store_true", help="Record policy running")
    parser.add_argument("--att-maps", action="store_true", help="Attention maps")
    parser.set_defaults(record=False, att_maps=False)
    return parser


def main(args):
    ptu.set_gpu_mode(True)

    # checkpoint loading
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    delay_hist = checkpoint["args"].delay_hist
    frame_hist = checkpoint["args"].frame_hist
    num_frames = len(frame_hist) + 1
    if not args.cam_list:
        args.cam_list = checkpoint["args"].cam_list
    else:
        args.cam_list = args.cam_list.split(",")

    assert len(args.cam_list) == len(checkpoint["args"].cam_list)

    num_streams = len(args.cam_list)

    # data processing definition
    normalization = "vit" if "vit" in checkpoint["args"].model else "resnet"
    norm_img = get_im_norm_stats(normalization)
    transform = create_transform(
        normalization=normalization,
        data_aug="",
    )

    # model
    vision_model = create_vision_model(
        checkpoint["args"].model,
        num_frames=num_frames,
        num_streams=len(args.cam_list),
        stream_integration=checkpoint["args"].stream_integration,
    )

    model = create_bc_model(
        vision_model=vision_model,
        data_stats=checkpoint["stats"],
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(ptu.device)

    # logging
    if args.record:
        video_writer = skvideo.io.FFmpegWriter(f"./videos/{int(time())}.mp4")
        if args.att_maps:
            att_hook = AttentionHook(model)

    # reset the agent before any experiment
    env = PickEnv(args.cam_list)
    obs = env.reset()

    traj = []

    render_w, render_h = REALSENSE_RESOLUTION
    W, H = REALSENSE_CROP
    frames = torch.zeros(
        (num_streams, num_frames, H, W, 3), dtype=torch.float32, device=ptu.device
    )

    step_frames = torch.zeros((num_streams, render_h, render_w, 3), dtype=torch.uint8)

    for i in range(args.max_steps):
        if args.record:
            record_frames = []

        # shift frames to the left
        frames[:, :-1] = frames[:, 1:].clone()
        for cam_i, cam_name in enumerate(args.cam_list):
            step_frames[cam_i, :, :, :] = torch.from_numpy(obs[f"rgb_{cam_name}"])
        frames[:, -1, :, :, :], t = realsense_resize_batch_crop(
            step_frames.to(ptu.device)
        )
        if i == 0:
            frames[:, :-1] = frames[:, -1].unsqueeze(1)
        frames_norm = frames.unsqueeze(0) / 255
        frames_norm = transform(frames_norm)

        if rospy.is_shutdown():
            break

        # Compute action
        with torch.no_grad():
            action = model(frames_norm)

        # Post-process action
        action = model.process_output(action)
        for k, v in action.items():
            if type(v) is not np.ndarray:
                action[k] = v.cpu().detach().numpy()

        action["linear_velocity"] = action["linear_velocity"].squeeze(0)
        if action["linear_velocity"].shape[0] == 2:
            linear_vel = np.zeros(3)
            linear_vel[:2] = action["linear_velocity"]
            action["linear_velocity"] = linear_vel

        if "angular_velocity" not in action.keys():
            action["angular_velocity"] = np.zeros_like(action["linear_velocity"])
            if "theta_velocity" in action.keys():
                action["angular_velocity"][-1] = action["theta_velocity"]
        else:
            action["angular_velocity"] = action["angular_velocity"].squeeze(0)

        obs, _, done, _ = env.step(action)
        if args.record:
            video_writer.close()

        if rospy.is_shutdown():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BC policy running script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise ValueError(f"Checkpoint directory {args.checkpoint} does not exist.")

    try:
        main(args)
    except rospy.ROSInterruptException:
        pass
