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

from robotamer.core.utils import resize, crop_center

warnings.filterwarnings("ignore")


class Arguments(tap.Tap):
    exp_config: str
    device: str = "cuda"
    num_demos: int = 10
    image_size: int = 128
    cam_list: List[str] = ["bravo_camera","charlie_camera"]
    arm: str = "left"
    env_name: str = "RealRobot-Pick-v0"
    hiveformer: bool = False
    num_steps: int = 7
    save_obs_outs_dir: str = None
    checkpoint: str = None

def process_keystep(obs, cam_list=["bravo_camera", "charlie_camera"], crop_size=None, hiveformer=True):
    rgb = []
    gripper_uv = {}
    pc = []
    gripper_pos = obs["gripper_pos"]
    gripper_quat = obs["gripper_quat"]
    gripper_pose = np.concatenate([gripper_pos, gripper_quat])
    for cam_name in cam_list:
        rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
        pc.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))
        gripper_uv[cam_name] = obs[f"gripper_uv_{cam_name}"]


    rgb = torch.stack(rgb) 
    pc = torch.stack(pc) 
        
    if crop_size:
        rgb = rgb.permute(0, 3, 1, 2)
        pc = pc.permute(0, 3, 1, 2)

        rgb, ratio = resize(rgb, crop_size, im_type="rgb")
        pc, ratio = resize(pc, crop_size, im_type="pc")
        rgb, start_x, start_y = crop_center(rgb, crop_size, crop_size)
        pc, start_x, start_y = crop_center(pc, crop_size, crop_size)
        rgb = rgb.permute(0, 2, 3, 1)
        pc = pc.permute(0, 2, 3, 1)
        for cam_name, uv in gripper_uv.items():
            gripper_uv[cam_name] = [int(uv[0]*ratio) - start_x, int(uv[1]*ratio) - start_y]


    im_height, im_width = rgb.shape[1], rgb.shape[2]
    gripper_imgs = np.zeros(
            (len(cam_list), 1, im_height, im_width), dtype=np.float32
            )
    for i, cam in enumerate(cam_list):
        u, v = gripper_uv[cam]
        if u > 0 and u < im_width and v > 0 and v < im_height:
            gripper_imgs[i, 0, v, u] = 1

    if hiveformer:
        print("FIXME! RGB float preprocessing")
        keystep = {"rgb": rgb.float().numpy(),
                "pc": pc.float().numpy(),
                "gripper_imgs": gripper_imgs,
                "gripper": gripper_pose}
    else:
        keystep = {"rgb": rgb.numpy().astype(np.uint8),
                "pc": pc.float().numpy(),
                "gripper_imgs": gripper_imgs,
                "gripper": gripper_pose}
                
    return keystep



def main(args):
    #FIXME: Update code to be installed as vlc_rlbench
    if args.hiveformer:
        print("Running hiveformer!")
        code_dir = '/home/rgarciap/Code/vlc_rlbench'
    else:
        print("Running PolarNet!")
        code_dir = '/home/rgarciap/Code/work_csz/vlc_rlbench'
    
    sys.path.insert(0, code_dir)
    from eval_models import Actioner

    actioner = Actioner(args)

    # Create env
    env = gym.make('RealRobot-Pick-v0',
            cam_list=args.cam_list,
            arm=args.arm,
            version="v0",
            depth=True,
            pcd=True,
            gripper_attn=True)

    obs = env.reset() 

    keystep = process_keystep(obs, cam_list=args.cam_list, crop_size=args.image_size, hiveformer=args.hiveformer)
    step_id = 0
    taskvar_id = 0

    actioner.reset("push_buttons", 7, None, None)

    for step_id in range(args.num_steps):
        action = actioner.predict(taskvar_id, step_id, keystep)
        action = action['action']

        pos = action[:3]
        quat = action[3:7]
        open_gripper = action[7] <= 0.5

        # print("Position:", pos)
        # print("Quaternion:", quat)
        # print("Open gripper:", open_gripper)

        obs, _, _, _ = env.move(pos, quat, open_gripper)
        # return

        keystep = process_keystep(obs, cam_list=args.cam_list, crop_size=args.image_size, hiveformer=args.hiveformer)

        if rospy.is_shutdown():
            break

    gripper_pos = obs["gripper_pos"]
    gripper_pos[-1] += 0.1
    gripper_state = obs["gripper_state"]
    env.move(gripper_pos, open_gripper=gripper_state)


if __name__ == "__main__":
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    if not Path(args.exp_config).exists():
        raise ValueError(f"Config path {args.exp_config} does not exist.")

    try:
        main(args)
    except rospy.ROSInterruptException:
        pass
