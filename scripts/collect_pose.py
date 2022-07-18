#!/usr/bin/env python3
import rospkg
import rospy
import numpy as np
import pickle as pkl
import muse.envs
import robotamer.envs
import gym

from math import pi
from pathlib import Path
from tqdm import tqdm
from yaml import load
from robotamer.core.tf import pos_quat_to_hom

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_args_parser():
    parser = argparse.ArgumentParser("Pose data collector script", add_help=False)
    # Directory to save trajectory
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument(
        "--poses",
        default=1000,
        type=int,
        help="Number of poses to collect",
    )
    parser.add_argument("--init-seed", default=50000, type=int)
    parser.add_argument("--cam-list", default="", type=str)
    parser.add_argument("--env-name", default="Pick-v0", type=str)
    return parser


def compute_target_pos(obs):
    gripper_pos, gripper_quat = obs["gripper_pos"], obs["gripper_quat"]
    cube_pos, cube_quat = obs["cube0_pos"], [0, 0, 0, 1]

    world_T_target = pos_quat_to_hom(cube_pos, cube_quat)
    world_T_gripper = pos_quat_to_hom(gripper_pos, gripper_quat)
    gripper_T_world = np.linalg.inv(world_T_gripper)
    gripper_T_target = np.matmul(gripper_T_world, world_T_target)
    target_pos = hom_to_pos(gripper_T_target)

    return target_pos


def main(args):
    output_dir = Path(args.output_dir)
    real_output_dir = outpur_dir / "real"
    sim_output_dir = outpur_dir / "sim"

    sim_env_name = args.env_name
    sim_env = gym.make(sim_env_name)

    # define cameras
    if args.cam_list:
        cam_list = args.cam_list.split(",")
    else:
        cam_list = sim_env.unwrapped.cam_list

    real_env_name = f"RealRobot-{sim_env_name}"
    real_env = gym.make(real_env_name, cam_list=cam_list)

    stats = {
        "cam_list": cam_list,
        "gripper_pos": [],
        "gripper_quat": [],
        "target_pos": [],
    }

    dataset = []
    try:
        for seed in tqdm(range(args.init_seed, args.init_seed + args.poses)):
            sim_env.seed(seed)
            sim_obs = sim_env.reset()
            target_pos = compute_target_pos(sim_obs)
            gripper_pos = sim_obs["gripper_pos"]
            gripper_quat = sim_obs["gripper_quat"]

            stats["gripper_pos"].append(gripper_pos)
            stats["gripper_quat"].append(gripper_quat)
            stats["target_pos"].append(target_pos)

            sim_processed_obs = dict(
                gripper_pos=gripper_pos,
                gripper_quat=gripper_quat,
            )

            for cam_name in cam_list:
                sim_processed_obs[f"rgb_{cam_name}"] = sim_obs[f"rgb_{cam_name}"]

            set_success = real_env.set_scene(sim_env)
            real_obs = real_env.reset(gripper_pos=gripper_pos)

            real_processed_obs = dict(
                gripper_pos=gripper_pos,
                gripper_quat=gripper_quat,
            )

            for cam_name in cam_list:
                real_processed_obs[f"rgb_{cam_name}"] = real_obs[f"rgb_{cam_name}"]

            with open(str(real_output_path / f"{seed:07d}.pkl"), "wb") as f:
                pkl.dump((real_processed_obs, target_pos), f)

            with open(str(sim_output_path / f"{seed:07d}.pkl"), "wb") as f:
                pkl.dump((sim_processed_obs, target_pos), f)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Pose data collector script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
