import argparse
import muse.envs
import robotamer.envs
import gym
import rospy
import torch
import torchvision.transforms as T

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import ibc.utils.torch as ptu

from math import pi
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from muse.envs.utils import realsense_resize_crop, realsense_resize_batch_crop
from muse.core.constants import REALSENSE_CROP, REALSENSE_RESOLUTION
from ibc.data.utils import filter_state


def compute_statistics(output_path, collect_stats):
    data = {}
    all_actions = collect_stats.pop("actions")
    for action in all_actions:
        for k, v in action.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    all_obs = collect_stats.pop("obs")
    for obs in all_obs:
        for k, v in obs.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    velocity_dim = 0
    for k, v in collect_stats["action_space"].items():
        if "grip" not in k:
            velocity_dim += v.shape[0]

    stats = {
        "num_cameras": collect_stats.pop("num_cameras"),
        "cam_list": collect_stats.pop("cam_list"),
        "action_space": collect_stats.pop("action_space"),
        "vel_dim": velocity_dim,
        "collect_stats": collect_stats,
        "dataset_size": len(all_actions),
        "traj_stats": {},
    }

    for k, v in data.items():
        stats["traj_stats"][k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
        }

    with open(str(output_path / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)


def get_args_parser():
    parser = argparse.ArgumentParser("Data collection script", add_help=False)
    # Directory to save trajectory
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument("--init-seed", default=5000, type=int)
    parser.add_argument("--cam-list", default="", type=str)
    parser.add_argument("--env-name", default="Pick-v0", type=str)
    parser.add_argument("--num-episodes", default=25, type=int)
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize sim2real comparison"
    )
    parser.set_defaults(visualize=False)
    return parser


def main(args):
    ptu.set_gpu_mode(True)

    # create output dirs
    output_dir = Path(args.output_dir)
    real_exp_dir = output_dir / "real"
    sim_exp_dir = output_dir / "sim"
    real_exp_dir.mkdir(parents=True, exist_ok=True)
    sim_exp_dir.mkdir(parents=True, exist_ok=True)

    # create envs
    sim_env_name = args.env_name
    real_env_name = f"RealRobot-{sim_env_name}"
    sim_env = gym.make(sim_env_name)

    # define cameras
    if args.cam_list:
        cam_list = args.cam_list.split(",")
    else:
        cam_list = sim_env.unwrapped.cam_list

    real_env = gym.make(real_env_name, cam_list= cam_list)

    num_streams = len(cam_list)

    # visualize sim trajectory
    if args.visualize:
        scene = sim_env.unwrapped.scene
        scene.renders(True)

    seeds = (args.init_seed, args.init_seed + args.num_episodes)

    render_w, render_h = REALSENSE_RESOLUTION
    W, H = REALSENSE_CROP
    step_frames = torch.zeros((num_streams, render_h, render_w, 3), dtype=torch.uint8)

    real_env.reset()

    sim_stats = {
        "num_steps": [],
        "successful_seeds": [],
        "failure_seeds": [],
        "action_space": sim_env.unwrapped.action_space.spaces,
        "cam_list": sim_env.unwrapped.cam_list,
        "num_cameras": 1,
        "actions": [],
        "obs": [],
    }

    real_stats = {
        "num_steps": [],
        "successful_seeds": [],
        "failure_seeds": [],
        "action_space": sim_env.unwrapped.action_space.spaces,
        "cam_list": cam_list,
        "num_cameras": 1,
        "actions": [],
        "obs": [],
    }

    for seed in tqdm(range(*seeds)):
        # pre-compute expert trajectory on simulation for the given seed
        sim_env.seed(seed)
        sim_obs = sim_env.reset()
        gripper_pos = sim_obs["gripper_pos"]

        agent = sim_env.unwrapped.oracle()
        gripper_pos = sim_obs["gripper_pos"]

        # simulation trajectory
        sim_traj = [sim_obs["gripper_pos"]]
        sim_obs_list = []
        sim_episode_traj = []
        actions = []
        for i in range(sim_env.spec.max_episode_steps):
            action = agent.get_action()
            actions.append(action)
            if action is None:
                print("Expert trajectory finished.")
                info = {"success": False}
                break
            sim_episode_traj.append((sim_obs, action))
            sim_obs, _, done, info = sim_env.unwrapped.step(action)
            sim_traj.append(sim_obs["gripper_pos"])
            if rospy.is_shutdown() or done:
                break

        if not info["success"]:
            sim_stats["failure_seeds"].append(seed)
            continue

        sim_stats["successful_seeds"].append(seed)
        sim_stats["num_steps"].append(len(sim_episode_traj))
        for i, step in enumerate(sim_episode_traj):
            with open(str(sim_exp_dir / f"{seed:04d}_{i:05d}.pkl"), "wb") as f:
                pkl.dump(step, f)

        for obs, action in sim_episode_traj:
            state = filter_state(obs)
            sim_stats["obs"] += [state]
            sim_stats["actions"] += [action]

        # real trajectory
        set_success = real_env.set_scene(sim_env)
        real_obs = real_env.reset(gripper_pos=gripper_pos)
        real_traj = [real_obs["gripper_pos"]]
        real_episode_traj = []
        for action_idx in range(len(actions)):
            action = actions[action_idx]
            action["angular_velocity"] = np.zeros_like(action["linear_velocity"])

            # copy sim joint velocities - maybe use real ones
            # real_obs["arms_joint_vel"] = sim_episode_traj[action_idx][0][
                # "arms_joint_vel"
            # ]
            for cam_i, cam_name in enumerate(cam_list):
                step_frames[cam_i, :, :, :] = torch.from_numpy(
                    real_obs[f"rgb_{cam_name}"]
                )
            frames, _ = realsense_resize_batch_crop(step_frames.to(ptu.device))
            for cam_i, cam_name in enumerate(cam_list):
                real_obs[f"rgb_{cam_name}"] = frames[cam_i].cpu().numpy()

            real_episode_traj.append((real_obs, action))
            real_obs, _, _, _ = real_env.step(action)
            real_traj.append(real_obs["gripper_pos"])
            if rospy.is_shutdown():
                break

        # avoid strong stop in the robot
        action["linear_velocity"] = np.zeros_like(action["linear_velocity"])
        action["angular_velocity"] = np.zeros_like(action["linear_velocity"])
        real_env.step(action)

        if not info["success"]:
            continue

        print("real finished, saving...")
        real_stats["successful_seeds"].append(seed)
        real_stats["num_steps"].append(len(real_episode_traj))
        for i, step in enumerate(real_episode_traj):
            with open(str(real_exp_dir / f"{seed:04d}_{i:05d}.pkl"), "wb") as f:
                pkl.dump(step, f)

        for obs, action in real_episode_traj:
            state = filter_state(obs)
            real_stats["obs"] += [state]
            real_stats["actions"] += [action]

        clean_success = real_env.clean_scene(sim_env)

    compute_statistics(real_exp_dir, real_stats)
    compute_statistics(sim_exp_dir, sim_stats)

    if rospy.is_shutdown():
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Data collection script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        main(args)
    except rospy.ROSInterruptException:
        pass
