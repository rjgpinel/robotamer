import argparse
import torch
import pickle as pkl
import numpy as np

from pathlib import Path
from copy import deepcopy
from robotamer.core.constants import CAM_INFO
from muse.envs.utils import realsense_resize_crop, realsense_resize_batch_crop
from muse.core.constants import REALSENSE_CROP, REALSENSE_RESOLUTION
from robotamer.core.tf import pos_quat_to_hom, hom_to_pos


def get_args_parser():
    parser = argparse.ArgumentParser("Data cleaning script", add_help=False)
    # Directory to save trajectory
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--num-cubes", default=3, type=int)
    return parser


def main(args):
    dataset_dir = Path(args.dataset_dir)
    real_dir = dataset_dir / "real"
    real_processed_dir = dataset_dir / "real_processed"
    sim_dir = dataset_dir / "sim"
    real_processed_dir.mkdir(parents=True, exist_ok=True)

    cam_list = list(CAM_INFO.keys())

    render_w, render_h = REALSENSE_RESOLUTION
    W, H = REALSENSE_CROP
    step_frames = torch.zeros((len(cam_list), render_h, render_w, 3), dtype=torch.uint8)

    stats = {
        "num_cubes": args.num_cubes,
        "cam_list": cam_list,
        "gripper_pos": [],
        "gripper_quat": [],
        "target_pos": [],
    }

    for filename in real_dir.glob("*.pkl"):
        if filename.name == "stats.pkl":
            continue
        print(filename.name)
        with open(str(filename), "rb") as f:
            print(filename)
            obs, target_pos = pkl.load(f)

            for cam_i, cam_name in enumerate(cam_list):
                step_frames[cam_i, :, :, :] = torch.from_numpy(obs[f"rgb_{cam_name}"])

            frames, _ = realsense_resize_batch_crop(step_frames.to("cuda"))

            for cam_i, cam_name in enumerate(cam_list):
                obs[f"rgb_{cam_name}"] = frames[cam_i].cpu().numpy()

            gripper_pos, gripper_quat = obs["gripper_pos"], obs["gripper_quat"]

            target_pos_bis = []
            for i in range(args.num_cubes):
                cube_pos, cube_quat = obs[f"cube{i}_pos"], [0,0,0,1]
                world_T_target = pos_quat_to_hom(cube_pos, cube_quat)
                world_T_gripper = pos_quat_to_hom(gripper_pos, gripper_quat)
                gripper_T_world = np.linalg.inv(world_T_gripper)
                gripper_T_target = np.matmul(gripper_T_world, world_T_target)
                target_pos_bis.append(hom_to_pos(gripper_T_target))

            target_pos = np.concatenate(target_pos_bis)
            stats["gripper_pos"].append(gripper_pos)
            stats["gripper_quat"].append(gripper_quat)
            stats["target_pos"].append(target_pos)

            with open(str(real_processed_dir / filename.name), "wb") as f:
                pkl.dump((obs, target_pos), f)

    for k, v in stats.items():
        if k not in ["cam_list", "num_cubes"]:
            stats[k] = {
                "mean": np.mean(v, axis=0),
                "std": np.std(v, axis=0),
            }

    with open(str(real_processed_dir / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)

    with open(str(sim_dir / "stats.pkl"), "wb") as f:
        pkl.dump(stats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Data cleaning script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    assert Path(args.dataset_dir).exists(), "Provided dataset dir does not exist"
    main(args)
