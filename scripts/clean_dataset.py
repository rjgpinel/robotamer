import argparse
import torch
import pickle as pkl

from copy import deepcopy
from robotamer.core.constants import CAM_INFO
from muse.envs.utils import realsense_resize_crop, realsense_resize_batch_crop
from muse.core.constants import REALSENSE_CROP, REALSENSE_RESOLUTION


def get_args_parser():
    parser = argparse.ArgumentParser("Data cleaning script", add_help=False)
    # Directory to save trajectory
    parser.add_argument("dataset-dir", type=str, required=True)
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
    step_frames = torch.zeros((num_streams, render_h, render_w, 3), dtype=torch.uint8)

    stats = {
        "cam_list": cam_list,
        "gripper_pos": [],
        "gripper_quat": [],
        "target_pos": [],
    }

    for filename in real_processed_dir.glob("*.pkl"):
        with open(str(real_dir / filename), "rb") as f:
            obs, target_pos = pkl.load(f)
            for cam_i, cam_name in enumerate(cam_list):
                step_frames[cam_i, :, :, :] = torch.from_numpy(obs[f"rgb_{cam_name}"])

            frames, _ = realsense_resize_batch_crop(step_frames.to(ptu.device))

            for cam_i, cam_name in enumerate(cam_list):
                obs[f"rgb_{cam_name}"] = frames[cam_i].cpu().numpy()

            stats["gripper_pos"].append(obs["gripper_pos"])
            stats["gripper_quat"].append(obs["gripper_quat"])
            stats["target_pos"].append(target_pos)

            with open(str(real_processed_dir / filename), "wb") as f:
                pkl.dump((obs, target_pos), f)

    for k, v in stats.items():
        if k != "cam_list":
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
