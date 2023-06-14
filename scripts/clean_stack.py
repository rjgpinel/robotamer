import argparse
import torch
import pickle as pkl
import numpy as np

from pathlib import Path
from copy import deepcopy


def get_args_parser():
    parser = argparse.ArgumentParser("Data cleaning stack script", add_help=False)
    # Directory to save trajectory
    parser.add_argument("dataset_dir", type=str)
    return parser


def main(args):
    dataset_dir = Path(args.dataset_dir)
    real_dir = dataset_dir / "real"
    real_processed_dir = dataset_dir / "real_processed"
    sim_dir = dataset_dir / "sim"
    real_processed_dir.mkdir(parents=True, exist_ok=True)

    for filename in real_dir.glob("*.pkl"):
        with open(str(filename), "rb") as f:
            print(filename)

            if filename.name == "stats.pkl":
                stats = pkl.load(f)
                stats["traj_stats"].pop("angular_velocity")
                with open(str(real_processed_dir / "stats.pkl"), "wb") as f:
                    pkl.dump(stats, f)
            else:
                obs, action = pkl.load(f)
                action.pop("angular_velocity")
                with open(str(real_processed_dir / f"0000{str(filename.name)}"), "wb") as f:
                    pkl.dump((obs, action), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Data cleaning stack script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    assert Path(args.dataset_dir).exists(), "Provided dataset dir does not exist"
    main(args)
