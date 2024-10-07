import pinocchio
import moveit_commander
import torch
import rospy
import warnings
import pickle as pkl

import tap
import os

import sys
import json

import robotamer.envs

import gym
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
from robotamer.core.utils import resize, crop_center
from robotamer.envs.utils import quat_to_euler

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


warnings.filterwarnings("ignore")

DEFAULT_ROBOT_POS = [-0.1876253 ,  0.18788611,  0.11547332]

NUM_STEPS = {"real_push_buttons": 7, "real_stack_cup": 10, "real_put_fruit_in_box": 10, "real_put_item_in_cabinet": 9, "push_buttons": 6, "real_open_drawer": 3, "real_put_plate": 7, "real_put_item_in_drawer": 15, "real_hang_mug": 10, "real_put_fruits_in_plate": 20}
ONLY_CARTESIAN = {"real_push_buttons": True, "real_stack_cup": False, "real_put_fruit_in_box": False, "real_put_item_in_cabinet": False, "push_buttons": True, "real_open_drawer": False , "real_take_plate": False, "real_put_item_in_drawer": False, "real_hang_mug": True, "real_put_fruits_in_plate": True}

class Arguments(tap.Tap):
    exp_config: str
    device: str = "cuda"
    num_demos: int = 10
    image_size: int = 256
    # cam_list: List[str] = ["bravo_camera","charlie_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera","left_camera"]
    cam_list: List[str] = ["bravo_camera","charlie_camera","alpha_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera","alpha_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera","alpha_camera", "bravo_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera","bravo_camera", "charlie_camera"]
    arm: str = "left"
    env_name: str = "RealRobot-Pick-v0"
    # arch: str = "robo3d"
    # arch: str = "polarnet"
    arch: str = "ptv3"
    save_obs_outs_dir: str = None
    checkpoint: str = None
    taskvar: str = "real_push_buttons+0"
    use_sem_ft: bool = False
    ip: str = "cleps.inria.fr"
    port: int = 8001
    

def process_keystep(obs, cam_list=["bravo_camera", "charlie_camera", "left_camera"], crop_size=None):
    with open("/home/rgarciap/catkin_ws/src/robotamer/src/robotamer/assets/real_robot_bbox_info.pkl", "rb") as f:
        links_bbox = pkl.load(f)

    rgb = []
    pc = []
    gripper_pos = obs["gripper_pos"]
    gripper_quat = obs["gripper_quat"]
    gripper_pose = np.concatenate([gripper_pos, gripper_quat])
    for cam_name in cam_list:
        rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
        pc.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))

    rgb = torch.stack(rgb)  # (C, H, W, 3)
    pc = torch.stack(pc)    # (C, H, W, 3)
        
    if crop_size:
        rgb = rgb.permute(0, 3, 1, 2)
        pc = pc.permute(0, 3, 1, 2)

        rgb, ratio = resize(rgb, crop_size, im_type="rgb")
        pc, ratio = resize(pc, crop_size, im_type="pc")
        rgb, start_x, start_y = crop_center(rgb, crop_size, crop_size)
        pc, start_x, start_y = crop_center(pc, crop_size, crop_size)
        rgb = rgb.permute(0, 2, 3, 1)
        pc = pc.permute(0, 2, 3, 1)

    robot_info = obs["robot_info"]
    bbox_info = {}
    pose_info = {}
    for link_name, link_pose in robot_info.items():
        pose_info[f"{link_name}_pose"] = link_pose
        bbox_info[f"{link_name}_bbox"] = links_bbox[link_name]

    keystep = {
        "rgb": rgb.numpy().astype(np.uint8),
        "pc": pc.float().numpy(),
        "gripper": gripper_pose,
        "arm_links_info": (bbox_info, pose_info),
    }
    return keystep



def main(args):
    if args.arch == 'ptv3':
        print('Running point cloud transformer v3!')
        code_dir = '/home/rgarciap/Code/genrobot3d'
        sys.path.insert(0, code_dir)
        from genrobo3d.evaluation.eval_simple_policy import Actioner

        args.real_robot = True
        args.best_disc_pos = 'max' # ens1, max
        args.num_ensembles = 1

    else:
        raise ValueError(f"arch {args.arch} not found")
    
    taskvar_instructions = json.load(open(
        os.path.join(code_dir, 'assets/taskvars_instructions_realrobotv1.json')
    ))
    # import pudb; pudb.set_trace()
    actioner = Actioner(args)

    # Create env
    env = gym.make('RealRobot-Pick-v0',
            cam_list=args.cam_list,
            cam_async=False,
            arm=args.arm,
            version="wide",
            depth=True,
            pcd=True,
            gripper_attn=True)


    taskvar = args.taskvar
    task, variation = taskvar.split("+")

    obs = env.reset(gripper_pos=DEFAULT_ROBOT_POS)
    keystep_real = process_keystep(obs, cam_list=args.cam_list, crop_size=args.image_size)
    step_id = 0

    num_steps = NUM_STEPS[task]

    for step_id in range(num_steps):
        # keystep = {
        #     'rgb': episode['rgb'][step_id],
        #     'pc': episode['pc'][step_id],
        #     'gripper': episode['action'][step_id]
        # }
        # action = actioner.predict(taskvar_id, step_id, keystep)

        # keystep_real['pc'] += np.random.randn(*keystep_real['pc'].shape)*0.01
        # import pudb; pudb.set_trace() 
        if rospy.is_shutdown():
            break
        init_t = rospy.get_time()
        # import pudb; pudb.set_trace()
        for key, value in keystep_real.items():
            print(key)
            if isinstance(value, np.ndarray):
                print(value.shape)
        instructions = taskvar_instructions[taskvar]
        action = actioner.predict(
            task_str=task, variation=variation, step_id=step_id, 
            obs_state_dict=keystep_real, 
            episode_id=0, 
            instructions=instructions,
        )
        
        # print(rospy.get_time()-init_t)
        action = action['action']
        keystep_real["action"] = action
    
        np.save(f"/scratch/azimov/rgarciap/models_icra25/v1/ptv3/real_robot_data/v1_val/{step_id}.npy", keystep_real)
        # action = episode['action'][step_id+1]

        pos = deepcopy(action[:3]).astype(np.double)
        quat = deepcopy(action[3:7]).astype(np.double)
        # open_gripper = action[7] >= 0.5 if step_id > 0 else True
        open_gripper = action[7] >= 0.5

        print('action', action)
        print("Predicting step:", step_id, f"Open gripper {open_gripper}")

        # print("Position:", pos)
        # print("Quaternion:", quat)
        # print("Open gripper:", open_gripper)
        # print("Rotation:", quat_to_euler(quat, True))
        if rospy.is_shutdown():
            break
        obs, _, _, _ = env.move(pos, quat, open_gripper, only_cartesian=ONLY_CARTESIAN[task])
        # obs, _, _, _ = env.move(pos, gripper_quat=None, open_gripper=open_gripper)
        # return

        keystep_real = process_keystep(obs, cam_list=args.cam_list, crop_size=args.image_size)

        if rospy.is_shutdown():
            break

    # gripper_pos = obs["gripper_pos"]
    # gripper_pos[-1] += 0.03
    # env.move(gripper_pos, open_gripper=True)


if __name__ == "__main__":
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    args.real_robot = True
    if not Path(args.exp_config).exists():
        raise ValueError(f"Config path {args.exp_config} does not exist.")

    try:
        main(args)
    except rospy.ROSInterruptException:
        pass

