import pinocchio
import moveit_commander
import torch
import rospy
import warnings

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

DEFAULT_ROBOT_POS = [-0.46947826, -0.00696641,  0.09270071] 

NUM_STEPS = {"real_push_buttons": 7, "real_stack_cup": 7, "real_put_fruit_in_box":7, "real_put_item_in_cabinet": 9, "push_buttons": 6, "real_open_drawer": 3, "real_put_plate": 7, "real_put_item_in_drawer":10, "real_hang_mug": 5}
ONLY_CARTESIAN = {"real_push_buttons": True, "real_stack_cup": True, "real_put_fruit_in_box": False, "real_put_item_in_cabinet": False, "push_buttons": True, "real_open_drawer": False , "real_take_plate": False, "real_put_item_in_drawer":False, "real_hang_mug":False}

class Arguments(tap.Tap):
    exp_config: str
    device: str = "cuda"
    num_demos: int = 10
    image_size: int = 128
    # cam_list: List[str] = ["bravo_camera","charlie_camera"]
    cam_list: List[str] = ["bravo_camera","charlie_camera","left_camera"]
    # cam_list: List[str] = ["bravo_camera","charlie_camera","bravo_camera", "charlie_camera"]
    arm: str = "left"
    env_name: str = "RealRobot-Pick-v0"
    # arch: str = "robo3d"
    # arch: str = "polarnet"
    arch: str = "hiveformer"
    save_obs_outs_dir: str = None
    checkpoint: str = None
    taskvar: str = "real_push_buttons+0"
    use_sem_ft: bool = False

def process_keystep(obs, cam_list=["bravo_camera", "charlie_camera", "left_camera"], crop_size=None):
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

    keystep = {"rgb": rgb.numpy().astype(np.uint8),
               "pc": pc.float().numpy(),
               "gripper_imgs": gripper_imgs,
               "gripper": gripper_pose}
    return keystep



def main(args):
    #FIXME: Update code to be installed as vlc_rlbench
    if args.arch == "hiveformer":
        print("Running hiveformer!")
        code_dir = '/home/rgarciap/Code/vlc_rlbench'
        sys.path.insert(0, code_dir)
        from eval_models import Actioner
    elif args.arch == "polarnet":
        print("Running PolarNet!")
        # code_dir = '/home/rgarciap/Code/work_csz/vlc_rlbench'
        code_dir = '/home/rgarciap/Code/polarnet_segm/'
        sys.path.insert(0, code_dir)
        from polarnet.eval_models import Actioner
    elif args.arch == "robo3d":
        print("Running Robo3D!")
        code_dir = '/home/rgarciap/Code/robo3d'
        sys.path.insert(0, code_dir)
        from robo3d.run.rlbench.eval_models import Actioner
    else:
        raise ValueError(f"arch {args.arch} not found")
    

    args.instr_embed_file = None
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
    task, var = taskvar.split("+")

    i =0 
    while i<1:
        i+=1
        actioner.reset(task, var, None, None)
        actioner.config.DATASET["camera_ids"] = None
        actioner.config.DATASET["camera"] = args.cam_list

        # actioner.config.defrost()
        # actioner.config.DATASET.remove_pcd_outliers = True
        # actioner.config.DATASET.camera_ids = [0, 1]
        # actioner.config.DATASET.taskvars = "/home/rgarciap/Code/robo3d/robo3d/assets/rlbench/real_robot_tasks.json"
        # actioner.config.freeze() 

    # lmdb_env = lmdb.open('/scratch/azimov/rgarciap/corl2023_jz/real_push_buttons+0', readonly=True)
    # txn = lmdb_env.begin()
    # for key, value in txn.cursor():
    #     print(key)
    #     episode = msgpack.unpackb(value)
    #     break
    
        obs = env.reset(gripper_pos=DEFAULT_ROBOT_POS)
        keystep_real = process_keystep(obs, cam_list=args.cam_list, crop_size=args.image_size)
        step_id = 0

        taskvars = actioner.config.DATASET.taskvars
        # with open(actioner.config.DATASET.taskvars) as f:
            # taskvars = json.load(f)
            # actioner.taskvars = taskvars
        print(actioner.taskvars)
        if args.taskvar not in actioner.taskvars:
            rospy.logwarn("Task was not included during training")
            actioner.taskvars.append(args.taskvar)
        num_steps = NUM_STEPS[task]

        taskvar_id = taskvars.index(taskvar)
        print("Taskvar ID: ", taskvar_id)
    
    
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
            action = actioner.predict(taskvar_id, step_id, keystep_real)
            # print(rospy.get_time()-init_t)
            action = action['action']
            keystep_real["action"] = action
        
            # np.save(f"/scratch/azimov/rgarciap/corl2023/real2/{step_id}", keystep_real)
            # action = episode['action'][step_id+1]

            pos = deepcopy(action[:3]).astype(np.double)
            quat = deepcopy(action[3:7]).astype(np.double)
            # open_gripper = action[7] >= 0.5 if step_id > 0 else True
            open_gripper = action[7] >= 0.5
            # open_gripper = action[7] >= 0.5 if step_id > 0 and step_id !=4 else True

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
