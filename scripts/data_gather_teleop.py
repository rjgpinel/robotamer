import functools
import gym
import argparse
import rospy
import robotamer.envs
import torch
import sys

import numpy as np
import lmdb
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from pathlib import Path
from robotamer.core.utils import crop_center, resize

from sensor_msgs.msg import Joy

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def get_args_parser():
    parser = argparse.ArgumentParser("Remote controller data gathering", add_help=False)
    parser.add_argument("--arm", default="left", type=str, help="left or right")
    parser.add_argument("--sim", action="store_true", help="If true (running in simulation), use proprioceptive observations only. Else initialize cameras.")
    parser.add_argument("--debug", action="store_true", help="If true print joystick received inputs.")
    parser.add_argument("--data-dir", default="/scratch/azimov/rgarciap/corl2023/new_data/", type=str)
    parser.add_argument("--task", default="push_buttons", type=str)
    parser.add_argument("--var", default=0, type=int)
    parser.add_argument("--crop-size", default=128)
    parser.set_defaults(sim=False, debug=False)
    return parser


class Dataset:
    def __init__(self, output_dir, camera_list, crop_size=None):
        self.output_dir = output_dir
        self.lmdb_env = lmdb.open(str(self.output_dir), map_size=int(1024**4))
        self.data = []
        self.episode_idx = 0
        self.camera_list = camera_list
        self.crop_size = crop_size

    def reset(self):
        self.data = []

    def add_keystep(self, obs):
        gripper_pos = obs["gripper_pos"]
        gripper_quat = obs["gripper_quat"]
        gripper_pose = np.concatenate([gripper_pos, gripper_quat])
        gripper_state = obs["gripper_state"]

        rgb = []
        gripper_uv = {}
        pc = []
        for cam_name in self.camera_list:
            rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
            pc.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))
            gripper_uv[cam_name] = obs[f"gripper_uv_{cam_name}"]


        rgb = torch.stack(rgb) 
        pc = torch.stack(pc) 
        action = np.concatenate([gripper_pose, np.array([int(gripper_state)])], axis=-1)
        
        # import pudb; pudb.set_trace()
        if self.crop_size:
            rgb = rgb.permute(0, 3, 1, 2)
            pc = pc.permute(0, 3, 1, 2)

            rgb, ratio = resize(rgb, self.crop_size, im_type="rgb")
            pc, ratio = resize(pc, self.crop_size, im_type="pc")
            rgb, start_x, start_y = crop_center(rgb, self.crop_size, self.crop_size)
            pc, start_x, start_y = crop_center(pc, self.crop_size, self.crop_size)
            rgb = rgb.permute(0, 2, 3, 1)
            pc = pc.permute(0, 2, 3, 1)
            for cam_name, uv in gripper_uv.items():
                gripper_uv[cam_name] = [int(uv[0]*ratio) - start_x, int(uv[1]*ratio) - start_y]

        keystep = {"rgb": rgb,
                   "pc": pc,
                   "gripper_uv": gripper_uv,
                   "action": action}

        self.data.append(keystep)

    def save(self):
        
        rgbs = []
        pcs = []
        gripper_uv = []
        actions = []

        for keystep in self.data:
            rgbs.append(keystep["rgb"])
            pcs.append(keystep["pc"])
            gripper_uv.append(keystep["gripper_uv"])
            actions.append(keystep["action"])

        outs = {
                "rgb": torch.stack(rgbs).float().numpy(),
                "pc": torch.stack(pcs).float().numpy(),
                "gripper_uv": gripper_uv,
                "action": np.stack(actions).astype(np.float32),
            }

        txn = self.lmdb_env.begin(write=True)
        txn.put(f"episode{self.episode_idx}".encode('ascii'), msgpack.packb(outs))
        txn.commit()
        self.episode_idx += 1
        self.reset()

    def done(self):
        self.lmdb_env.close()

def teleop_callback(data, env, dataset, x_scale=0.15, y_scale=0.15, z_scale=0.05, x_rot_scale=0.15, y_rot_scale=0.15, z_rot_scale=0.15, debug=False):
    if debug:
        print('Received', data)
        print('eef', env.robot.eef_pose()[0])

    left_joy_left = data.axes[0]
    left_joy_up = data.axes[1]
    right_joy_left = data.axes[2]
    right_joy_up = data.axes[3]
    x_rot = data.buttons[5]/2 - data.buttons[4]/2
    y_rot = data.buttons[7]/2 - data.buttons[6]/2

    save = data.buttons[3]  # Y
    keystep = data.buttons[0]  # X
    discard = data.buttons[8] # back
    finish = data.buttons[9] # start
    open_gripper = data.buttons[2]  # B
    close_gripper = data.buttons[1]  # A

    vx = x_scale * -left_joy_up
    vy = y_scale * -left_joy_left
    vz = z_scale * right_joy_up
    wx = x_rot_scale * -x_rot
    wy = y_rot_scale * -y_rot
    wz = z_rot_scale * -right_joy_left

    action = {
        'linear_velocity': np.array([vx, vy, vz]),
        'angular_velocity': np.array([wx, wy, wz]),
    }

    if open_gripper or close_gripper:
        action["grip_open"] = open_gripper - close_gripper

    if debug:
        print('Sending', action)
    if finish:
        dataset.done()
        print("Dataset recorded")
        sys.exit()
    elif keystep:
        obs = env.render()
        dataset.add_keystep(obs)
        print("Keystep recorded")
    elif save:
        print('Finished episode; Resetting arm; Saving episode')
        dataset.save()
        obs = env.reset()
        print('Reset finished')
        print('Ready to receive joystick controls')
    elif discard:
        print('Discard episode')
        obs = env.reset()
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
        # pass
        real_obs, done, reward, info = env.step(action)


def test_displacement(env):
    print('pose before:', env.robot.eef_pose())
    for _ in range(4):
        action = {
            'linear_velocity': np.array([1, 0, 0]),
            'angular_velocity': np.array([0, 0, 0]),
        }
        env.step(action)
    print('pose after:', env.robot.eef_pose())


def main(args):
    try:
        if args.sim:
            cam_list = []
        elif args.arm == 'right':
            cam_list = ['left_camera', 'spare_camera']
        else:
            cam_list = ['bravo_camera', 'charlie_camera']
        env = gym.make('RealRobot-Pick-v0',
                       cam_list=cam_list,
                       arm=args.arm,
                       version="v0",
                       depth=not args.sim,
                       pcd=not args.sim,
                       gripper_attn=True,
                       )
    
        output_dir = Path(args.data_dir) / f"{args.task}+{args.var}"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = Dataset(str(output_dir), camera_list=cam_list, crop_size=args.crop_size)
        real_obs = env.reset()
        print('Cartesian pose', env.robot.eef_pose())
        #print('Config', env.env._get_current_config())

        x_scale = y_scale = 0.05
        env_step_callback = functools.partial(
            teleop_callback, env=env, dataset=dataset, x_scale=x_scale,
            y_scale=y_scale, debug=args.debug)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')
        print('Observation fields', real_obs.keys())

        rospy.spin()

    except rospy.ROSInterruptException:
        print('Exiting')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Remote controller data gathering", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)


