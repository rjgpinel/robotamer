import gym

from robotamer.core import datasets
from robotamer.envs.pick import PickEnv


def init_env(sim, arm, offline_dataset_path=None, task_version='v0'):
    obs_dataset = None
    if sim:
        cam_list = []
        main_camera = 'left' if arm == 'right' else 'charlie'
        obs_dataset = datasets.OfflineDataset(offline_dataset_path)
    elif arm == 'right':
        cam_list = ['left_camera', 'spare_camera']
        main_camera = 'left'
    else:
        cam_list = ['bravo_camera', 'charlie_camera']
        main_camera = 'charlie'
    env = gym.make(f'RealRobot-Cylinder-Push-{task_version}',
                   cam_list=cam_list,
                   arm=arm,
                   version=task_version,
                   depth=False)
    env.is_ready = False
    return env, obs_dataset

