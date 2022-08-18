import gym

from robotamer.core import datasets
from robotamer.envs.pick import PickEnv

from robotamer.env_wrappers import teleop
from robotamer.env_wrappers import observations


def init_env(sim, arm, input_type, task, visible_state_features=[],
             num_input_frames=1, crop=None, image_size=None, grayscale=False,
             offline_dataset_path=None, task_version='v0'):
    obs_dataset = None
    # For testing with observations from offline dataset also on the real robot
    # (once we have seen in sim the actions are safe).
    # obs_dataset = datasets.OfflineDataset(offline_dataset_path)
    if sim:
        cam_list = []
        main_camera = 'left' if arm == 'right' else 'charlie'
        if offline_dataset_path is not None:
            print('Loading dataset', offline_dataset_path)
            obs_dataset = datasets.OfflineDataset(offline_dataset_path)
    elif arm == 'right':
        cam_list = ['left_camera', 'spare_camera']
        main_camera = 'left'
    else:
        cam_list = ['bravo_camera', 'charlie_camera']
        main_camera = 'charlie'

    if task == 'Push':
        env_id = f'RealRobot-Cylinder-{task}-{task_version}'
        gripper_in_2d = True
    else:
        env_id = f'RealRobot-{task}-{task_version}'
        gripper_in_2d = False

    env = gym.make(env_id,
                   cam_list=cam_list,
                   arm=arm,
                   version=task_version,
                   depth=True)
    # TODO: Separate obs_dataset from teleoperation.
    # if obs_dataset is not None:
    #     env = observations.StaticDatasetWrapper(obs_dataset)
    env = teleop.TeleopWrapper(env, obs_dataset)
    image_key_in = f'rgb_{main_camera}_camera'
    image_key_out = 'rgb'
    env = observations.ImageObservationWrapper(
        env, image_key_in, image_key_out, crop, image_size, grayscale)
    env = observations.VisibleStateWrapper(
        env, [image_key_out] + visible_state_features, gripper_in_2d)
    env = observations.ImageStackingWrapper(
        env, image_key_out, num_input_frames)
    return env, main_camera

