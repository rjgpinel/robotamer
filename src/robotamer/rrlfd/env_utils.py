import gym

from robotamer.envs.pick import PickEnv

from robotamer.env_wrappers import datasets
from robotamer.env_wrappers import observations
from robotamer.env_wrappers import rewards
from robotamer.env_wrappers import teleop


def init_env(task, sim, arm, input_type, visible_state_features=[],
             num_input_frames=1, crop=None, image_size=None, grayscale=False,
             offline_dataset_path=None, out_dataset_path=None,
             task_version='v0'):
    if sim:
        cam_list = []
        main_camera = 'left' if arm == 'right' else 'charlie'
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
                   depth=False)
    env = teleop.TeleopWrapper(env)
    if offline_dataset_path is not None:
        env = observations.StaticDatasetWrapper(env, offline_dataset_path)
    if out_dataset_path is not None:
        env = datasets.RecordEpisodesWrapper(env, out_dataset_path)

    image_key_in = f'rgb_{main_camera}_camera'
    image_key_out = 'rgb'
    env = observations.ImageObservationWrapper(
        env, image_key_in, image_key_out, crop, image_size, grayscale)
    env = observations.VisibleStateWrapper(
        env, [image_key_out] + visible_state_features, gripper_in_2d)
    # TODO: Be careful about whether to duplicate obs at the start of the episode.
    # bc_agent takes care of this but not sure if non-residual RL agent does.
    env = observations.ImageStackingWrapper(
        env, image_key_out, num_input_frames)

    env = rewards.SparseRewardWrapper(env)
    return env

