import copy
import functools
import os
import pickle
import random
import time

import gym
import rospy

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from rrlfd import prl_ur5_utils
from rrlfd.bc import bc_agent
from rrlfd.bc import train_utils
from rrlfd.bc import flags as bc_flags

from robotamer.core import datasets
from robotamer.core import utils as robotamer_utils
from robotamer.envs.pick import PickEnv
from robotamer.rrlfd import observations
from robotamer.rrlfd import utils
from sensor_msgs.msg import Joy


flags.DEFINE_bool('sim', False,
                  'If true (running in simulation), use proprioceptive '
                  'observations only. Else initialize cameras.')
flags.DEFINE_enum('arm', 'left', ['left', 'right'],
                  'Which arm to use.')
flags.DEFINE_enum('task_version', 'v0', ['v0', 'v1'],
                  'Which version of the task to use.')
flags.DEFINE_boolean('grayscale', True,
                     'If True, convert RGB camera images to grayscale.')
# Default values for charlie_camera.
flags.DEFINE_list('main_camera_crop', [140, -50, 180, 470],
                  'Region of full camera image to crop to before rescaling.')
flags.DEFINE_string('offline_dataset_path', None,
                    'If developing in sim, path to dataset from which to load '
                    'image observations.')

FLAGS = flags.FLAGS


def predict_actions(env, eval_dataset, obs_stack, agent, obs_dataset=None):
    new_obs = obs_stack.newest_observation
    obs_hist = obs_stack.observation_history
    action = agent.get_action(new_obs, obs_hist, env)
    print('action:', action)
    full_action = {'linear_velocity': np.concatenate([action, [0.]], axis=0),
                   'angular_velocity': np.array([0., 0., 0.]),
                   'grip_open': 0}
    obs = env.step(full_action)
    if obs_dataset is not None:
        obs = obs_dataset.step()
    eval_dataset.append(action, obs)
    obs_stack.append(obs)


def teleop_callback(teleop, env, dataset, obs_stack, agent, obs_dataset=None):
    if teleop.buttons[0]:  # A
        env.is_ready = True
        if obs_dataset is None:
            obs = env.env.render()
        else:
            obs = obs_dataset.reset()
        dataset.reset(obs)
        obs_stack.reset(obs)
        print('Starting the episode')
    elif teleop.buttons[1] or teleop.buttons[2]:  # B, X
        env.is_ready = False
        # Wait for arm to come to a stop // send zero action or nothing?
        # rospy.sleep(1)
        env.reset()
        dataset.flag_success(teleop.buttons[2])
        dataset.save()


def start_episode(env, dataset, obs_stack, agent, obs_dataset):
    rate = rospy.Rate(5)
    # Wait to receive a first image after a reset.
    while not env.is_ready:
        rate.sleep()
    
    prev_time = time.time()
    while env.is_ready and not rospy.is_shutdown():
        predict_actions(env, dataset, obs_stack, agent, obs_dataset)
        new_time = time.time()
        print('dt =', new_time - prev_time)
        prev_time = new_time
        rate.sleep()


def load_saved_agent(env, main_camera, main_camera_crop, grayscale):
    demos_file, ckpt_dir, summary_dir = train_utils.set_paths(FLAGS.demo_task)
    demo_task = FLAGS.demo_task or FLAGS.eval_task
    visible_state_features = prl_ur5_utils.get_visible_features_for_task(
      demo_task, FLAGS.visible_state)
    image_size = FLAGS.image_size

    agent = bc_agent.BCAgent(
        network_type=FLAGS.network,
        input_type=FLAGS.input_type,
        binary_grip_action=FLAGS.binary_grip_action,
        grip_action_from_state=FLAGS.grip_action_from_state,
        zero_action_keeps_state=FLAGS.zero_action_keeps_state,
        early_closing=FLAGS.early_closing,
        num_input_frames=FLAGS.num_input_frames,
        crop_frames=FLAGS.crop_frames,
        full_image_size=image_size,
        crop_size=image_size - FLAGS.crop_margin_size,
        target_offsets=[int(t) for t in FLAGS.target_offsets],
        visible_state_features=visible_state_features,
        action_norm=FLAGS.action_norm,
        signals_norm=FLAGS.signals_norm,
        action_space='tool_lin' if FLAGS.domain == 'mime' else demo_task,
        last_activation=FLAGS.last_activation,
        fc_layer_sizes=[int(i) for i in FLAGS.fc_layer_sizes],
        weight_decay=FLAGS.weight_decay,
        env=env,
        late_fusion=FLAGS.late_fusion,
        init_scheme=FLAGS.weight_init_scheme,
        num_input_channels=1 if FLAGS.grayscale else 3)

    obs_stack = observations.ObservationConverter(
        main_camera,
        visible_state_features,
        agent.num_input_frames,
        image_size,
        grayscale=grayscale,
        crop=main_camera_crop,
    )
    # For setting normalization stats.
    dataset = train_utils.prepare_demos(
        demos_file, FLAGS.input_type, FLAGS.max_demos_to_load,
        FLAGS.max_demo_length, FLAGS.augment_frames, agent, ckpt_dir,
        FLAGS.val_size, FLAGS.val_full_episodes)

    ckpt_to_load = os.path.join(ckpt_dir, 'ckpt')
    print('Loading from', ckpt_to_load)
    obs_space = dataset.observations[0][0]
    agent.restore_from_ckpt(ckpt_to_load, compile_model=True,
                            obs_space=obs_space)

    return agent, obs_stack, dataset, ckpt_dir


def init_env(sim, arm, task_version, offline_dataset_path):
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
                   arm=FLAGS.arm,
                   version=task_version,
                   depth=False)
    env.is_ready = False
    return env, obs_dataset


def main(_):
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    env, obs_dataset = utils.init_env(
        FLAGS.sim, FLAGS.arm, FLAGS.offline_dataset_path, FLAGS.task_version)
    agent, obs_stack, demo_dataset, ckpt_dir = load_saved_agent(
        env, main_camera, FLAGS.main_camera_crop, FLAGS.grayscale)

    env.reset()

    try:
        timestamp = robotamer_utils.get_timestamp()
        dataset_path = os.path.join(
            ckpt_dir, 'real_robot_eval', f'evalPush_{timestamp}.pkl')
        eval_dataset = datasets.EpisodeDataset(dataset_path)
        # TODO: Make sure to also reset stacked frames
        callback = functools.partial(
            teleop_callback, env=env, dataset=eval_dataset, obs_stack=obs_stack,
            agent=agent, obs_dataset=obs_dataset)
        rospy.Subscriber('joy_teleop', Joy, callback, queue_size=1)

        while not rospy.is_shutdown():
            print('Waiting for episode start')
            start_episode(env, eval_dataset, obs_stack, agent, obs_dataset)
    except rospy.ROSInterruptException:
        print('Exiting')

 
if __name__ == '__main__':
    app.run(main)
