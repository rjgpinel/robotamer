import os
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
from robotamer.rrlfd import env_utils


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


def predict_actions(env, agent):
    new_obs = env.newest_observation
    obs_hist = env.observation_history
    action = agent.get_action(new_obs, obs_hist, env)
    print('action:', action)
    full_action = {'linear_velocity': np.concatenate([action, [0.]], axis=0),
                   'angular_velocity': np.array([0., 0., 0.]),
                   'grip_open': 0}
    obs, reward, done, info = env.step(full_action)
    print('reward, done, info:', reward, done, info)
    return done


def attempt_reset(env, max_attempts=2):
    reset_attempts = 0
    while reset_attempts < max_attempts:
        try:
            obs = env.reset()
            break
        except RuntimeError as e:
            reset_attempts += 1
            if reset_attempts >= max_attempts:
                raise e
    return obs


def start_episode(env, agent):
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        obs = attempt_reset(env)
        times = []
        prev_time = time.time()
        done = False
        while not done and not rospy.is_shutdown():
            done = predict_actions(env, agent)
            new_time = time.time()
            print('dt =', new_time - prev_time)
            times.append(new_time - prev_time)
            prev_time = new_time
            rate.sleep()
        print('Episode mean dt =', np.mean(times))


def load_saved_agent(env, main_camera_crop, grayscale):
    demo_task = FLAGS.demo_task or FLAGS.eval_task
    demos_file, ckpt_dir, _ = train_utils.set_paths(demo_task)
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
        grayscale=FLAGS.grayscale)

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
    val_losses = train_utils.eval_on_valset(
          dataset, agent, FLAGS.regression_loss, FLAGS.l2_weight)
    test_set_size = FLAGS.test_set_size
    if test_set_size > 0:
        test_set_start = FLAGS.test_set_start or FLAGS.max_demos_to_load
        test_dataset = train_utils.prepare_demos(
            demos_file, FLAGS.input_type, test_set_start + test_set_size,
            FLAGS.max_demo_length, augment_frames=False, agent=agent,
            split_dir=None,
            val_size=test_set_size / (test_set_start + test_set_size),
            val_full_episodes=True, reset_agent_stats=False)
        test_losses = train_utils.eval_on_valset(
            test_dataset, agent, FLAGS.regression_loss, FLAGS.l2_weight)

    return agent, dataset, ckpt_dir


def main(_):
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    eval_task = FLAGS.eval_task
    visible_state_features = prl_ur5_utils.get_visible_features_for_task(
        eval_task, FLAGS.visible_state)
    demo_task = FLAGS.demo_task or FLAGS.eval_task
    _, ckpt_dir, _ = train_utils.set_paths(demo_task)
    timestamp = robotamer_utils.get_timestamp()
    eval_id = f'_{FLAGS.eval_id}' if FLAGS.eval_id else ''
    dataset_path = os.path.join(
        ckpt_dir, 'real_robot_eval', f'evalPush_{timestamp}{eval_id}.pkl')
    try:
        env = env_utils.init_env(
            eval_task, FLAGS.sim, FLAGS.arm, FLAGS.input_type,
            visible_state_features,
            num_input_frames=FLAGS.num_input_frames,
            crop=FLAGS.main_camera_crop,
            image_size=FLAGS.image_size,
            grayscale=FLAGS.grayscale,
            offline_dataset_path=FLAGS.offline_dataset_path,
            out_dataset_path=dataset_path,
            task_version=FLAGS.task_version)
        agent, demo_dataset, ckpt_dir = load_saved_agent(
            env, FLAGS.main_camera_crop, FLAGS.grayscale)

        start_episode(env, agent)
    except rospy.ROSInterruptException:
        print('Exiting')

 
if __name__ == '__main__':
    app.run(main)
