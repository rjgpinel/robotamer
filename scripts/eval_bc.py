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
from robotamer.core import utils
from robotamer.envs.pick import PickEnv
from robotamer.rrlfd import observations
from sensor_msgs.msg import Joy


# flags.DEFINE_string('input_type', 'rgb_left_camera', 'Input modality.')
# flags.DEFINE_boolean('binary_grip_action', False,
#                      'If True, use open/close action space for gripper. Else '
#                      'use gripper velocity.')
# flags.DEFINE_boolean('grip_action_from_state', False,
#                      'If True, use gripper state as gripper action.')
# flags.DEFINE_boolean('zero_action_keeps_state', False,
#                      'If True, convert a zero-action in a demonstration to '
#                      'maintain gripper state (as opposed to opening). Only '
#                      'makes sense when not using grip_action_from_state.')
# flags.DEFINE_boolean('early_closing', False,
#                      'If True, clone gripper closing action in advance.')
# flags.DEFINE_enum('action_norm', 'unit', ['none', 'unit', 'zeromean_unitvar'],
#                   'Which normalization to apply to actions.')
# flags.DEFINE_enum('signals_norm', 'none', ['none', 'unit', 'zeromean_unitvar'],
#                   'Which normalization to apply to signal observations.')
# 
# flags.DEFINE_string('last_activation', None,
#                     'Activation function to apply to network output, if any.')
# flags.DEFINE_list('fc_layer_sizes', [],
#                   'Sizes of fully connected layers to add on top of bottleneck '
#                   'layer, if any.')
# flags.DEFINE_integer('num_input_frames', 3,
#                      'Number of frames to condition policy on.')
# flags.DEFINE_integer('image_size', None, 'Size of rendered images.')
# flags.DEFINE_integer('crop_margin_size', 16,
#                      'If crop_frames is True, the number of pixels to crop '
#                      'from each dimension.')
# 
# flags.DEFINE_boolean('crop_frames', True,
#                      'If True, crop input frames by crop_margin_size pixels in '
#                      'H and W.')
# flags.DEFINE_boolean('augment_frames', True,
#                      'If True, augment images by scaling, cropping and '
#                      'rotating.')
# flags.DEFINE_list('target_offsets', [0, 10, 20, 30],
#                   'Offsets in time for actions to predict in behavioral '
#                   'cloning.')
# 
# flags.DEFINE_string('visible_state', 'image',
#                     'Which scalar features to condition the policy on.')
# flags.DEFINE_string('network', 'resnet18_narrow32', 'Policy network to train.')
# flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train for.')
# 
# flags.DEFINE_enum('optimizer', 'adam', ['adam', 'rmsprop'],
#                   'Keras optimizer for training.')
# flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training.')
# flags.DEFINE_float('weight_decay', 0., 'Weight decay for training.')
# flags.DEFINE_float('val_size', 0.05,
#                    'Amount of data to validate on. If < 1, the fraction of '
#                    'total loaded data points. Else the number of data points.')
# flags.DEFINE_boolean('val_full_episodes', True,
#                      'If True, split data into train and validation on an '
#                      'episode basis. Else split by individual time steps.')
# flags.DEFINE_integer('test_set_size', 0,
#                      'Number of additional demonstrations on which to evaluate '
#                      'final model.')
# flags.DEFINE_integer('test_set_start', None,
#                      'Where in the dataset to start test set.')
# 
# flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
# flags.DEFINE_integer('seed', 0, 'Experiment seed.')
# flags.DEFINE_integer('eval_seed', 1,
#                      'Seed for environment in which trained policy is '
#                      'evaluated.')
# flags.DEFINE_boolean('increment_eval_seed', False,
#                      'If True, increment eval seed after each eval episode.')
# 
# 
# # Flags for setting paths automatically.
# flags.DEFINE_string('top_dir', None,
#                     'If set, unset paths will be set relative to this '
#                     'directory.')
# flags.DEFINE_string('dataset_origin', '', 'Name of subdirectory for dataset.')
# flags.DEFINE_string('dataset', None,
#                     'Filename of demonstration dataset, of form '
#                     's<seed>_e<num_episodes>.')
# flags.DEFINE_string('exp_id', '', 'Experiment ID to add to output paths.')
# flags.DEFINE_string('job_id', '', 'Job ID to add to output paths.')
# 
# # Flags for setting paths manually.
# flags.DEFINE_string('demos_file', None,
#                     'Pickle file from which to read demonstration data.')
# flags.DEFINE_string('ckpt_dir', None,
#                     'If set, directory for model checkpoints.')
# 
# flags.DEFINE_integer('max_demos_to_load', None,
#                      'Maximum number of demos from demos_file (in order) to '
#                      'use.')
# flags.DEFINE_string('demo_task', None,
#                     'Task used to gather demos in dataset, if different from '
#                     'eval_task.')
# flags.DEFINE_integer('num_eval_episodes', 100,
#                      'If eval_task is set, number of episodes to evaluate '
#                      'trained policy for.')
# 
# flags.DEFINE_boolean('clip_actions', False,
#                      'If True, clip actions to unit interval before '
#                      'normalization.')

# Robotamer flags.
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


class OfflineDataset:
    """Replaces image observations when testing in sim."""

    def __init__(self, data_path):
        self.episodes = []
        with open(data_path, 'rb') as f:
            while True:
                try:
                    self.episodes.append(pickle.load(f))
                except EOFError:
                    break
        self.e = 0
        self.t = 0
        self.env = self

    def render(self):
        return self.step()

    def step(self, unused_action=None):
        obs = self.episodes[self.e]['observations'][self.t]
        current_length = len(self.episodes[self.e]['observations'])
        self.t = (self.t + 1) % current_length
        return obs

    def reset(self):
        self.e = (self.e + 1) % len(self.episodes)
        self.t = 0
        return self.step()

    @property
    def action_space(self):
        return gym.spaces.Box(low=np.array([-0.05, -0.05]),
                              high=np.array([0.05, 0.05]))


def predict_actions(env, eval_dataset, obs_stack, agent, obs_dataset=None):
    # TODO: Define history and newest obs in obs_stack.
    stacked_obs = obs_stack.current_observation
    new_obs = copy.deepcopy(stacked_obs)
    img_key = obs_stack.image_key_out
    obs_hist = [{img_key: new_obs[img_key][t]}
                for t in range(len(new_obs[img_key]) - 1)]
    new_obs = {k: v[-1] if k == img_key else v for k, v in new_obs.items()}
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
        _ = env.reset()
        dataset.flag_success(teleop.buttons[2])
        dataset.save()
        # start_episode(env, dataset, obs_stack, agent, obs_dataset)


def start_episode(env, dataset, obs_stack, agent, obs_dataset):
    print('Waiting for episode start')
    # Wait to receive a first image after a reset.
    rate = rospy.Rate(5)
    while not env.is_ready:
        rate.sleep()
    # while teleop is None or not teleop.buttons[0]:  # A
    #     print('Waiting for teleop')
    #     teleop = rospy.wait_for_message('/joy_teleop', Joy, timeout=2)
    #     print('Received teleop', teleop)
    # env.is_ready = True
    # print('Env is ready')
    # if obs_dataset is None:
    #     obs = env.env.render()
    # else:
    #     obs = obs_dataset.reset()
    # dataset.reset(obs)
    # obs_stack.reset(obs)
    # print('Starting the episode')
    
    prev_time = time.time()
    while env.is_ready and not rospy.is_shutdown():
        predict_actions(env, dataset, obs_stack, agent, obs_dataset)
        new_time = time.time()
        print('dt =', new_time - prev_time)
        prev_time = new_time
        rate.sleep()
    print('Env is no longer ready')


def load_saved_agent(env, main_camera, main_camera_crop, grayscale):
    demos_file, ckpt_dir, summary_dir = train_utils.set_paths(FLAGS.demo_task)
    demo_task = FLAGS.demo_task or FLAGS.eval_task
    visible_state_features = prl_ur5_utils.get_visible_features_for_task(
      demo_task, FLAGS.visible_state)
    image_size = FLAGS.image_size

    # TODO: Make sure network weights are really loaded.
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
        init_scheme=FLAGS.weight_init_scheme)
    ckpt_to_load = os.path.join(ckpt_dir, 'ckpt')
    print('Loading from', ckpt_to_load)
    agent.load(ckpt_to_load)

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

    return agent, obs_stack, dataset, ckpt_dir


def main(_):
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    if FLAGS.sim:
        # TODO: Get camera observations from an unprocessed dataset to test
        # policy.
        cam_list = []
        main_camera = 'left' if FLAGS.arm == 'right' else 'charlie'
    elif FLAGS.arm == 'right':
        cam_list = ['left_camera', 'spare_camera']
        main_camera = 'left'
    else:
        cam_list = ['bravo_camera', 'charlie_camera']
        main_camera = 'charlie'
    if FLAGS.sim:
        # rospy.init_node('bc_policy')
        obs_dataset = OfflineDataset(FLAGS.offline_dataset_path)
    else:
        obs_dataset = None
    print('Using cameras', cam_list)
    env = gym.make(f'RealRobot-Cylinder-Push-{FLAGS.task_version}',
                   cam_list=cam_list,
                   arm=FLAGS.arm,
                   version=FLAGS.task_version,
                   depth=False)
    env.is_ready = False

    agent, obs_stack, demo_dataset, ckpt_dir = load_saved_agent(
        env, main_camera, FLAGS.main_camera_crop, FLAGS.grayscale)

    # rate = rospy.Rate(5)
    real_obs = env.reset()
    
    timestamp = utils.get_timestamp()
    dataset_path = os.path.join(
        ckpt_dir, 'real_robot_eval', f'evalPush_{timestamp}.pkl')
    eval_dataset = datasets.EpisodeDataset(dataset_path)
    # Failure conditions: cube hits tape, cube exits action space
    # What about passing through target region but not stopping?
    # -> consider it a success, assuming success detection is instantaneous
    # TODO: Make sure to also reset stacked frames
    callback = functools.partial(
        teleop_callback, env=env, dataset=eval_dataset, obs_stack=obs_stack,
        agent=agent, obs_dataset=obs_dataset)
    rospy.Subscriber('joy_teleop', Joy, callback, queue_size=1)

    # prev_time = time.time()
    while not rospy.is_shutdown():
        start_episode(env, eval_dataset, obs_stack, agent, obs_dataset)
    #   
    #     # Should this node handle the control flow?
    #     # Always publish actions at 5Hz, they will simply be ignored by 
    #     predict_actions(env, eval_dataset, obs_stack, agent, obs_dataset)
    #     new_time = time.time()
    #     print('dt =', new_time - prev_time)
    #     prev_time = new_time
    #     rate.sleep()

    # TODO: remove this line
    # rospy.spin()
    # try:
    #     # TODO: wrap all the setup above in this block
    #     # (clean up and separate into fns)
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     print('Exiting')
  
 
if __name__ == '__main__':
    app.run(main)
