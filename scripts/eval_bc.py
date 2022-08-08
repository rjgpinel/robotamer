from absl import flags
import tensorflow as tf
from rrlfd import prl_ur5_utils
from rrlfd.bc import bc_agent
from rrlfd.bc import train_utils

from robotamer.envs.pick import PickEnv
from robotamer.core.pushing_utils import reset_env 
from sensor_msgs.msg import Joy


flags.DEFINE_string('input_type', 'rgb_left_camera', 'Input modality.')
ags.DEFINE_boolean('binary_grip_action', False,
                     'If True, use open/close action space for gripper. Else '
                     'use gripper velocity.')
flags.DEFINE_boolean('grip_action_from_state', False,
                     'If True, use gripper state as gripper action.')
flags.DEFINE_boolean('zero_action_keeps_state', False,
                     'If True, convert a zero-action in a demonstration to '
                     'maintain gripper state (as opposed to opening). Only '
                     'makes sense when not using grip_action_from_state.')
flags.DEFINE_boolean('early_closing', False,
                     'If True, clone gripper closing action in advance.')
flags.DEFINE_enum('action_norm', 'unit', ['none', 'unit', 'zeromean_unitvar'],
                  'Which normalization to apply to actions.')
flags.DEFINE_enum('signals_norm', 'none', ['none', 'unit', 'zeromean_unitvar'],
                  'Which normalization to apply to signal observations.')

flags.DEFINE_string('last_activation', None,
                    'Activation function to apply to network output, if any.')
flags.DEFINE_list('fc_layer_sizes', [],
                  'Sizes of fully connected layers to add on top of bottleneck '
                  'layer, if any.')
flags.DEFINE_integer('num_input_frames', 3,
                     'Number of frames to condition policy on.')
flags.DEFINE_integer('image_size', None, 'Size of rendered images.')
flags.DEFINE_integer('crop_margin_size', 16,
                     'If crop_frames is True, the number of pixels to crop '
                     'from each dimension.')

flags.DEFINE_boolean('crop_frames', True,
                     'If True, crop input frames by crop_margin_size pixels in '
                     'H and W.')
flags.DEFINE_boolean('augment_frames', True,
                     'If True, augment images by scaling, cropping and '
                     'rotating.')
flags.DEFINE_list('target_offsets', [0, 10, 20, 30],
                  'Offsets in time for actions to predict in behavioral '
                  'cloning.')

flags.DEFINE_string('visible_state', 'image',
                    'Which scalar features to condition the policy on.')


flags.DEFINE_integer('seed', 0, 'Experiment seed.')
flags.DEFINE_integer('eval_seed', 1,
                     'Seed for environment in which trained policy is '
                     'evaluated.')
flags.DEFINE_boolean('increment_eval_seed', False,
                     'If True, increment eval seed after each eval episode.')


# Flags for setting paths automatically.
flags.DEFINE_string('top_dir', None,
                    'If set, unset paths will be set relative to this '
                    'directory.')
flags.DEFINE_string('dataset_origin', '', 'Name of subdirectory for dataset.')
flags.DEFINE_string('dataset', None,
                    'Filename of demonstration dataset, of form '
                    's<seed>_e<num_episodes>.')
flags.DEFINE_string('exp_id', '', 'Experiment ID to add to output paths.')
flags.DEFINE_string('job_id', '', 'Job ID to add to output paths.')

flags.DEFINE_string('ckpt_dir', None,
                    'If set, directory for model checkpoints.')

flags.DEFINE_integer('max_demos_to_load', None,
                     'Maximum number of demos from demos_file (in order) to '
                     'use.')

flags.DEFINE_integer('num_eval_episodes', 100,
                     'If eval_task is set, number of episodes to evaluate '
                     'trained policy for.')

def ObservationStacker:

    def __init__(self, stack_length):
        self.observations = []
        self.stack_length = stack_length

    def reset(self, obs=None):
        self.observations = []
        if obs is not None:
            self.append(obs)

    def append(self, obs):
        self.observations.append(obs)
        self.observations = self.observations[-self.stack_length:]

    @property
    def current_observation(self):
        return np.array(self.observations[-self.stack_length:])



def predict_actions(env, obs_stack, agent):
    action = agent.get_action(obs, obs_stack.current_observation, env)


def end_episode(teleop, env, dataset, obs_stack):
    if teleop.buttons[1] or teleop.buttons[2]:  # B, X
        dataset.flag_success(teleop.buttons[2])
        dataset.save()
        reset_env(env)
        start_episode(env, dataset, obs_stack)


def start_episode(env, dataset, obs_stack):
    teleop = None
    print('Waiting for episode start')
    # Wait to receive a first image after a reset.
    while teleop is None or not teleop.buttons[0]:  # A
        teleop = rospy.wait_for_message('reset_teleop', Joy)
        obs = env.env.render()
        dataset.reset(obs)
        obs_stack.reset(obs)
        print('Starting the episode')


def load_saved_agent():
    demos_file, ckpt_dir, summary_dir = train_utils.set_paths(FLAGS.demo_task)
    dataset = train_utils.prepare_demos(
        demos_file, FLAGS.input_type, FLAGS.max_demos_to_load,
        FLAGS.max_demo_length, FLAGS.augment_frames, agent, ckpt_dir,
        FLAGS.val_size, FLAGS.val_full_episodes)

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
    return agent


def main():
    os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    agent = load_saved_agent()
    obs_stack = ObservationStacker(agent.num_input_frames)

    env = gym.make("RealRobot-Pick-v0",
                   cam_list=["left_camera", "spare_camera"],
                   arm='right',
                   depth=False)
    # eval_policy(bc_agent, env)
    rospy.init_node('bc_policy')  # receives observations (from gym?),
    rate = rospy.Rate(5)
    real_obs = reset_env(env)
    
    timestamp = utils.get_timestamp()
    dataset_path = os.path.join(os.environ['TOP_DATA_DIR'],
                                f'rrlfd/evalPush_{timestamp}.pkl')
    eval_dataset = datasets.EpisodeDataset(dataset_path)
    success_topic = 'success_teleop'
    # Failure conditions: cube hits tape, cube exits action space
    # What about passing through target region but not stopping?
    # -> consider it a success, assuming success detection is instantaneous
    teleop = None
    # TODO: Make sure to also reset stacked frames
    end_episode_callback = functools.partial(
        end_episode, env=env, dataset=dataset)
    rospy.Subscriber(success_topic, Joy, end_episode_callback, queue_size=1)

    start_episode(env, dataset, obs_stack)
    while not rospy.is_shutdown():
      
        # Should this node handle the control flow?
        # Always publish actions at 5Hz, they will simply be ignored by 
        predict_actions(env, obs_stack, agent)
        rate.sleep()
  
 
if __name__ == '__main__':
    main()
