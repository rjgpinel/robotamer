import functools
import gym
import os
import rospy
import robotamer.envs

import numpy as np
from absl import app
from absl import flags

from robotamer.envs.pick import PickEnv
from robotamer.core import datasets
from robotamer.core import utils
from robotamer.core.pushing_utils import reset_env
from sensor_msgs.msg import Joy


flags.DEFINE_bool('sim', False,
                  'If true (running in simulation), use proprioceptive '
                  'observations only. Else initialize cameras.')
FLAGS = flags.FLAGS


def callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    print('eef', env.robot.eef_pose()[0])
    joy_left = data.axes[0]
    joy_up = data.axes[1]

    start = data.buttons[0]  # A
    discard = data.buttons[1]  # B
    done = data.buttons[2]  # X

    vx = y_scale * joy_up
    vy = x_scale * joy_left
    action = {
        'linear_velocity': np.array([vx, vy, 0.0]),
        'angular_velocity': np.array([0.0, 0.0, 0.0]),
        'grip_open': 1,
    }
    action_2d = np.array([vx, vy])
    print('Sending', action)
    if start:
        obs = env.env.render()
        dataset.reset(obs)
        print('Observation fields', obs.keys())
    if done:
        dataset.append_action(np.array([0., 0.]))
        dataset.save()
        print('Finished episode; Resetting arm')
        obs = reset_env(env)
        print('Observation fields', obs.keys())
        # dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    elif discard:
        dataset.discard_episode()
        print('Resetting arm')
        obs = reset_env(env)
        print('Observation fields', obs.keys())
        # dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
        real_obs, done, reward, info = env.step(action)
        dataset.append(action_2d, real_obs)


def test_displacement(env):
    print('pose before:', env.robot.eef_pose())
    for _ in range(4):
        action = {
            'linear_velocity': np.array([1, 0, 0]),
            'angular_velocity': np.array([0, 0, 0]),
            'grip_open': 1
        }
        env.step(action)
    print('pose after:', env.robot.eef_pose())


def main(_):
    try:
        cam_list = [] if FLAGS.sim else ['left_camera', 'spare_camera']
        env = gym.make('RealRobot-Pick-v0',
                       cam_list=cam_list,
                       arm='right',
                       depth=False)

        real_obs = reset_env(env)
        print('Cartesian pose', env.robot.eef_pose())
        print('Config', env.env._get_current_config())

        timestamp = utils.get_timestamp()
        dataset_path = os.path.join(os.environ['TOP_DATA_DIR'],
                                    f'rrlfd/pushing_demos_sim_dev_{timestamp}.pkl')
        dataset = datasets.EpisodeDataset(dataset_path)

        env_step_callback = functools.partial(
            callback, env=env, dataset=dataset, x_scale=0.05, y_scale=0.05)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')
        print('Observation fields', real_obs.keys())

        rospy.spin()
    except rospy.ROSInterruptException:
        print('Exiting')


if __name__ == '__main__':
    app.run(main)


