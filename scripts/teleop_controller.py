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
from sensor_msgs.msg import Joy


flags.DEFINE_bool('sim', False,
                  'If true (running in simulation), use proprioceptive '
                  'observations only. Else initialize cameras.')
flags.DEFINE_enum('arm', 'left', ['left', 'right'],
                  'Which arm to use.')
flags.DEFINE_enum('task_version', 'v0', ['v0', 'v1'],
                  'Which version of the task to use.')
FLAGS = flags.FLAGS


def teleop_callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
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
        obs = env.reset()
        print('Observation fields', obs.keys())
        # dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    elif discard:
        dataset.discard_episode()
        print('Resetting arm')
        obs = env.reset()
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
        }
        env.step(action)
    print('pose after:', env.robot.eef_pose())


def main(_):
    try:
        if FLAGS.sim:
            cam_list = []
        elif FLAGS.arm == 'right':
            cam_list = ['left_camera', 'spare_camera']
        else:
            cam_list = ['bravo_camera', 'charlie_camera']
        env = gym.make(f'RealRobot-Cylinder-Push-{FLAGS.task_version}',
                       cam_list=cam_list,
                       arm=FLAGS.arm,
                       version=FLAGS.task_version,
                       depth=False)

        real_obs = env.reset()
        print('Cartesian pose', env.robot.eef_pose())
        print('Config', env.env._get_current_config())

        timestamp = utils.get_timestamp()
        dataset_type = 'sim_' if FLAGS.sim else ''
        dataset_path = os.path.join(
            os.environ['TOP_DATA_DIR'],
             f'rrlfd/pushing_demos_{FLAGS.task_version}_{dataset_type}'
             f'{timestamp}.pkl')
        dataset = datasets.EpisodeDataset(dataset_path)

        if FLAGS.arm == 'right':
            x_scale = y_scale = 0.05
        else:
            x_scale = y_scale = -0.05
        env_step_callback = functools.partial(
            teleop_callback, env=env, dataset=dataset, x_scale=x_scale,
            y_scale=y_scale)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')
        print('Observation fields', real_obs.keys())

        rospy.spin()
    except rospy.ROSInterruptException:
        print('Exiting')


if __name__ == '__main__':
    app.run(main)


