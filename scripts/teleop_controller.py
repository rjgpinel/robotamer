import functools
import gym
import os
import rospy
import robotamer.envs

import numpy as np
from absl import app
from absl import flags

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


def teleop_callback(data, env, x_scale=0.1, y_scale=0.1, z_scale=0.05, x_rot_scale=0.15, y_rot_scale=0.15, z_rot_scale=0.15):
    print('Received', data)
    print('eef', env.robot.eef_pose()[0])
    left_joy_left = data.axes[0]
    left_joy_up = data.axes[1]
    right_joy_left = data.axes[3]
    right_joy_up = data.axes[4]
    y_rot = ((data.axes[2] - 1.0)/2 - (data.axes[5] - 1.0)/2)/2
    x_rot = data.buttons[5]/2 - data.buttons[4]/2

    start = data.buttons[3]  # Y
    done = data.buttons[2]  # X
    discard = data.buttons[6] # back
    open_gripper = data.buttons[1]  # B
    close_gripper = data.buttons[0]  # A

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

    print('Sending', action)
    if start:
        obs = env.render()
        print('Observation fields', obs.keys())
    if done:
        print('Finished episode; Resetting arm')
        obs = env.reset()
        print('Observation fields', obs.keys())
        # dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
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


def main(_):
    try:
        if FLAGS.sim:
            cam_list = []
        elif FLAGS.arm == 'right':
            cam_list = ['left_camera', 'spare_camera']
        else:
            cam_list = ['bravo_camera', 'charlie_camera']
        env = gym.make(f'RealRobot-Pick-{FLAGS.task_version}',
                       cam_list=cam_list,
                       arm=FLAGS.arm,
                       version=FLAGS.task_version,
                       depth=not FLAGS.sim)

        real_obs = env.reset()
        print('Cartesian pose', env.robot.eef_pose())
        #print('Config', env.env._get_current_config())

        x_scale = y_scale = 0.05
        env_step_callback = functools.partial(
            teleop_callback, env=env, x_scale=x_scale,
            y_scale=y_scale)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')
        print('Observation fields', real_obs.keys())

        rospy.spin()
    except rospy.ROSInterruptException:
        print('Exiting')


if __name__ == '__main__':
    app.run(main)


