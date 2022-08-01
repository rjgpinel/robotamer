import functools
import gym
import os
import pickle
import rospy
import time
import robotamer.envs

import numpy as np
from math import pi
from datetime import datetime

from robotamer.envs.pick import PickEnv
from sensor_msgs.msg import Joy


PUSHING_START_CONFIG = [
        # 0.9326981650333579, -1.752163298993259, 1.7692008154315744, -1.070960116650423, 2.19026060548725, 2.3614391975469964
        # 0.9326981650333579, -1.752163298993259, 1.7692008154315744, -1.070960116650423, 2.19026060548725, 0
        # 3.8cm from the table (z = 0.05 in command space)
        # 0.8726706768437351, -1.6906466333371597, 1.636904689457615, -1.0095888375470192, 2.1850710910097035, 0
        # 3.8cm from the table (z = 0.05 in command space)
        # 0.8849127317995276, -1.668861578637351, 1.631950419976694, -1.0844151260136474, 2.2167711025394112, -0.07978964167831126
        # From the real robot
        0.8707625865936279, -1.7185638586627405, 1.6314215660095215, -0.9090965429889124, 2.146097183227539, 0.8783294558525085
]


class Dataset:

    def __init__(self, path):
        self.path = path
        self.episodes = []
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

    def reset(self, obs):
        # if self.episodes:
        #     for k in self.episodes[-1]:
        #         self.episodes[-1][k] = np.array(self.episodes[-1][k])
        print('Starting episode', len(self.episodes) + 1)
        self.episodes.append({'observations': [obs], 'actions': []})

    def append(self, act, next_obs):
        self.episodes[-1]['actions'].append(act)
        self.episodes[-1]['observations'].append(next_obs)

    def discard_episode(self):
        if self.episodes:
            self.episodes = self.episodes[:-1]
            print('Discarded episode', len(self.episodes) + 1)
        else:
            print('No episodes to discard')

    def save(self):
        episodes = self.episodes
        # if episodes and not episodes[-1]['actions']:
        #     # Leave out last empty episode.
        #     episodes = episodes[:-1]
        with open(self.path, 'wb') as f:
            pickle.dump(episodes, f)


def callback(data, env, dataset, x_scale=0.1, y_scale=0.1):
    print('Received', data)
    print('eef', env.robot.eef_pose()[0])
    joy_left = data.axes[0]
    joy_up = data.axes[1]
    done = data.buttons[2]
    discard = data.buttons[1]
    vx = y_scale * joy_up
    vy = x_scale * joy_left
    action = {
        "linear_velocity": np.array([vx, vy, 0.0]),
        "angular_velocity": np.array([0.0, 0.0, 0.0]),
        "grip_open": 1,
    }
    action_2d = [vx, vy]
    print('Sending', action)
    if done:
        dataset.save()
        print('Finished episode; Resetting arm')
        obs = reset_env(env)
        dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    elif discard:
        dataset.discard_episode()
        print('Resetting arm')
        obs = reset_env(env)
        dataset.reset(obs)
        print('Reset finished')
        print('Ready to receive joystick controls')
    else:
        real_obs = env.step(action)
        dataset.append(action_2d, real_obs)

def reset_arm(env):
    if env.arm_name == 'left':
        gripper_pos = [-0.40, 0, 0.1]
        gripper_orn = [pi, 0, pi / 2]
    else:
        gripper_pos = [0.40, 0, 0.1]
        gripper_orn = [pi, 0, -pi / 2]
    obs = env.reset(gripper_pos=gripper_pos, gripper_orn=gripper_orn)
    return obs


def reset_joints(env):
    obs = env.reset(joints=PUSHING_START_CONFIG)
    return obs


def reset_to_home(env):
    obs = env.reset(home_only=True)
    return obs


def reset_joints_and_eef(env):
    obs = env.reset(joints=PUSHING_START_CONFIG, gripper_pos=[0.4, 0, 0.04])
    return obs


def reset_eef(env):
    obs = env.reset(gripper_pos=[0.4, 0, 0.04])


def reset_env(env):
   obs = reset_joints(env)
   return obs


def main():
    try:
        env = gym.make("RealRobot-Pick-v0", cam_list=[], arm='right')
        # real_obs = reset_to_home(env)
        # pose = env.robot.eef_pose()
        # print('Original home pose', pose) 
        # import pdb; pdb.set_trace()

        # new_pos = pose[0]
        # new_pos[2] = 0.04  # 0.05
        # env.reset(gripper_pos=new_pos)
        # print('Home pose (z = 0.04)', env.robot.eef_pose())
        # import pdb; pdb.set_trace()

        real_obs = reset_env(env)
        print('Cartesian pose', env.robot.eef_pose())
        print('Config', env.env._get_current_config())
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # pose = env.robot.eef_pose()
        # print('pose before:', pose)
        # import pdb; pdb.set_trace()
        # obs = env.reset(gripper_pos=[0.4, 0., 0.025], gripper_orn=[pi, 0, 0])
        # reset_arm(env)
        # real_obs = reset_arm(env)
        # real_obs = reset_joints(env)
        # print('pose before:', env.robot.eef_pose())
        # for _ in range(4):
        #     action = {
        #         'linear_velocity': np.array([1, 0, 0]),
        #         'angular_velocity': np.array([0, 0, 0]),
        #         'grip_open': 1
        #     }
        #     env.step(action)
        # print('pose after:', env.robot.eef_pose())

        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        dataset_path = os.path.join(os.environ['TOP_DATA_DIR'],
                                    f'rrlfd/pushing_demos_sim_dev_{timestamp}.pkl')
        dataset = Dataset(dataset_path)
        dataset.reset(real_obs)
        env_step_callback = functools.partial(
            callback, env=env, dataset=dataset, x_scale=0.05, y_scale=0.05)
        rospy.Subscriber('joy_teleop', Joy, env_step_callback, queue_size=1)
        print('Ready to receive joystick controls')

        rospy.spin()
    except rospy.ROSInterruptException:
        dataset.save()


if __name__ == "__main__":
    main()


