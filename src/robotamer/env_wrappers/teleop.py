import gym
import rospy
import numpy as np

from sensor_msgs.msg import Joy


class TeleopWrapper(gym.Wrapper):
    """Use teleop commands to define success and to decide when to reset."""

    def __init__(self, env, obs_dataset=None, allow_teleop_actions=False,
                 x_scale=0.05, y_scale=0.05):
        super().__init__(env)
        self.is_ready = False
        self.obs_dataset = obs_dataset
        # If True, call env.step with joystick commands.
        self.allow_teleop_actions = allow_teleop_actions
        rospy.Subscriber('joy_teleop', Joy, self.teleop_callback, queue_size=1)
        self._rate = rospy.Rate(20)
        self._x_scale = x_scale
        self._y_scale = y_scale

    def step(self, action):
        if not self.is_ready:
            print('Called step when env is resetting')
        if self.obs_dataset is None:
            obs, reward, done, info = self.env.step(action)
        else:
            obs = self.obs_dataset.step()
            # TODO: Read from dataset if present.
            reward = 0
            done = False
            info = {}
        done = self.done
        info = {**info, **self.info}
        return obs, reward, done, info

    def reset(self):
        # Moves the arm to the starting position.
        self.env.reset()
        print('Waiting for episode start')
        while not self.is_ready:
            self._rate.sleep()
        if self.obs_dataset is None:
            obs = self.env.render()
        else:
            obs = self.obs_dataset.reset()
        return obs

    def teleop_callback(self, teleop):
        joy_left = teleop.axes[0]
        joy_up = teleop.axes[1]

        start = teleop.buttons[0]  # A
        failure = teleop.buttons[1]  # B
        success = teleop.buttons[2]  # X
        discard = teleop.buttons[3]  # Y
        if start:
            # if self.obs_dataset is None:
            #     obs = self.env.render()
            #     print('Observation fields', obs)
            # else:
            #     obs = self.obs_dataset.reset()
            # dataset.reset(obs)
            # obs_stack.reset(obs)
            self.info = {}
            self.is_ready = True
            self.done = False
            print('Starting the episode')
        elif failure or success:
            self.is_ready = False
            self.done = True
            self.info = {'success': success, 'discard': False}
            # Moves the arm to the starting position.
            # self.env.reset()
            # dataset.flag_success(teleop.buttons[2])
            # dataset.save()
        elif discard:
            # Something else went wrong (not because of the policy): discard.
            self.is_ready = False
            self.done = True
            self.info = {'success': False, 'discard': True}
            # Moves the arm to the starting position.
            # self.env.reset()
            # dataset.discard_episode()
            # obs_stack.reset()
        elif self.allow_teleop_actions:
            vx = self._x_scale * joy_up
            vy = self._y_scale * joy_left
            action = {
                'linear_velocity': np.array([vx, vy, 0.0]),
                'angular_velocity': np.array([0.0, 0.0, 0.0]),
            }
            self.step(action)
