import gym
import rospy
import numpy as np

from sensor_msgs.msg import Joy


class TeleopWrapper(gym.Wrapper):
    """Use teleop commands to define success and to decide when to reset."""

    def __init__(self, env, allow_teleop_actions=False, x_scale=0.05,
                 y_scale=0.05):
        super().__init__(env)
        self.is_ready = False
        # If True, call env.step with joystick commands.
        self.allow_teleop_actions = allow_teleop_actions
        rospy.Subscriber('joy_teleop', Joy, self.teleop_callback, queue_size=1)
        self._rate = rospy.Rate(20)
        self._x_scale = x_scale
        self._y_scale = y_scale

    def step(self, action):
        if not self.is_ready:
            print('Called step when env is resetting')
        obs, reward, done, info = self.env.step(action)
        # obs['rgb_charlie_camera'] = np.random.randint(
        #     256, size=(720, 480), dtype=np.uint8)
        # For overwriting success signal.
        info.pop('success', None)
        if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
            self.is_ready = False
            print('Time limit exceeded: waiting for success signal (X / B / Y)')
            while 'success' not in self.info:
                self._rate.sleep()
        done = done or self.done
        info = {**info, **self.info}
        return obs, reward, done, info

    def reset(self):
        # Moves the arm to the starting position.
        self.env.reset()
        print('Waiting for episode start')
        while not self.is_ready:
            self._rate.sleep()
        obs = self.env.render()
        # obs['rgb_charlie_camera'] = np.random.randint(
        #     256, size=(720, 480), dtype=np.uint8)
        return obs

    def teleop_callback(self, teleop):
        joy_left = teleop.axes[0]
        joy_up = teleop.axes[1]

        start = teleop.buttons[0]  # A
        failure = teleop.buttons[1]  # B
        success = teleop.buttons[2]  # X
        discard = teleop.buttons[3]  # Y
        if start:
            self.info = {}
            self.is_ready = True
            self.done = False
            print('Starting the episode')
        elif failure or success:
            self.is_ready = False
            self.done = True
            self.info = {'success': success, 'discard': False}
        elif discard:
            # Something else went wrong (not because of the policy): discard.
            self.is_ready = False
            self.done = True
            self.info = {'success': False, 'discard': True}
        elif self.allow_teleop_actions:
            vx = self._x_scale * joy_up
            vy = self._y_scale * joy_left
            action = {
                'linear_velocity': np.array([vx, vy, 0.0]),
                'angular_velocity': np.array([0.0, 0.0, 0.0]),
            }
            self.step(action)
