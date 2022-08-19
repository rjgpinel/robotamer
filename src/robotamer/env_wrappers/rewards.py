import gym


class SparseRewardWrapper(gym.Wrapper):

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = float('success' in info and info['success'])
        return obs, reward, done, info
