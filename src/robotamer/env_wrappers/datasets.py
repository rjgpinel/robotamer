import gym

from robotamer.core import datasets


class RecordEpisodesWrapper(gym.Wrapper):
    """Save episodes to an EpisodeDatset."""
    
    def __init__(self, env, dataset_path):
        super().__init__(env)
        self.dataset = datasets.EpisodeDataset(dataset_path)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.dataset.append(action, obs, reward, info)
        if done:
            if 'discard' in info and info['discard']:
                self.dataset.discard_episode()
            else:
                self.dataset.save()
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.dataset.reset(obs)
        return obs
        
