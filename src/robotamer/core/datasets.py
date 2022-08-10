import copy
import io
import os
import pickle

import numpy as np
from PIL import Image


class EpisodeDataset:

    def __init__(self, path):
        self.path = path
        self.episodes = []
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

    def compress_image(self, img_obs):
        pil_img = Image.fromarray(img_obs)
        img_buf = io.BytesIO()
        pil_img.save(img_buf, format='PNG')
        img_bytes = img_buf.getvalue()
        return img_bytes

    def compress_images(self, obs):
        obs = copy.deepcopy(obs)
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and len(v.shape) == 3:
                obs[k] = self.compress_image(v)
        return obs

    def reset(self, obs):
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

    def append_action(self, act):
        self.episodes[-1]['actions'].append(act)

    def save(self):
        with open(self.path, 'ab') as f:
            pickle.dump(self.episodes[-1], f)
        print('Finished saving to file')

    def flag_success(self, success):
        self.episodes[-1]['success'] = success


class OfflineDataset:
    """Replaces an env's image observations when testing in sim."""

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
