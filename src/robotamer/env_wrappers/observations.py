"""Environment wrappers for preprocessing observations."""

import copy
import gym
import numpy as np
from PIL import Image


class ImageObservationWrapper(gym.Wrapper):
    """Wrapper supporting cropped, resized and grayscaled image observations."""

    def __init__(self, env, image_key_in, image_key_out=None, crop=None,
                 image_size=None, grayscale=False):
        super().__init__(env)
        self.image_key_in = image_key_in
        self.image_key_out = image_key_out or image_key_in
        # Crop to apply before resize.
        self.crop = crop
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int)
            else image_size)
        self.grayscale = grayscale
    
    def _wrap_observation(self, obs):
        wrapped_obs = {}
        img = copy.deepcopy(obs[self.image_key_in])
        if self.crop is not None:
            crop = self.crop
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
        if self.grayscale or self.image_size:
            img = Image.fromarray(img)
            if self.grayscale:
                img = img.convert('L')
            if self.image_size:
                img = img.resize(size=self.image_size)
            img = np.array(img)
        wrapped_obs[self.image_key_out] = img
        return wrapped_obs
        
    def reset(self):
        obs = self.env.reset()
        return self._wrap_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._wrap_observation(obs), reward, done, info


class ImageStackingWrapper(gym.Wrapper):

    def __init__(self, env, image_key, stack_length):
        super().__init__(env)
        self.observations = []
        self.image_key = image_key
        self.stack_length = stack_length

    def reset(self):
        obs = self.env.reset()
        self.observations = [obs]
        return self.current_observation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.observations.append(obs)
        return self.current_observation, reward, done, info

    @property
    def current_observation(self):
        stacked_image = np.array(
            [obs[self.image_key]
             for obs in self.observations[-self.stack_length:]])
        obs = {k: stacked_image if k == self.image_key else v
               for k, v in self.observations[-1].items()}
        return obs

    @property
    def newest_observation(self):
        return self.observations[-1]

    @property
    def observation_history(self):
        return self.observations[:-1]
