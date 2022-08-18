"""Environment wrappers for preprocessing observations."""

import abc
import copy
import gym
import numpy as np
from PIL import Image


class ObservationWrapper(gym.Wrapper, abc.ABC):
    """Base environment wrapper for modifying observations."""

    @abc.abstractmethod
    def _wrap_observation(self, obs):
        """Modify observation obs from wrapped environment."""

    def reset(self):
        obs = self.env.reset()
        return self._wrap_observation(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._wrap_observation(obs), reward, done, info


class VisibleStateWrapper(ObservationWrapper):
    """Wrapper for preprocessing scalar observation fields."""

    def __init__(self, env, visible_state_features, gripper_in_2d=False):
        super().__init__(env)
        self._visible_keys = visible_state_features
        self._gripper_in_2d = gripper_in_2d

    def _wrap_observation(self, obs):
        """Leave out keys not in visible keys and process gripper states."""
        wrapped_obs = {}
        for k in self._visible_keys:
            v = obs[k]
            if self._gripper_in_2d and k in [
                'gripper_pos', 'gripper_trans_velocity']:
                 v = v[:2]
            wrapped_obs[k] = copy.deepcopy(v)
        return wrapped_obs


class ImageObservationWrapper(ObservationWrapper):
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
        wrapped_obs = {
            k: copy.deepcopy(v) for k, v in obs.items()
            if k != self.image_key_in}
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
        

class ImageStackingWrapper(gym.Wrapper):
    """Wrapper to keep a history of image fields for stacking."""

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


class StaticDatasetWrapper(gym.Wrapper):
    """Env wrapper that replaces observations with fields from a dataset."""

    def __init__(self, obs_dataset):
        super().__init__(env)
        self._obs_dataset = obs_dataset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_dataset.step()
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        obs = self._obs_dataset.reset()
        return obs

