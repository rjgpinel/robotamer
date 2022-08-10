"""Convert observations to format expected by policy."""

import copy

import numpy as np
from PIL import Image


class ObservationConverter:

    def __init__(self, camera, visible_keys, stack_length=1, image_size=None,
                 grayscale=False, crop=None):
        self.observations = []
        self.visible_keys = visible_keys
        self.stack_length = stack_length
        self.image_size = (image_size, image_size)
        self.grayscale = grayscale
        self.crop = crop
        self.image_key_in = f'rgb_{camera}_camera'
        self.image_key_out = 'rgb'

    def process_observation(self, obs):
        processed_obs = {}
        for k in self.visible_keys:
            v = obs[k]
            if k in ['gripper_pos', 'gripper_trans_velocity']:
                 v = v[:2]
            processed_obs[k] = copy.deepcopy(v)
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
        processed_obs[self.image_key_out] = img
        return processed_obs


    def reset(self, obs=None):
        self.observations = []
        if obs is not None:
            self.append(obs)

    def append(self, obs):
        obs = self.process_observation(obs)
        self.observations.append(obs)
        self.observations = self.observations[-self.stack_length:]

    @property
    def current_observation(self):
        stacked_image = np.array(
            [obs[self.image_key_out]
             for obs in self.observations[-self.stack_length:]])
        # TODO: history for scalar fields?
        scalar_fields = {k: self.observations[-1][k] for k in self.visible_keys}
        obs = {
            self.image_key_out: stacked_image,
            **scalar_fields}
        return obs



