import numpy as np
from gym.envs.registration import register

envs = [
    dict(
        id="RealRobot-Pick-v0",
        entry_point="robotamer.envs.pick:PickEnv",
        max_episode_steps=150,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-Stack-v0",
        entry_point="robotamer.envs.stack:StackEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-Push-v0",
        entry_point="robotamer.envs.push:PushEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-PushButtons-v0",
        entry_point="robotamer.envs.push_buttons:PushButtonsEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-NutAssembly-v0",
        entry_point="robotamer.envs.assembly:NutAssemblyEnv",
        max_episode_steps=1000,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-Rope-v0",
        entry_point="robotamer.envs.rope:RopeEnv",
        max_episode_steps=1000,
        reward_threshold=1.0,
    ),
    dict(
        id="RealRobot-Sweep-v0",
        entry_point="robotamer.envs.sweep:SweepEnv",
        max_episode_steps=1000,
        reward_threshold=1.0,
    ),
]


for env_dict in envs:
    name = env_dict["id"]
    register(**env_dict)
