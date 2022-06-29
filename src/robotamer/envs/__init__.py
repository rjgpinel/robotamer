from gym.envs.registration import register

envs = [
    dict(
        id="Pick-v0",
        entry_point="robotamer.envs.pick:PickEnv",
        max_episode_steps=150,
        reward_threshold=1.0,
    ),
    dict(
        id="Push-v0",
        entry_point="robotamer.envs.push:PushEnv",
        max_episode_steps=600,
        reward_threshold=1.0,
    ),
]


for env_dict in envs:
    name = env_dict["id"]
    register(**env_dict)
    # domain randomization
    dr_name = f"DR-{name}"
    env_dict = env_dict.copy()
    env_dict["id"] = dr_name
    env_dict["kwargs"] = dict(domain_randomization=True)
    register(**env_dict)
