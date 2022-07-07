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
]

num_variants = {"RealRobot-PushButtons-v0": 5220, "RealRobot-Stack-v0": 200}

for env_dict in envs:
    name = env_dict["id"]
    register(**env_dict)
    if name in num_variants.keys():
        num_v = num_variants[name]
        for var in range(num_v):
            var_env_dict = env_dict.copy()
            clean_env_name = "-".join(name.split("-")[1:])
            var_env_dict["id"] = f"RealRobot-Var{var}-{clean_env_name}"
            register(**var_env_dict)

