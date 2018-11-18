from gym.envs.registration import register

register(
    id='smarthomeenv-v0',
    entry_point='hackathon.solution.smarthomeenv.smarthomeenv.envs:SmartHomeEnv',
)
