from gym.envs.registration import register

register(
    id='env-v0',
    entry_point='safe_control_gym.envs:EnvClassName',
)
