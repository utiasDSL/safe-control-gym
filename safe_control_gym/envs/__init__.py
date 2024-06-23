"""Register environments."""

from gymnasium import register

register(
    id="drone_sim",
    entry_point="safe_control_gym.envs.drone_sim:DroneSim",
)

register(
    id="drone_racing-v0",
    entry_point="safe_control_gym.envs.drone_racing_env:DroneRacingEnv",
    max_episode_steps=900,  # 30 seconds * 30 Hz
)
