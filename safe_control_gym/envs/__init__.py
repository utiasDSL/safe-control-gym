"""Register environments."""

from gymnasium import register

register(
    id="quadrotor",
    entry_point="safe_control_gym.envs.quadrotor:Quadrotor",
)

register(
    id="firmware",
    entry_point="safe_control_gym.envs.firmware_wrapper:FirmwareWrapper",
    max_episode_steps=900,  # 30 seconds * 30 Hz
)
