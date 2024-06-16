"""Register environments."""

from safe_control_gym.utils.registration import register

register(
    id="quadrotor",
    entry_point="safe_control_gym.envs.quadrotor:Quadrotor",
    config_entry_point="safe_control_gym.envs:quadrotor.yaml",
)

register(
    id="firmware",
    entry_point="safe_control_gym.envs.firmware_wrapper:FirmwareWrapper",
    config_entry_point="safe_control_gym.controllers.firmware:firmware.yaml",
)
