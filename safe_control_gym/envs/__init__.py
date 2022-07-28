"""Register environments.

"""
from safe_control_gym.utils.registration import register

register(id="quadrotor",
         entry_point="safe_control_gym.envs.gym_pybullet_drones.quadrotor:Quadrotor",
         config_entry_point="safe_control_gym.envs.gym_pybullet_drones:quadrotor.yaml")
