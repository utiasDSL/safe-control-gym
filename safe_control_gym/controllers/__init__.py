"""Register controllers.

"""
from safe_control_gym.utils.registration import register

register(id="mpc",
         entry_point="safe_control_gym.controllers.mpc.mpc:MPC",
         config_entry_point="safe_control_gym.controllers.mpc:mpc.yaml")

register(id="linear_mpc",
         entry_point="safe_control_gym.controllers.mpc.linear_mpc:LinearMPC",
         config_entry_point="safe_control_gym.controllers.mpc:linear_mpc.yaml")

register(id="gp_mpc",
         entry_point="safe_control_gym.controllers.mpc.gp_mpc:GPMPC",
         config_entry_point="safe_control_gym.controllers.mpc:gp_mpc.yaml")

register(id="mpsc",
         entry_point="safe_control_gym.controllers.mpsc.mpsc:MPSC",
         config_entry_point="safe_control_gym.controllers.mpsc:mpsc.yaml")

register(id="ppo",
         entry_point="safe_control_gym.controllers.ppo.ppo:PPO",
         config_entry_point="safe_control_gym.controllers.ppo:ppo.yaml")

register(id="safe_explorer_ppo",
         entry_point="safe_control_gym.controllers.safe_explorer.safe_ppo:SafeExplorerPPO",
         config_entry_point="safe_control_gym.controllers.safe_explorer:safe_ppo.yaml")
