"""Register safety filters.

"""
from safe_control_gym.utils.registration import register


register(id="mpsc",
         entry_point="safe_control_gym.safety_filters.mpsc.mpsc:MPSC",
         config_entry_point="safe_control_gym.safety_filters.mpsc:mpsc.yaml")
