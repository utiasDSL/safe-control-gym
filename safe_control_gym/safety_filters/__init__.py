"""Register safety filters.

"""
from safe_control_gym.utils.registration import register


register(id="mpsc_sf",
         entry_point="safe_control_gym.safety_filters.mpsc.mpsc:MPSC",
         config_entry_point="safe_control_gym.safety_filters.mpsc:mpsc.yaml")

register(id="cbf_sf",
         entry_point="safe_control_gym.safety_filters.cbf.cbf_qp:CBF_QP",
         config_entry_point="safe_control_gym.safety_filters.cbf:cbf_qp.yaml")

register(id="cbf_nn_sf",
         entry_point="safe_control_gym.safety_filters.cbf.cbf_qp_nn:CBF_QP_NN",
         config_entry_point="safe_control_gym.safety_filters.cbf:cbf_qp_nn.yaml")

register(id="cbf_sos_sf",
         entry_point="safe_control_gym.safety_filters.cbf.cbf_qp_sos:CBF_QP_SOS",
         config_entry_point="safe_control_gym.safety_filters.cbf:cbf_qp_sos.yaml")
