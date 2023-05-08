'''Register safety filters.'''

from safe_control_gym.utils.registration import register


register(idx='linear_mpsc',
         entry_point='safe_control_gym.safety_filters.mpsc.linear_mpsc:LINEAR_MPSC',
         config_entry_point='safe_control_gym.safety_filters.mpsc:mpsc.yaml')

register(idx='cbf',
         entry_point='safe_control_gym.safety_filters.cbf.cbf:CBF',
         config_entry_point='safe_control_gym.safety_filters.cbf:cbf.yaml')

register(idx='cbf_nn',
         entry_point='safe_control_gym.safety_filters.cbf.cbf_nn:CBF_NN',
         config_entry_point='safe_control_gym.safety_filters.cbf:cbf_nn.yaml')
