'''Register safety filters. '''

from safe_control_gym.utils.registration import register


register(id='linear_mpsc',
         entry_point='safe_control_gym.safety_filters.mpsc.linear_mpsc:LINEAR_MPSC',
         config_entry_point='safe_control_gym.safety_filters.mpsc:mpsc.yaml')
