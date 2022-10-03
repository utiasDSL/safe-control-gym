'''Register explore policies. '''

from safe_control_gym.utils.registration import register

register(id='epsilon',
         entry_point='safe_control_gym.explore.base_explore:EpsilonGreedyExplore')

register(id='gaussian',
         entry_point='safe_control_gym.explore.base_explore:GaussianNoiseExplore')

register(id='ornstein',
         entry_point='safe_control_gym.explore.base_explore:OrnsteinUhlenbeckNoiseExplore')






