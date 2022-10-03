'''Subclass Experiment with exploration. '''

from time import time
from copy import deepcopy
from collections import defaultdict

import gym
import numpy as np
from termcolor import colored

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.utils import is_wrapped
from safe_control_gym.math_and_models.metrics.performance_metrics import compute_cvar


class ExploreExperiment(Experiment):
    '''Experiment Class with exploration strategies.'''

    def __init__(self,
                 env,
                 ctrl,
                 train_env=None,
                 safety_filter=None,
                 explore_policy=None,
                 ):
        '''Creates a generic experiment class to run evaluations and collect standard metrics.
        '''
        self.explore_policy = explore_policy
        super().__init__(env, ctrl, train_env=train_env, safety_filter=safety_filter)

    def reset(self):
        '''Resets the environments, controller, and safety filter to prepare for training or evaluation. 
        '''
        super().reset()
        if self.explore_policy:
            self.explore_policy.reset()
    
    def _evaluation_reset(self, ctrl_data, sf_data):
        '''Resets the evaluation between runs.
        '''
        obs, info = super()._evaluation_reset(ctrl_data, sf_data)
        if self.explore_policy:
            self.explore_policy.reset_before_run()
        return obs, info
    
    def _select_action(self, obs, info):
        '''Determines the executed action using the controller and safety filter.
        '''
        action = self.ctrl.select_action(obs, info)
        if self.explore_policy:
            action = self.explore_policy.select_action(obs, info=info, action=action)

        if self.safety_filter is not None:
            physical_action = self.env.denormalize_action(action)
            unextended_obs = obs[:self.env.symbolic.nx]
            certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
            if success:
                action = self.env.normalize_action(certified_action)
        return action