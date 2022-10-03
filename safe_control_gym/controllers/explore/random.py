'''A random actor that samples actions uniformly from env.action_space
'''

import os
import numpy as np
import math

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.envs.benchmark_env import Task, Environment


class Random(BaseController):
    ''' Random actor. '''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path="model_latest.pt",
                 output_dir="temp",
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        '''Common control classes __init__ method.
        '''

        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        self.env = env_func(seed=seed)

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''
        action = self.env.action_space.sample()
        return action

    def reset(self):
        '''Do initializations for training or evaluation. '''
        pass
        
    def close(self):
        '''Shuts down and cleans up lingering resources. '''
        self.env.close()

    