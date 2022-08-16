'''Base class for safety filter. '''

from abc import ABC, abstractmethod


class BaseSafetyFilter(ABC):
    '''Template for safety filter, implement the following methods as needed. '''

    def __init__(self,
                 env_func,
                 output_dir='temp',
                 seed=0,
                 **kwargs
                 ):
        '''Initializes controller agent.

        Args:
            env_func (callable): function to instantiate task/env.
            output_dir (str): folder to write outputs.
            seed (int): random seed.
        '''

        # Base args.
        self.env_func = env_func
        self.output_dir = output_dir
        self.seed = seed
        # Algorithm specific args.
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def reset(self):
        '''Do initializations for training or evaluation. '''
        return

    def close(self):
        '''Shuts down and cleans up lingering resources. '''
        return

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path. '''
        return

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path. '''
        return

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc). '''
        return

    @abstractmethod
    def certify_action(self,
                       current_state,
                       uncertified_action,
                       **kwargs
                       ):
        '''Determines a safe action from the current state and proposed action. '''
        return
