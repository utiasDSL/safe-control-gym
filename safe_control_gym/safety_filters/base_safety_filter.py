'''Base class for safety filter. '''

from abc import ABC, abstractmethod


class BaseSafetyFilter(ABC):
    '''Template for safety filter, implement the following methods as needed. '''

    def __init__(self,
                 env_func,
                 output_dir: str='temp',
                 seed: int=0,
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

    @abstractmethod
    def certify_action(self,
                       current_state,
                       uncertified_action,
                       iteration=None,
                       ):
        '''Determines a safe action from the current state and proposed action.

        Args:
            current_state (ndarray): current state/observation.
            uncertified_action (ndarray): the uncertified_controller's action.
            iteration (int): the current iteration, used for trajectory tracking.

        Returns:
            action (ndarray): the certified action.
            success (bool): whether the safety filtering was successful or not.
        '''
        return

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).

        Args:
            env (gym.Env): the environment to be used for training.
        '''
        return

    def reset(self):
        '''Do initializations for training or evaluation. '''
        return

    def close(self):
        '''Shuts down and cleans up lingering resources. '''
        return

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path.

        Args:
            path (str): the path where to save the model params/experiment state.
        '''
        return

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path.

        Args:
            path (str): the path where the model params/experiment state are saved.
        '''
        return
