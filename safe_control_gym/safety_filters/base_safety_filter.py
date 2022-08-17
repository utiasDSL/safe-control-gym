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
            env_func (callable): Function to instantiate task/env.
            output_dir (str): Folder to write outputs.
            seed (int): Random seed.
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
                       info=None,
                       ):
        '''Determines a safe action from the current state and proposed action.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The certified action.
            success (bool): Whether the safety filtering was successful or not.
        '''
        raise NotImplementedError

    def extract_step(self, info=None):
        '''Extracts the current step from the info.

        Args:
            info (dict): The info list returned from the environment.

        Returns:
            step (int): The current step/iteration of the environment.
        '''

        if info is not None:
            step = info['current_step']
        else:
            step = 0

        return step

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).

        Args:
            env (BenchmarkEnv): The environment to be used for training.
        '''
        return

    @abstractmethod
    def reset(self):
        '''Do initializations for training or evaluation. '''
        raise NotImplementedError

    def reset_before_run(self, env=None):
        '''Reinitialize just the safety filter before a new run.

        Args:
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.setup_results_dict()

    @abstractmethod
    def close(self):
        '''Shuts down and cleans up lingering resources. '''
        raise NotImplementedError

    def save(self,
             path
             ):
        '''Saves model params and experiment state to checkpoint path.

        Args:
            path (str): The path where to save the model params/experiment state.
        '''
        return

    def load(self,
             path
             ):
        '''Restores model and experiment given checkpoint path.

        Args:
            path (str): The path where the model params/experiment state are saved.
        '''
        return

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information. '''
        self.results_dict = {}
