'''Abstract class for MPSC Cost Functions.'''

from abc import ABC, abstractmethod

from safe_control_gym.envs.benchmark_env import BenchmarkEnv


class MPSC_COST(ABC):
    '''Abstract MPSC Cost Function to be extended for different cost functions.'''

    def __init__(self,
                 env: BenchmarkEnv = None,
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env (BenchmarkEnv): Environment for the task.
        '''

        self.env = env

        # Setup attributes.
        self.model = self.env.symbolic if env is not None else None

    @abstractmethod
    def get_cost(self, opti_dict):
        '''Returns the cost function for the MPSC optimization in symbolic form.

        Args:
            opti_dict (dict): The dictionary of optimization variables.

        Returns:
            cost (casadi symbolic expression): The symbolic cost function using casadi.
        '''
        raise NotImplementedError

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepares all the symbolic variable initial values for the next optimization.

        Args:
            opti_dict (dict): The dictionary of optimization variables.
            obs (ndarray): Current state/observation.
            iteration (int): The current iteration, used for trajectory tracking.
        '''
        return
