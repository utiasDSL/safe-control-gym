'''Base controller.'''

from abc import ABC, abstractmethod

import torch


class BaseController(ABC):
    '''Template for controller/agent, implement the following methods as needed.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='temp/model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs
                 ):
        '''Initializes controller agent.

        Args:
            env_func (callable): Function to instantiate task/env.
            training (bool): Training flag.
            checkpoint_path (str): File to save trained model & experiment state.
            output_dir (str): Folder to write outputs.
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): Random seed.
        '''

        # Base args.
        self.env_func = env_func
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cpu' if self.use_gpu is False else 'cuda'
        self.seed = seed
        self.prior_info = {}

        # Algorithm specific args.
        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.setup_results_dict()

    @abstractmethod
    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
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
        '''Do initializations for training or evaluation.'''
        raise NotImplementedError

    def reset_before_run(self, obs=None, info=None, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            obs (ndarray): The initial observation for the new run.
            info (dict): The first info of the new run.
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.setup_results_dict()

    @abstractmethod
    def close(self):
        '''Shuts down and cleans up lingering resources.'''
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
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {}

    def get_prior(self, env, prior_info={}):
        '''Fetch the prior model from the env for the controller.

        Note there's a default env.symbolic when each each env is created.
        To make a different prior model, do the following when initializing a ctrl::

            self.env = env_func()
            self.model = self.get_prior(self.env)

        Besides the env config `base.yaml` and ctrl config `mpc.yaml`,
        you can define an additional prior config `prior.yaml` that looks like::

            algo_config:
                prior_info:
                    prior_prop:
                        M: 0.03
                        Iyy: 0.00003
                    randomize_prior_prop: False
                    prior_prop_rand_info: {}

        and to ensure the resulting config.algo_config contains both the params
        from ctrl config and prior config, chain them to the --overrides like::

            python safe_control_gym/experiments/execute_rl_controller.py --algo mpc --task quadrotor --overrides base.yaml mpc.yaml prior.yaml ...

        Also note we look for prior_info from the incoming function arg first, then the ctrl itself.
        this allows changing the prior model during learning by calling::

            new_model = self.get_prior(same_env, new_prior_info)

        Alternatively, you can overwrite this method and use your own format for prior_info
        to customize how you get/change the prior model for your controller.

        Args:
            env (BenchmarkEnv): the environment to fetch prior model from.
            prior_info (dict): specifies the prior properties or other things to
                overwrite the default prior model in the env.

        Returns:
            SymbolicModel: CasAdi prior model.
        '''
        if not prior_info:
            prior_info = getattr(self, 'prior_info', {})
        prior_prop = prior_info.get('prior_prop', {})

        # randomize prior prop, similar to randomizing the inertial_prop in BenchmarkEnv
        # this can simulate the estimation errors in the prior model
        randomize_prior_prop = prior_info.get('randomize_prior_prop', False)
        prior_prop_rand_info = prior_info.get('prior_prop_rand_info', {})
        if randomize_prior_prop and prior_prop_rand_info:
            # check keys, this is due to the current implementation of BenchmarkEnv._randomize_values_by_info()
            for k in prior_prop_rand_info:
                assert k in prior_prop, 'A prior param to randomize does not have a base value in prior_prop.'
            prior_prop = env._randomize_values_by_info(prior_prop, prior_prop_rand_info)

        # Note we only reset the symbolic model when prior_prop is nonempty
        if prior_prop:
            env._setup_symbolic(prior_prop=prior_prop)

        # Note this ensures the env can still access the prior model,
        # which is used to get quadratic costs in env.step()
        prior_model = env.symbolic
        return prior_model
