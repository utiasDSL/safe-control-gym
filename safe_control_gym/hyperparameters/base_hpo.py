'''Base hyerparameter optimization class.'''


import os
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
import numpy as np

from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs

class BaseHPO(ABC):
    
    def __init__(self,
                 hpo_config,
                 task_config,
                 algo_config,
                 algo='ilqr', 
                 task='stabilization',
                 output_dir='./results', 
                 safety_filter=None, 
                 sf_config=None,
                 load_study=False):
        """
        Base class for Hyperparameter Optimization (HPO).
        
        Args:
            hpo_config: Configuration specific to hyperparameter optimization.
            task_config: Configuration for the task.
            algo_config: Algorithm configuration.
            algo (str): Algorithm name.
            task (str): The task/environment the agent will interact with.
            output_dir (str): Directory where results and models will be saved.
            safety_filter (str): Safety filter to be applied (optional).
            sf_config: Safety filter configuration (optional).
            load_study (bool): Load existing study if True.
        """
        self.algo = algo
        self.task = task
        self.output_dir = output_dir
        self.task_config = task_config
        self.hpo_config = hpo_config
        self.algo_config = algo_config
        self.safety_filter = safety_filter
        self.sf_config = sf_config
        self.logger = ExperimentLogger(output_dir, log_file_out=False)
        self.load_study = load_study
        self.study_name = algo + '_hpo'
        self.hps_config = hpo_config.hps_config
        self.n_episodes = hpo_config.n_episodes

        env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
        agent = make(self.algo,
                    env_func,
                    training=True,
                    checkpoint_path=os.path.join(self.output_dir, 'model_latest.pt'),
                    output_dir=os.path.join(self.output_dir, 'hpo'),
                    use_gpu=self.hpo_config.use_gpu,
                    **deepcopy(self.algo_config))
        
        self.state_dim = agent.env.state_dim
        self.action_dim = agent.env.action_dim

        self.check_hyperparmeter_config()

        assert len(hpo_config.objective) == len(hpo_config.direction), 'objective and direction must have the same length'
        assert len(hpo_config.objective) == 1, 'Only single-objective optimization is supported'

    def check_hyperparmeter_config(self):
        """
        Check if the hyperparameter configuration is valid.
        """
        valid = True
        for param in self.hps_config:
            if HYPERPARAMS_DICT[self.algo][param]['type'] != type(self.hps_config[param]):
                valid = False
                assert valid, f'Hyperparameter {param} should be of type {HYPERPARAMS_DICT[self.algo][param]["type"]}'

    @abstractmethod
    def setup_problem(self):
        """
        Setup hyperparameter optimization, e.g., search space, study, algorithm, etc.
        Needs to be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def warm_start(self, params):
        """
        Warm start the study.
        
        Args:
            params (dict): Specified hyperparameters.
            objective (float): Objective value.
        """
        raise NotImplementedError
    
    def cast_to_original_type_from_config(self, param_name, param_value):
        """
        Cast the parameter to its original type based on the existing task or algo config.

        Args:
            param_name (str): Name of the hyperparameter.
            param_value (Any): Sampled value of the hyperparameter.

        Returns:
            Any: The parameter value cast to the appropriate type.
        """
        # Check if the parameter exists in task_config or algo_config
        if param_name in self.task_config:
            current_value = self.task_config[param_name]
        elif param_name in self.algo_config:
            current_value = self.algo_config[param_name]
        else:
            raise ValueError(f"Unknown parameter {param_name} - cannot cast to original type from config")

        return type(current_value)(param_value)
        
    def cast_to_original_type_from_hyperparams_dict(self, param_name, param_value):
        """
        Cast the parameter to its original type based on HYPERPARAMS_DICT.

        Args:
            param_name (str): Name of the hyperparameter.
            param_value (Any): Sampled value of the hyperparameter.

        Returns:
            Any: The parameter value cast to the appropriate type.
        """
        return HYPERPARAMS_DICT[self.algo][param_name]['type'](param_value)
    
    def param_to_config(self, params):
        """
        Convert sampled hyperparameters to configuration (self.task_config, self.algo_config, etc).

        Args:
            params (dict): Sampled hyperparameters.
        
        """
        # Iterate through the params dictionary
        for param_name, param_value in params.items():
            # Handle multidimensional hyperparameters (e.g., q_mpc_0, q_mpc_1)
            if '_' in param_name:
                base_param, index_str = param_name.rsplit('_', 1)
                if index_str.isdigit() and (base_param in self.algo_config or base_param in self.task_config):
                    # If base parameter exists in algo_config as a list/array
                    index = int(index_str)
                    if base_param in self.algo_config:
                        self.algo_config[base_param][index] = param_value
                    elif base_param in self.task_config:
                        self.task_config[base_param][index] = param_value
            # Handle the mapping based on the parameter name
            elif param_name in self.algo_config:
                # If the param is related to the algorithm, update the algo_config
                self.algo_config[param_name] = self.cast_to_original_type_from_config(param_name, param_value)
            elif param_name in self.task_config:
                # If the param is related to the task, update the task_config
                self.task_config[param_name] = self.cast_to_original_type_from_config(param_name, param_value)
            else:
                # If the parameter doesn't map to a known config, log or handle it
                print(f"Warning: Unknown parameter {param_name} - not mapped to any configuration")

    def config_to_param(self, params):
        """
        Convert configuration to hyperparameters (mainly to handle multidimensional hyperparameters) as the input to add_trial.

        Args:
            params (dict): User specified hyperparameters.

        Returns:
            dict: Hyperparameter representation in the problem.
        """
        for param in list(params.keys()):
            is_list = isinstance(params[param], list)
            if is_list:
                for i, value in enumerate(params[param]):
                    new_param = f'{param}_{i}'
                    params[new_param] = value
                del params[param]

        return params

    def post_process_best_hyperparams(self, params):
        """
        Post-process the best hyperparameters after optimization (mainly to handle multidimensional hyperparameters).
        
        Args:
            params (dict): Best hyperparameters.
            
        Returns:
            params (dict): Post-processed hyperparameters.
        """
        aggregated_params = {}
        for param_name, param_value in params.items():
            # Handle multidimensional hyperparameters (e.g., q_mpc_0, q_mpc_1)
            if '_' in param_name:
                base_param, index_str = param_name.rsplit('_', 1)
                if index_str.isdigit() and (base_param in self.algo_config or base_param in self.task_config):
                    # If base parameter exists in algo_config as a list/array
                    index = int(index_str)
                    if base_param in aggregated_params:
                        aggregated_params[base_param] += [param_value]
                    else:
                        aggregated_params[base_param] = [param_value]  

        for param_name, param_value in aggregated_params.items():            
            for hp in list(params.keys()):  # Create a list of keys to iterate over
                if param_name in hp:
                    del params[hp]
            params[param_name] = param_value

        for hp in params:
            params[hp] = self.cast_to_original_type_from_config(hp, params[hp])

        return params

    def evaluate(self, params):
        """
        Evaluation of hyperparameters.
        Needs to be implemented by subclasses.
        
        Args:
            params (dict): Hyperparameters to be evaluated.
            
        Returns:
            Sampled objective value (list)
        """
        sampled_hyperparams = params

        returns, seeds = [], []
        for i in range(self.hpo_config.repetitions):
            
            seed = np.random.randint(0, 10000)

            # update the agent config with sample candidate hyperparameters
            # pass the hyperparameters to config
            self.param_to_config(sampled_hyperparams)

            seeds.append(seed)
            self.logger.info('Sample hyperparameters: {}'.format(sampled_hyperparams))
            self.logger.info('Seeds: {}'.format(seeds))

            try:
                self.env_func = partial(make, self.task, output_dir=self.output_dir, **self.task_config)
                # using deepcopy(self.algo_config) prevent setting being overwritten
                self.agent = make(self.algo,
                                    self.env_func,
                                    training=True,
                                    checkpoint_path=os.path.join(self.output_dir, 'model_latest.pt'),
                                    output_dir=os.path.join(self.output_dir, 'hpo'),
                                    use_gpu=self.hpo_config.use_gpu,
                                    seed=seed,
                                    **deepcopy(self.algo_config))

                self.agent.reset()
                eval_env = self.env_func(seed=seed * 111)
                # Setup safety filter
                if self.safety_filter is not None:
                    env_func_filter = partial(make,
                                            self.task,
                                            **self.task_config)
                    safety_filter = make(self.safety_filter,
                                        env_func_filter,
                                        **self.sf_config)
                    safety_filter.reset()
                    try: 
                        safety_filter.learn()
                    except Exception as e:
                        self.logger.info(f'Exception occurs when constructing safety filter: {e}')
                        self.logger.info('Safety filter config: {}'.format(self.sf_config))
                        self.logger.std_out_logger.logger.exception("Full exception traceback")
                        self.agent.close()
                        del self.agent
                        del self.env_func
                        return None
                    mkdirs(f'{self.output_dir}/models/')
                    safety_filter.save(path=f'{self.output_dir}/models/{self.safety_filter}.pkl')
                    experiment = BaseExperiment(eval_env, self.agent, safety_filter=safety_filter)
                else:
                    experiment = BaseExperiment(eval_env, self.agent)
            except Exception as e:
                # catch exception
                self.logger.info(f'Exception occurs when constructing agent: {e}')
                self.logger.std_out_logger.logger.exception("Full exception traceback")
                if hasattr(self, 'agent'):
                    self.agent.close()
                    del self.agent
                del self.env_func
                return None

            # return objective estimate
            try:
                # self.agent.learn()
                experiment.launch_training()
            except Exception as e:
                # catch the NaN generated by the sampler
                self.agent.close()
                del self.agent
                del self.env_func
                del experiment
                self.logger.info(f'Exception occurs during learning: {e}')
                self.logger.std_out_logger.logger.exception("Full exception traceback")
                print(e)
                print('Sampled hyperparameters:')
                print(sampled_hyperparams)
                return None

            # TODO: add n_episondes to the config
            try:
                _, metrics = experiment.run_evaluation(n_episodes=self.n_episodes, n_steps=None, done_on_max_steps=False)
            except Exception as e:
                self.agent.close()
                # delete instances
                del self.agent
                del self.env_func
                del experiment
                self.logger.info(f'Exception occurs during evaluation: {e}')
                self.logger.std_out_logger.logger.exception("Full exception traceback")
                print(e)
                print('Sampled hyperparameters:')
                print(sampled_hyperparams)
                return None

            # at the moment, only single-objective optimization is supported
            returns.append(metrics[self.hpo_config.objective[0]])
            self.logger.info('Sampled objectives: {}'.format(returns))

            self.agent.close()
            # delete instances
            del self.agent
            del self.env_func

        return returns
    
    @abstractmethod
    def hyperparameter_optimization(self):
        """
        Hyperparameter optimization loop. Should be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def checkpoint(self):
        """
        Save checkpoints, results, and logs during optimization.
        """
        raise NotImplementedError
