from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import servers

import os, time
from copy import deepcopy
from functools import partial
import numpy as np
import yaml

from safe_control_gym.hyperparameters.hpo_sampler import HYPERPARAMS_DICT
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs

class HPO(object):

    def __init__(self, algo, task, load_study, output_dir, task_config, hpo_config, algo_config, safety_filter=None, sf_config=None):
        """ Hyperparameter optimization class

        args:
            algo: algo name
            env_func: environment that the agent will interact with
            output_dir: output directory
            hpo_config: hyperparameter optimization configuration
            algo_config: algorithm configuration
            config: other configurations
        """

        self.algo = algo
        self.study_name = algo + '_hpo'
        self.task = task
        self.load_study = load_study
        self.task_config = task_config
        self.hpo_config = hpo_config
        self.hps_config = hpo_config.hps_config
        self.output_dir = output_dir
        self.algo_config = algo_config
        self.safety_filter = safety_filter
        self.sf_config = sf_config
        self.logger = ExperimentLogger(output_dir, log_file_out=False)
        self.total_runs = 0
        self.client_id = f"client_{os.getpid()}" # use process id as client id

        # define the problem statement
        self.problem = vz.ProblemStatement()
        # define the search space
        self.search_space = HYPERPARAMS_DICT[self.algo]
        for cat, hps in self.search_space.items():
            if cat == 'float':
                for hp_name, hp_range in hps.items():
                    self.problem.search_space.root.add_float_param(hp_name, hp_range[0], hp_range[1])
            elif cat == 'categorical':
                for hp_name, hp_choices in hps.items():
                    # check if the choices are strings
                    if isinstance(hp_choices[0], str):
                        self.problem.search_space.root.add_categorical_param(hp_name, hp_choices)
                    else:
                        self.problem.search_space.root.add_discrete_param(hp_name, hp_choices)
            else:
                raise ValueError('Invalid hyperparameter category')
        
        assert len(hpo_config.objective) == 1, 'Only single-objective optimization is supported'

        if self.hpo_config.direction[0] == 'maximize':
            self.problem.metric_information.append(vz.MetricInformation(name=self.hpo_config.objective[0], goal=vz.ObjectiveMetricGoal.MAXIMIZE))
        elif self.hpo_config.direction[0] == 'minimize':
            self.problem.metric_information.append(vz.MetricInformation(name=self.hpo_config.objective[0], goal=vz.ObjectiveMetricGoal.MINIMIZE))

        assert len(hpo_config.objective) == len(hpo_config.direction), 'objective and direction must have the same length'

    def evaluate(self, params: dict) -> float:
        """ Evaluate the suggested hyperparameters

        args:
            params: suggested hyperparameters

        returns:
            objective value
        """

        sampled_hyperparams = params


        returns, seeds = [], []
        for i in range(self.hpo_config.repetitions):
            # np.random.seed()
            seed = np.random.randint(0, 10000)
            # update the agent config with sample candidate hyperparameters
            # new agent with the new hps
            for hp in sampled_hyperparams:
                if hp == 'state_weight' or hp == 'state_dot_weight' or hp == 'action_weight':
                    if self.algo == 'gp_mpc' or self.algo == 'gpmpc_acados':
                        if self.task == 'cartpole':
                            self.algo_config['q_mpc'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.algo_config['r_mpc'] = [sampled_hyperparams['action_weight']]
                        elif self.task == 'quadrotor':
                            self.algo_config['q_mpc'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.algo_config['r_mpc'] = [sampled_hyperparams['action_weight'], sampled_hyperparams['action_weight']]
                    elif self.algo == 'ilqr':
                        if self.task == 'cartpole':
                            self.algo_config['q_lqr'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.algo_config['r_lqr'] = [sampled_hyperparams['action_weight']]
                        elif self.task == 'quadrotor':
                            self.algo_config['q_lqr'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.algo_config['r_lqr'] = [sampled_hyperparams['action_weight'], sampled_hyperparams['action_weight']]
                    else:
                        if self.task == 'cartpole':
                            self.task_config['rew_state_weight'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.task_config['rew_action_weight'] = [sampled_hyperparams['action_weight']]
                        elif self.task == 'quadrotor':
                            self.task_config['rew_state_weight'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                            self.task_config['rew_action_weight'] = [sampled_hyperparams['action_weight'], sampled_hyperparams['action_weight']]
                else:
                    # check key in algo_config
                    if hp in self.algo_config:
                        self.algo_config[hp] = sampled_hyperparams[hp]
                    elif hp in self.sf_config or hp == 'sf_state_weight' or hp == 'sf_state_dot_weight' or hp == 'sf_action_weight':
                        if self.task == 'cartpole':
                            if hp == 'sf_state_weight' or hp == 'sf_state_dot_weight' or hp == 'sf_action_weight':
                                self.sf_config['q_lin'] = [sampled_hyperparams['sf_state_weight'], sampled_hyperparams['sf_state_dot_weight'], sampled_hyperparams['sf_state_weight'], sampled_hyperparams['sf_state_dot_weight']]
                                self.sf_config['r_lin'] = [sampled_hyperparams['sf_action_weight']] 
                            else:
                                self.sf_config[hp] = sampled_hyperparams[hp]
                        else:
                            raise ValueError('Only cartpole task is supported for linear_mpsc.')
                    else:
                        raise ValueError('Unknown hyperparameter: {}'.format(hp))

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
            # TODO: report intermediate results to Optuna for pruning
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

            # avg_return = self.agent._run()
            # TODO: add n_episondes to the config
            try:
                _, metrics = experiment.run_evaluation(n_episodes=5, n_steps=None, done_on_max_steps=True)
                self.total_runs += 1
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

        return np.mean(returns)
    
    def hyperparameter_optimization(self) -> None:
        """ Hyperparameter optimization
        """
        if self.load_study:
            with open(f'{self.study_name}_endpoint.yaml', 'r') as config_file:
                endpoint = yaml.safe_load(config_file)['endpoint']
            clients.environment_variables.server_endpoint = endpoint
            study_config = vz.StudyConfig.from_problem(self.problem)
            study_client = clients.Study.from_study_config(study_config, owner='owner', study_id = self.study_name)
            study_client = clients.Study.from_resource_name(study_client.resource_name)
        else:
            server = servers.DefaultVizierServer(database_url=f'sqlite:///{self.study_name}.db')
            clients.environment_variables.server_endpoint = server.endpoint
            endpoint = server.endpoint
            with open(f'{self.study_name}_endpoint.yaml', 'w') as config_file:
                yaml.dump({'endpoint': endpoint}, config_file, default_flow_style=False)
        
            study_config = vz.StudyConfig.from_problem(self.problem)
            study_client = clients.Study.from_study_config(study_config, owner='owner', study_id = self.study_name)
        
        # study_config = vz.StudyConfig.from_problem(self.problem)
        # study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)

        existing_trials = len(study_client._client.list_trials())
        while existing_trials < self.hpo_config.trials:
            self.logger.info(f'Hyperparameter optimization trial {existing_trials+1}/{self.hpo_config.trials}')
            # get suggested hyperparameters
            suggestions = study_client.suggest(count=1, client_id=self.client_id)
            # suggestions = study_client.suggest(count=1)

            for suggestion in suggestions:
                # evaluate the suggested hyperparameters
                materialized_suggestion = suggestion.materialize()
                suggested_params = {key: val.value for key, val in materialized_suggestion.parameters._items.items()}
                objective_value = self.evaluate(suggested_params)
                self.logger.info(f'Returns: {objective_value}')
                if objective_value is None:
                    objective_value = 0.0
                final_measurement = vz.Measurement({f'{self.hpo_config.objective[0]}': objective_value})
                suggestion.complete(final_measurement)
            
            existing_trials = len(study_client._client.list_trials())

        trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
        finished_trials = len(study_client.trials(trial_filter=trial_filter)._client.list_trials())
        # wait until other clients to finish
        while finished_trials < self.hpo_config.trials:
            self.logger.info(f'Waiting for other clients to finish. {finished_trials}/{self.hpo_config.trials}')
            finished_trials = len(study_client.trials(trial_filter=trial_filter)._client.list_trials())
            # sleep for 10 seconds
            time.sleep(10)
                
        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for optimal_trial in study_client.optimal_trials():
            optimal_trial = optimal_trial.materialize()
            params = {key: val.value for key, val in optimal_trial.parameters._items.items()}
            with open(f'{output_dir}/hyperparameters_{optimal_trial.final_measurement.metrics[self.hpo_config.objective[0]].value:.4f}.yaml', 'w')as f: 
                yaml.dump(params, f, default_flow_style=False)