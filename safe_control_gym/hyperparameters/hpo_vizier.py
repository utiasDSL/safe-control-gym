from vizier import service
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import servers

import os, time
from copy import deepcopy
from functools import partial
import numpy as np
import yaml, csv
import matplotlib.pyplot as plt
import wandb

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
        self.state_dim = len(hpo_config.hps_config.state_weight)
        self.action_dim = len(hpo_config.hps_config.action_weight)
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
                    # if it is learning rate, use log scale
                    if 'lr' in hp_name or 'learning_rate' in hp_name or 'entropy_coef' in hp_name:
                        self.problem.search_space.root.add_float_param(hp_name, hp_range[0], hp_range[1], scale_type=vz.ScaleType.LOG)
                    elif 'state_weight' == hp_name:
                        for i in range(self.state_dim):
                            self.problem.search_space.root.add_float_param(hp_name + f'_{i}', hp_range[0], hp_range[1])
                    elif 'action_weight' == hp_name:
                        for i in range(self.action_dim):
                            self.problem.search_space.root.add_float_param(hp_name + f'_{i}', hp_range[0], hp_range[1])
                    else:
                        self.problem.search_space.root.add_float_param(hp_name, hp_range[0], hp_range[1])
            elif cat == 'categorical':
                for hp_name, hp_choices in hps.items():
                    # check if the choices are strings
                    if isinstance(hp_choices[0], str):
                        self.problem.search_space.root.add_categorical_param(hp_name, hp_choices)
                    else:
                        self.problem.search_space.root.add_discrete_param(hp_name, hp_choices, auto_cast=True)
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

        state_weight = [sampled_hyperparams[weight] for weight in sampled_hyperparams.keys() if 'state_weight' in weight]
        action_weight = [sampled_hyperparams[weight] for weight in sampled_hyperparams.keys() if 'action_weight' in weight]
        sf_state_weight = [sampled_hyperparams[weight] for weight in sampled_hyperparams.keys() if 'sf_state_weight' in weight]
        sf_action_weight = [sampled_hyperparams[weight] for weight in sampled_hyperparams.keys() if 'sf_action_weight' in weight]
        for hp in list(sampled_hyperparams.keys()):  # Create a list of keys to iterate over
            if 'state_weight' in hp or 'action_weight' in hp or 'sf_state_weight' in hp or 'sf_action_weight' in hp:
                del sampled_hyperparams[hp]
        sampled_hyperparams['state_weight'] = state_weight
        sampled_hyperparams['action_weight'] = action_weight
        if len(sf_state_weight) > 0:
            sampled_hyperparams['sf_state_weight'] = sf_state_weight
        if len(sf_action_weight) > 0:
            sampled_hyperparams['sf_action_weight'] = sf_action_weight

        returns, seeds = [], []
        for i in range(self.hpo_config.repetitions):
            # np.random.seed()
            seed = np.random.randint(0, 10000)
            # update the agent config with sample candidate hyperparameters
            # new agent with the new hps
            for hp in sampled_hyperparams:
                if hp == 'state_weight' or hp == 'action_weight':
                    if self.algo == 'gp_mpc' or self.algo == 'gpmpc_acados':
                        self.algo_config['q_mpc'] = state_weight
                        self.algo_config['r_mpc'] = action_weight
                    elif self.algo == 'ilqr':
                        self.algo_config['q_lqr'] = state_weight
                        self.algo_config['r_lqr'] = action_weight
                    else:
                        self.task_config['rew_state_weight'] = state_weight
                        self.task_config['rew_action_weight'] = action_weight
                else:
                    # check key in algo_config
                    if hp in self.algo_config:
                        # cast to int if the value is an integer
                        if isinstance(self.algo_config[hp], int):
                            self.algo_config[hp] = int(sampled_hyperparams[hp])
                        elif isinstance(self.algo_config[hp], list):
                            if isinstance(self.algo_config[hp][0], int):
                                if isinstance(sampled_hyperparams[hp], list) == False:
                                    self.algo_config[hp] = [int(sampled_hyperparams[hp])] * len(self.algo_config[hp])
                        else:
                            self.algo_config[hp] = sampled_hyperparams[hp]
                    elif hp in self.sf_config or hp == 'sf_state_weight' or hp == 'sf_action_weight':
                        if self.task == 'cartpole':
                            if hp == 'sf_state_weight' or hp == 'sf_action_weight':
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
                _, metrics = experiment.run_evaluation(n_episodes=5, n_steps=None, done_on_max_steps=False)
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
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id = self.study_name)
            self.study_client = clients.Study.from_resource_name(self.study_client.resource_name)
        else:
            server = servers.DefaultVizierServer(database_url=f'sqlite:///{self.study_name}.db')
            clients.environment_variables.server_endpoint = server.endpoint
            endpoint = server.endpoint
            with open(f'{self.study_name}_endpoint.yaml', 'w') as config_file:
                yaml.dump({'endpoint': endpoint}, config_file, default_flow_style=False)
        
            study_config = vz.StudyConfig.from_problem(self.problem)
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id = self.study_name)
        
        # study_config = vz.StudyConfig.from_problem(self.problem)
        # self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)

        existing_trials = 0
        while existing_trials < self.hpo_config.trials:
            # get suggested hyperparameters
            suggestions = self.study_client.suggest(count=1, client_id=self.client_id)
            # suggestions = self.study_client.suggest(count=1)

            for suggestion in suggestions:
                if suggestion.id > self.hpo_config.trials:
                    self.logger.info(f'Trial {suggestion.id} is deleted as it exceeds the maximum number of trials.')
                    suggestion.delete()
                    existing_trials = suggestion.id
                    break
                self.logger.info(f'Hyperparameter optimization trial {suggestion.id}/{self.hpo_config.trials}')
                existing_trials = suggestion.id
                # evaluate the suggested hyperparameters
                materialized_suggestion = suggestion.materialize()
                suggested_params = {key: val.value for key, val in materialized_suggestion.parameters._items.items()}
                objective_value = self.evaluate(suggested_params)
                self.logger.info(f'Returns: {objective_value}')
                if objective_value is None:
                    objective_value = 0.0
                final_measurement = vz.Measurement({f'{self.hpo_config.objective[0]}': objective_value})
                self.objective_value = objective_value
                wandb.log({f'{self.hpo_config.objective[0]}': objective_value})
                suggestion.complete(final_measurement)
            
            if existing_trials > 0:
                self.checkpoint()
            
        # self.study_client.set_state(vz.StudyState.COMPLETED)
        
        if self.load_study == False:

            completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
            finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
            # wait until other clients to finish
            while finished_trials != self.hpo_config.trials:
                self.logger.info(f'Waiting for other clients to finish remaining trials: {self.hpo_config.trials - finished_trials}')
                finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
                # sleep for 10 seconds
                time.sleep(10)

            self.logger.info(f'Have finished trials: {finished_trials}/{self.hpo_config.trials}')

            self.checkpoint()

            self.logger.info('Deleting server.')
            del server

    def checkpoint(self):
        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            for optimal_trial in self.study_client.optimal_trials():
                optimal_trial = optimal_trial.materialize()
                params = {key: val.value for key, val in optimal_trial.parameters._items.items()}
                state_weight = [params[weight] for weight in params.keys() if 'state_weight' in weight]
                action_weight = [params[weight] for weight in params.keys() if 'action_weight' in weight]
                for hp in list(params.keys()):  # Create a list of keys to iterate over
                    if 'state_weight' in hp or 'action_weight' in hp:
                        del params[hp]
                params['state_weight'] = state_weight
                params['action_weight'] = action_weight
                for hp in params:
                    if hp in self.algo_config:
                        # cast to int if the value is an integer
                        if isinstance(self.algo_config[hp], int):
                            params[hp] = int(params[hp])
                        if isinstance(self.algo_config[hp], list):
                            if isinstance(self.algo_config[hp][0], int):
                                if isinstance(params[hp], list) == False:
                                    params[hp] = [int(params[hp])] * len(self.algo_config[hp])
                with open(f'{output_dir}/hyperparameters_{optimal_trial.final_measurement.metrics[self.hpo_config.objective[0]].value:.4f}.yaml', 'w')as f: 
                    yaml.dump(params, f, default_flow_style=False)
        except Exception as e:
            print(e)
            print('Saving hyperparameters failed')
        
        try:
            completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
            all_trials = [tc.materialize() for tc in self.study_client.trials(trial_filter=completed_trial_filter)]

            # Visualize all trials so-far.
            trial_i = [t for t in range(len(all_trials))]
            trial_ys = [t.final_measurement.metrics[self.hpo_config.objective[0]].value for t in all_trials]
            plt.scatter(trial_i, trial_ys, label='trials', marker='o', color='blue')

            # Mark optimal trial so far.
            optimal_trial_i = [optimal_trial.id-1]
            optimal_trial_ys = [optimal_trial.final_measurement.metrics[self.hpo_config.objective[0]].value]
            plt.scatter(optimal_trial_i, optimal_trial_ys, label='optimal', marker='x', color='green', s = 100)

            # Plot.
            plt.legend()
            plt.title(f'Optimization History Plot')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.tight_layout()
            plt.savefig(output_dir + '/optimization_history.png')
            plt.close()

            trial_data = []

            # Collect all the parameter keys across trials for use in the CSV header
            parameter_keys = set()

            for t in all_trials:
                trial_number = t.id - 1 
                trial_value = t.final_measurement.metrics[self.hpo_config.objective[0]].value
                
                # Extract parameters for each trial
                trial_params = {key: val.value for key, val in t.parameters._items.items()}
                for hp in trial_params:
                    if hp in self.algo_config:
                        # cast to int if the value is an integer
                        if isinstance(self.algo_config[hp], int):
                            trial_params[hp] = int(trial_params[hp])
                parameter_keys.update(trial_params.keys())  # Collect parameter keys dynamically
                trial_data.append((trial_number, trial_value, trial_params))

            # Convert set to list for a consistent order in the CSV header
            parameter_keys = sorted(list(parameter_keys))

            # Save to CSV file
            csv_file = 'trials.csv'
            with open(output_dir + '/' + csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write the header with 'number', 'value', followed by all parameter keys
                writer.writerow(['number', 'value'] + parameter_keys)
                
                # Write the trial data
                for trial_number, trial_value, trial_params in trial_data:
                    # Ensure that parameters are written in the same order as the header
                    param_values = [trial_params.get(key, '') for key in parameter_keys]
                    writer.writerow([trial_number, trial_value] + param_values)
        except Exception as e:
            print(e)
            print('Saving optimization history failed')