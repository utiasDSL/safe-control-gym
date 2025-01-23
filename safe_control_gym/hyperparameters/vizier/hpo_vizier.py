""" The implementation of HPO class using Vizier

Reference:
    * https://oss-vizier.readthedocs.io/en/latest/
    * https://arxiv.org/pdf/0912.3995

"""

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import servers

from safe_control_gym.hyperparameters.base_hpo import BaseHPO
from safe_control_gym.hyperparameters.hpo_search_space import HYPERPARAMS_DICT


class HPO_Vizier(BaseHPO):

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
        Hyperparameter Optimization (HPO) class using package Vizier.

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
        super().__init__(hpo_config, task_config, algo_config, algo, task, output_dir, safety_filter, sf_config, load_study)

        self.client_id = f'client_{os.getpid()}'  # use process id as client id
        self.setup_problem()

    def setup_problem(self):
        """ Setup hyperparameter optimization, e.g., search space, study, algorithm, etc. """

        # define the problem statement
        self.problem = vz.ProblemStatement()

        # define the search space
        self.search_space = HYPERPARAMS_DICT[self.search_space_key]

        for hp_name, hp_info in self.search_space.items():
            hp_values = hp_info['values']
            scale = hp_info['scale']
            is_list = hp_info['type'] == list
            cat = hp_info['cat']

            if cat == 'float':
                if scale == 'uniform':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_float_param(f'{hp_name}_{i}', hp_values[0], hp_values[1])
                    else:
                        self.problem.search_space.root.add_float_param(hp_name, hp_values[0], hp_values[1])
                elif scale == 'log':
                    if is_list:
                        for i in range(len(self.hps_config[hp_name])):
                            self.problem.search_space.root.add_float_param(f'{hp_name}_{i}', hp_values[0], hp_values[1], scale_type=vz.ScaleType.LOG)
                    else:
                        self.problem.search_space.root.add_float_param(hp_name, hp_values[0], hp_values[1], scale_type=vz.ScaleType.LOG)
                else:
                    raise ValueError('Invalid scale')

            elif cat == 'discrete':
                if scale == 'uniform':
                    self.problem.search_space.root.add_discrete_param(hp_name, hp_values)
                elif scale == 'log':
                    self.problem.search_space.root.add_discrete_param(hp_name, hp_values, scale_type=vz.ScaleType.LOG)
                else:
                    raise ValueError('Invalid scale')

            elif cat == 'categorical':
                self.problem.search_space.root.add_categorical_param(hp_name, hp_values)
            else:
                raise ValueError('Invalid hyperparameter category')

        # Set optimization direction based on objective and direction from the HPO config
        if self.hpo_config.direction[0] == 'maximize':
            self.problem.metric_information.append(
                vz.MetricInformation(name=self.hpo_config.objective[0], goal=vz.ObjectiveMetricGoal.MAXIMIZE))
        elif self.hpo_config.direction[0] == 'minimize':
            self.problem.metric_information.append(
                vz.MetricInformation(name=self.hpo_config.objective[0], goal=vz.ObjectiveMetricGoal.MINIMIZE))

    def hyperparameter_optimization(self) -> None:
        """ Hyperparameter optimization.
        """
        if self.load_study:
            with open(f'{self.study_name}_endpoint.yaml', 'r') as config_file:
                endpoint = yaml.safe_load(config_file)['endpoint']
            clients.environment_variables.server_endpoint = endpoint
            study_config = vz.StudyConfig.from_problem(self.problem)
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)
            self.study_client = clients.Study.from_resource_name(self.study_client.resource_name)
        else:
            server = servers.DefaultVizierServer(database_url=f'sqlite:///{self.study_name}_vizier.db')
            clients.environment_variables.server_endpoint = server.endpoint
            endpoint = server.endpoint
            with open(f'{self.study_name}_endpoint.yaml', 'w') as config_file:
                yaml.dump({'endpoint': endpoint}, config_file, default_flow_style=False)

            study_config = vz.StudyConfig.from_problem(self.problem)
            self.study_client = clients.Study.from_study_config(study_config, owner='owner', study_id=self.study_name)
            self.warm_start(self.config_to_param(self.hps_config))

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
                res = self.evaluate(suggested_params)
                objective_value = np.mean(res)
                self.logger.info(f'Returns: {objective_value}')
                if objective_value is None:
                    objective_value = 0.0
                final_measurement = vz.Measurement({f'{self.hpo_config.objective[0]}': objective_value})
                self.objective_value = objective_value
                # wandb.log({f'{self.hpo_config.objective[0]}': objective_value})
                suggestion.complete(final_measurement)

            if existing_trials > 0:
                self.checkpoint()

        if self.load_study is False:

            completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
            finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
            # wait until other clients to finish
            while finished_trials < self.hpo_config.trials:
                self.logger.info(f'Waiting for other clients to finish remaining trials: {self.hpo_config.trials - finished_trials}')
                finished_trials = len(list(self.study_client.trials(trial_filter=completed_trial_filter).get()))
                # sleep for 10 seconds
                time.sleep(10)

            self.logger.info(f'Have finished trials: {finished_trials}/{self.hpo_config.trials}')

            self.checkpoint()

            self.logger.info('Deleting server.')
            del server

        self.logger.close()

    def warm_start(self, params):
        """
        Warm start the study.

        Args:
            params (dict): Specified hyperparameters to be evaluated.
        """
        if hasattr(self, 'study_client'):
            res = self.evaluate(params)
            objective_value = np.mean(res)
            trial = vz.Trial(parameters=params, final_measurement=vz.Measurement({f'{self.hpo_config.objective[0]}': objective_value}))
            self.study_client._add_trial(trial)

    def checkpoint(self):
        """
        Save checkpoints, results, and logs during optimization.
        """
        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        completed_trial_filter = vz.TrialFilter(status=[vz.TrialStatus.COMPLETED])
        all_trials = [tc.materialize() for tc in self.study_client.trials(trial_filter=completed_trial_filter)]

        try:
            for optimal_trial in self.study_client.optimal_trials():
                optimal_trial = optimal_trial.materialize()
                params = {key: val.value for key, val in optimal_trial.parameters._items.items()}
                params = self.post_process_best_hyperparams(params)
                with open(f'{output_dir}/hyperparameters_trial{len(all_trials)}_{optimal_trial.final_measurement.metrics[self.hpo_config.objective[0]].value:.4f}.yaml', 'w')as f:
                    yaml.dump(params, f, default_flow_style=False)
        except Exception as e:
            print(e)
            print('Saving hyperparameters failed')

        try:
            # Visualize all trials so-far.
            trial_i = [t for t in range(len(all_trials))]
            trial_ys = [t.final_measurement.metrics[self.hpo_config.objective[0]].value for t in all_trials]
            plt.scatter(trial_i, trial_ys, label='trials', marker='o', color='blue')

            # Mark optimal trial so far.
            optimal_trial_i = [optimal_trial.id - 1]
            optimal_trial_ys = [optimal_trial.final_measurement.metrics[self.hpo_config.objective[0]].value]
            plt.scatter(optimal_trial_i, optimal_trial_ys, label='optimal', marker='x', color='green', s=100)

            # Plot.
            plt.legend()
            plt.title('Optimization History Plot')
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
                trial_params = self.post_process_best_hyperparams(trial_params)
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
