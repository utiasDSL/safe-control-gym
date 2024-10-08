""" The implementation of HPO class using Optuna

Reference:
    * stable baselines3 https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
    * Optuna: https://optuna.org

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from safe_control_gym.hyperparameters.base_hpo import BaseHPO
from safe_control_gym.hyperparameters.optuna.hpo_optuna_utils import HYPERPARAMS_SAMPLER


class HPO_Optuna(BaseHPO):

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
        Hyperparameter Optimization (HPO) class using package Optuna.

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
        self.setup_problem()

    def setup_problem(self):
        """ Setup hyperparameter optimization, e.g., search space, study, algorithm, etc."""

        # init sampler
        self.sampler = TPESampler(seed=self.hpo_config.seed)

    def objective(self, trial: optuna.Trial) -> float:
        """ The stochastic objective function for a HPO tool to optimize over

        args:
            trial: A single trial object that contains the hyperparameters to be evaluated

        """

        # sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.search_space_key](trial, self.state_dim, self.action_dim)

        # log trial number
        self.logger.info('Trial number: {}'.format(trial.number))

        returns = self.evaluate(sampled_hyperparams)
        Gss = np.array(returns).mean()

        self.logger.info('Returns: {}'.format(Gss))

        if len(self.study.trials) > 0:
            self.checkpoint()

        self.objective_value = Gss
        # wandb.log({self.hpo_config.objective[0]: Gss})
        return Gss

    def hyperparameter_optimization(self) -> None:
        """ Hyperparameter optimization.
        """
        if self.load_study:
            self.study = optuna.load_study(study_name=self.study_name, storage=f'sqlite:///{self.study_name}_optuna.db')
        else:
            # single-objective optimization
            if len(self.hpo_config.direction) == 1:
                self.study = optuna.create_study(
                    direction=self.hpo_config.direction[0],
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                    storage='sqlite:///{}_optuna.db'.format(self.study_name),
                    load_if_exists=self.hpo_config.load_if_exists
                )
            # multi-objective optimization
            else:
                self.study = optuna.create_study(
                    directions=self.hpo_config.direction,
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                    storage='sqlite:///{}_optuna.db'.format(self.study_name),
                    load_if_exists=self.hpo_config.load_if_exists
                )
            self.warm_start(self.config_to_param(self.hps_config))

        self.study.optimize(self.objective,
                            catch=(RuntimeError,),
                            callbacks=[MaxTrialsCallback(self.hpo_config.trials, states=(TrialState.COMPLETE,)),
                                       self._warn_unused_parameter_callback],
                            )

        self.checkpoint()

        self.logger.close()

        return

    def warm_start(self, params):
        """
        Warm start the study.

        Args:
            params (dict): Specified hyperparameters to be evaluated.
        """
        if hasattr(self, 'study'):
            self.study.enqueue_trial(params, skip_if_exists=True)

    def checkpoint(self):
        """
        Save checkpoints, results, and logs during optimization.
        """
        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # save meta data
            self.study.trials_dataframe().to_csv(output_dir + '/trials.csv')
        except Exception as e:
            print(e)
            print('Saving trials.csv failed.')

        try:
            # save top-n best hyperparameters
            if len(self.hpo_config.direction) == 1:
                trials = self.study.get_trials(deepcopy=True, states=(TrialState.COMPLETE,))
                if self.hpo_config.direction[0] == 'minimize':
                    trials.sort(key=self._value_key)
                else:
                    trials.sort(key=self._value_key, reverse=True)
                for i in range(min(self.hpo_config.save_n_best_hps, len(self.study.trials))):
                    params = trials[i].params
                    params = self.post_process_best_hyperparams(params)
                    with open(f'{output_dir}/hyperparameters_trial{len(trials)}_{trials[i].value:.4f}.yaml', 'w')as f:
                        yaml.dump(params, f, default_flow_style=False)
            else:
                best_trials = self.study.best_trials
                for i in range(len(self.study.best_trials)):
                    params = best_trials[i].params
                    params = self.post_process_best_hyperparams(params)
                    with open(f'{output_dir}/best_hyperparameters_[{best_trials[i].values[0]:.4f},{best_trials[i].values[1]:.4f}].yaml', 'w')as f:
                        yaml.dump(params, f, default_flow_style=False)
        except Exception as e:
            print(e)
            print('Saving best hyperparameters failed.')

        # save plot
        try:
            if len(self.hpo_config.objective) == 1:
                plot_param_importances(self.study)
                plt.tight_layout()
                plt.savefig(output_dir + '/param_importances.png')
                # plt.show()
                plt.close()
                plot_optimization_history(self.study)
                plt.tight_layout()
                plt.savefig(output_dir + '/optimization_history.png')
                # plt.show()
                plt.close()
            else:
                for i in range(len(self.hpo_config.objective)):
                    plot_param_importances(self.study, target=lambda t: t.values[i])
                    plt.tight_layout()
                    plt.savefig(output_dir + '/param_importances_{}.png'.format(self.hpo_config.objective[i]))
                    # plt.show()
                    plt.close()
                    plot_optimization_history(self.study, target=lambda t: t.values[i])
                    plt.tight_layout()
                    plt.savefig(output_dir + '/optimization_history_{}.png'.format(self.hpo_config.objective[i]))
                    # plt.show()
                    plt.close()
        except Exception as e:
            print(e)
            print('Plotting failed.')

    def _value_key(self, trial: FrozenTrial) -> float:
        """ Returns value of trial object for sorting

        """
        if trial.value is None:
            if self.hpo_config.direction[0] == 'minimize':
                return self.objective_bounds[0][1]
            else:
                return self.objective_bounds[0][0]
        else:
            return trial.value

    def _compute_cvar(self, returns: np.ndarray, alpha: float = 0.2) -> float:
        """ Compute CVaR

        """
        assert returns.ndim == 1, 'returns must be 1D array'
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        VaR_idx = int(alpha * n)
        if VaR_idx == 0:
            VaR_idx = 1

        if self.hpo_config.direction[0] == 'minimize':
            CVaR = sorted_returns[-VaR_idx:].mean()
        else:
            CVaR = sorted_returns[:VaR_idx].mean()

        return CVaR

    def _warn_unused_parameter_callback(self, study: optuna.Study, trial: FrozenTrial) -> None:
        """User-defined callback to warn unused parameters."""
        fixed_params = trial.system_attrs.get('fixed_params')
        if fixed_params is None:
            return

        for param_name, param_value in fixed_params.items():
            distribution = trial.distributions.get(param_name)
            if distribution is None:
                # Or you can raise a something exception here.
                self.logger.info(f"Parameter '{param_name}' is not used at trial {trial.number}.")
                continue

            param_value_internal_repr = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_value_internal_repr):
                # Or you can raise a something exception here.
                self.logger.info(f"Parameter '{param_name}' is not used at trial {trial.number}.")
