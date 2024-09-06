""" The implementation of HPO class

Reference:
    * stable baselines3 https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
    * Optuna: https://optuna.org

Reruirement:
    * python -m pip install pymysql

"""
import os
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from optuna.samplers import RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from optuna_dashboard import run_server

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.hyperparameters.hpo_sampler import HYPERPARAMS_SAMPLER
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs


class HPO(object):

    def __init__(self, algo, task, sampler, load_study, output_dir, task_config, hpo_config, algo_config, safety_filter=None, sf_config=None):
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
        # init sampler
        if sampler == 'RandomSampler':
            self.sampler = RandomSampler(seed=self.hpo_config.seed)
        elif sampler == 'TPESampler':
            self.sampler = TPESampler(seed=self.hpo_config.seed)
        else:
            raise ValueError('Unknown sampler.')

        assert len(hpo_config.objective) == len(hpo_config.direction), 'objective and direction must have the same length'

    def objective(self, trial: optuna.Trial) -> float:
        """ The stochastic objective function for a HPO tool to optimize over

        args:
            trial: A single trial object that contains the hyperparameters to be evaluated

        """

        # sample candidate hyperparameters
        if self.algo == 'ilqr' and self.safety_filter == 'linear_mpsc':
            sampled_hyperparams = HYPERPARAMS_SAMPLER['ilqr_sf'](self.hps_config, trial)
        else:
            sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](self.hps_config, trial)

        # log trial number
        self.logger.info('Trial number: {}'.format(trial.number))

        # flag for increasing runs
        increase_runs = True
        first_iteration = True

        # do repetition
        returns, seeds = [], []
        while increase_runs:
            increase_runs = False
            if first_iteration:
                Gs = np.inf
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
                                #TODO if implemented for quadrotor, pitch rate penalty should be small.
                                # raise ValueError('Only cartpole task is supported for gp_mpc.')
                                self.algo_config['q_mpc'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                                self.algo_config['r_mpc'] = [sampled_hyperparams['action_weight'], sampled_hyperparams['action_weight']]
                        elif self.algo == 'ilqr':
                            if self.task == 'cartpole':
                                self.algo_config['q_lqr'] = [sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight'], sampled_hyperparams['state_weight'], sampled_hyperparams['state_dot_weight']]
                                self.algo_config['r_lqr'] = [sampled_hyperparams['action_weight']]
                            elif self.task == 'quadrotor':
                                #TODO if implemented for quadrotor, pitch rate penalty should be small.
                                # raise ValueError('Only cartpole task is supported for ilqr.')
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

            Gss = self._compute_cvar(np.array(returns), self.hpo_config.alpha)

            # if the current objective is better than the best objective, trigger more runs to avoid maximization bias
            if self.hpo_config.warm_trials < len(self.study.trials) and self.hpo_config.dynamical_runs:
                if Gss > self.study.best_value or first_iteration is False:
                    if abs(Gs - Gss) > self.hpo_config.approximation_threshold:
                        increase_runs = True
                        first_iteration = False
                        Gs = Gss
                        self.logger.info('Trigger more runs')
                    else:
                        increase_runs = False

        self.logger.info('Returns: {}'.format(Gss))

        return Gss

    def hyperparameter_optimization(self) -> None:

        if self.load_study:
            self.study = optuna.load_study(study_name=self.study_name, storage='mysql+pymysql://optuna@localhost/{}'.format(self.study_name))
        elif self.hpo_config.use_database is False:
            # single-objective optimization
            if len(self.hpo_config.direction) == 1:
                self.study = optuna.create_study(
                    direction=self.hpo_config.direction[0],
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                )
            # multi-objective optimization
            else:
                self.study = optuna.create_study(
                    directions=self.hpo_config.direction,
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                )
        else:
            # single-objective optimization
            if len(self.hpo_config.direction) == 1:
                self.study = optuna.create_study(
                    direction=self.hpo_config.direction[0],
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                    storage='mysql+pymysql://optuna@localhost/{}'.format(self.study_name),
                    load_if_exists=self.hpo_config.load_if_exists
                )
            # multi-objective optimization
            else:
                self.study = optuna.create_study(
                    directions=self.hpo_config.direction,
                    sampler=self.sampler,
                    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                    study_name=self.study_name,
                    storage='mysql+pymysql://optuna@localhost/{}'.format(self.study_name),
                    load_if_exists=self.hpo_config.load_if_exists
                )

        self.study.optimize(self.objective,
                            catch=(RuntimeError,),
                            callbacks=[MaxTrialsCallback(self.hpo_config.trials-1, states=(TrialState.COMPLETE,))],
                            )

        output_dir = os.path.join(self.output_dir, 'hpo')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # save meta data
        self.study.trials_dataframe().to_csv(output_dir + '/trials.csv')

        # save top-n best hyperparameters
        if len(self.hpo_config.direction) == 1:
            trials = self.study.trials
            if self.hpo_config.direction[0] == 'minimize':
                trials.sort(key=self._value_key)
            else:
                trials.sort(key=self._value_key, reverse=True)
            for i in range(min(self.hpo_config.save_n_best_hps, len(self.study.trials))):
                params = trials[i].params
                with open(f'{output_dir}/hyperparameters_{trials[i].value:.4f}.yaml', 'w')as f:
                    yaml.dump(params, f, default_flow_style=False)
        else:
            best_trials = self.study.best_trials
            for i in range(len(self.study.best_trials)):
                params = best_trials[i].params
                with open(f'{output_dir}/best_hyperparameters_[{best_trials[i].values[0]:.4f},{best_trials[i].values[1]:.4f}].yaml', 'w')as f:
                    yaml.dump(params, f, default_flow_style=False)

        # dashboard
        if self.hpo_config.dashboard and self.hpo_config.use_database:
            run_server('mysql+pymysql://optuna@localhost/{}'.format(self.study_name))

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

        self.logger.info('Total runs: {}'.format(self.total_runs))
        self.logger.close()

        return

    def _value_key(self, trial: FrozenTrial) -> float:
        """ Returns value of trial object for sorting

        """
        if trial.value is None:
            if self.hpo_config.direction[0] == 'minimize':
                return float('inf')
            else:
                return float('-inf')
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
