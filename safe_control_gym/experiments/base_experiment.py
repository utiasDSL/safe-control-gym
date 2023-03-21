'''To standardize training/evaluation interface. '''

from time import time
from copy import deepcopy
from collections import defaultdict
from munch import munchify

import gym
import numpy as np
from termcolor import colored

from safe_control_gym.utils.utils import is_wrapped
from safe_control_gym.math_and_models.metrics.performance_metrics import compute_cvar


class BaseExperiment:
    '''Generic Experiment Class. '''

    def __init__(self,
                 env,
                 ctrl,
                 train_env=None,
                 safety_filter=None,
                 ):
        '''Creates a generic experiment class to run evaluations and collect standard metrics.

        Args:
            env (BenchmarkEnv): The environment for the task.
            ctrl (BaseController): The controller for the task.
            train_env (BenchmarkEnv): The environment used for training.
            safety_filter (BaseSafetyFilter): The safety filter to filter the controller.
        '''

        self.env = env
        if not is_wrapped(self.env, RecordDataWrapper):
            self.env = RecordDataWrapper(self.env)
        self.ctrl = ctrl

        self.train_env = train_env
        if train_env is not None and not is_wrapped(self.train_env, RecordDataWrapper):
            self.train_env = RecordDataWrapper(self.train_env)
        self.safety_filter = safety_filter

        self.reset()

    def run_evaluation(self, training=False, n_episodes=None, n_steps=None, log_freq=None, verbose=True, **kwargs):
        '''Evaluate a trained controller.

        Args:
            training (bool): Whether run_evaluation is being run as part of a training loop or not.
            n_episodes (int): Number of runs to execute.
            n_steps (int): The number of steps to collect in total.
            log_freq (int): The frequency with which to log information.

        Returns:
            trajs_data (dict): The raw data from the executed runs.
            metrics (dict): The metrics calculated from the raw data.
        '''

        if not training:
            self.reset()
        trajs_data = self._execute_evaluations(log_freq=log_freq, n_episodes=n_episodes, n_steps=n_steps, **kwargs)
        metrics = self.compute_metrics(trajs_data)

        # terminal printouts
        if verbose:
            for metric_key, metric_val in metrics.items():
                if isinstance(metric_val, list) or isinstance(metric_val, np.ndarray):
                    rounded = [f'{elem:.3f}' for elem in metric_val]
                    print('{}: {}'.format(colored(metric_key, 'yellow'), rounded))
                else:
                    print('{}: {:.3f}'.format(colored(metric_key, 'yellow'), metric_val))
            print('Evaluation done.')
        return dict(trajs_data), metrics

    def _execute_evaluations(self, n_episodes=None, n_steps=None, log_freq=None, seeds=None):
        trajs_data = self._execute_task(ctrl=self.ctrl,
                                        env=self.env,
                                        n_episodes=n_episodes,
                                        n_steps=n_steps,
                                        log_freq=log_freq,
                                        seeds=seeds)
        return trajs_data

    def _execute_task(self, ctrl=None, env=None, n_episodes=None, n_steps=None, log_freq=None, seeds=None):
        '''Runs the experiments and collects all the required data.

        Args:
            ctrl : Controller to run the task. Default is self.train.
            env : Environment of the task to run. Defulat is self.env.
            n_episodes (int): Number of runs to execute.
            n_steps (int): The number of steps to collect in total.
            log_freq (int): The frequency with which to log information.

        Returns:
            trajs_data (defaultdict(list)): The raw data from the executed runs.
        '''

        if n_episodes is None and n_steps is None:
            raise ValueError('One of n_episodes or n_steps must be defined.')
        elif n_episodes is not None and n_steps is not None:
            raise ValueError('Only one of n_episodes or n_steps can be defined.')
        if ctrl is None:
            ctrl = self.ctrl
        if env is None:
            env = self.env
        if seeds is not None:
            assert len(seeds) == n_episodes, "Number of seeds must match the number of episodes"

        # initialize
        sim_steps = log_freq // env.CTRL_FREQ if log_freq else 1
        steps, trajs = 0, 0
        if seeds is not None:
            seed = seeds[0]
        else:
            seed = None
        obs, info = self._evaluation_reset(ctrl=ctrl,
                                           env=env,
                                           ctrl_data=None,
                                           sf_data=None,
                                           seed=seed)
        ctrl_data = defaultdict(list)
        sf_data = defaultdict(list)

        if n_episodes is not None:
            while trajs < n_episodes:
                action = self._select_action(obs=obs, info=info, ctrl=ctrl, env=env)
                # inner sim loop to accomodate different control frequencies
                for _ in range(sim_steps):
                    obs, _, done, info = env.step(action)
                    if done:
                        trajs += 1
                        if trajs < n_episodes:
                            if seeds is not None:
                                seed = seeds[trajs]
                            else:
                                seed = None
                            obs, info = self._evaluation_reset(ctrl=ctrl,
                                                               env=env,
                                                               ctrl_data=ctrl_data,
                                                               sf_data=sf_data,
                                                               seed=seed)
                        break
        elif n_steps is not None:
            while steps < n_steps:
                action = self._select_action(obs=obs, info=info, ctrl=ctrl, env=env)
                # inner sim loop to accomodate different control frequencies
                for _ in range(sim_steps):
                    obs, _, done, info = env.step(action)
                    steps += 1
                    if steps >= n_steps:
                        env.save_data()
                        for data_key, data_val in ctrl.results_dict.items():
                            ctrl_data[data_key].append(np.array(deepcopy(data_val)))
                        if self.safety_filter is not None:
                            for data_key, data_val in self.safety_filter.results_dict.items():
                                sf_data[data_key].append(np.array(deepcopy(data_val)))
                        break
                    if done:
                        obs, info = self._evaluation_reset(ctrl=ctrl,
                                                           env=env,
                                                           ctrl_data=ctrl_data,
                                                           sf_data=sf_data)
                        break

        trajs_data = env.data
        trajs_data['controller_data'].append(munchify(dict(ctrl_data)))
        if self.safety_filter is not None:
            trajs_data['safety_filter_data'].append(munchify(dict(sf_data)))
        return munchify(trajs_data)

    def _select_action(self, obs, info, ctrl=None, env=None):
        '''Determines the executed action using the controller and safety filter.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.
            ctrl (BaseController): Controller to select the action.
            env (BenchmarkEnv): Environment to normalize actions.

        Returns:
            action (ndarray): The action chosen by the controller and safety filter.
        '''
        if env is None:
            env = self.env
        if ctrl is None:
            ctrl = self.ctrl

        action = ctrl.select_action(obs, info)

        if self.safety_filter is not None:
            physical_action = env.denormalize_action(action)
            unextended_obs = obs[:env.symbolic.nx]
            certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
            if success:
                action = env.normalize_action(certified_action)

        return action

    def _evaluation_reset(self, ctrl, env, ctrl_data, sf_data, seed=None):
        '''Resets the evaluation between runs.

        Args:
            ctrl_data (defaultdict): The controller specific data collected during execution.
            sf_data (defaultdict): The safety filter specific data collected during execution.

        Returns:
            obs (ndarray): The initial observation.
            info (dict): The initial info.
        '''
        if ctrl is None:
            ctrl = self.ctrl
        if env is None:
            env = self.env
        if env.INFO_IN_RESET:
            obs, info = env.reset(seed=seed)
        else:
            obs = env.reset(seed=seed)
            info = None
        if ctrl_data is not None:
            for data_key, data_val in ctrl.results_dict.items():
                ctrl_data[data_key].append(np.array(deepcopy(data_val)))
        if sf_data is not None and self.safety_filter is not None:
            for data_key, data_val in self.safety_filter.results_dict.items():
                sf_data[data_key].append(np.array(deepcopy(data_val)))
        ctrl.reset_before_run(obs, info, env=env)
        if self.safety_filter is not None:
            self.safety_filter.reset_before_run(env=env)
        return obs, info

    def launch_training(self, **kwargs):
        '''Since the learning loop varies among controllers, can only delegate to its own `learn()` method.

        Returns:
            trajs_data (defaultdict(list)): The raw data from the training.
        '''

        self.reset()
        self.ctrl.learn(env=self.train_env, **kwargs)

        if self.safety_filter:
            self.safety_filter.learn(env=self.train_env, **kwargs)

        print('Training done.')

        trajs_data = {}
        if self.train_env is not None:
            trajs_data = self.train_env.data
        return dict(trajs_data)

    @staticmethod
    def compute_metrics(trajs_data):
        '''Compute all standard metrics on the given trajectory data.

        Args:
            trajs_data (defaultdict(list)): The raw data from the executed runs.

        Returns:
            metrics (dict): The metrics calculated from the raw data.
        '''

        met = MetricExtractor(trajs_data)
        # collect & compute all sorts of metrics here
        metrics = {
            'average_length': np.asarray(met.get_episode_lengths()).mean(),
            'length': met.get_episode_lengths() if len(met.get_episode_lengths()) > 1 else met.get_episode_lengths()[0],
            'average_return': np.asarray(met.get_episode_returns()).mean(),
            'average_rmse': np.asarray(met.get_episode_rmse()).mean(),
            'rmse': np.asarray(met.get_episode_rmse())  if len(met.get_episode_rmse()) > 1 else met.get_episode_rmse()[0],
            'rmse_std': np.asarray(met.get_episode_rmse()).std(),
            'worst_case_rmse_at_0.5': compute_cvar(np.asarray(met.get_episode_rmse()), 0.5, lower_range=False),
            'failure_rate':  np.asarray(met.get_episode_constraint_violations()).mean(),
            'average_constraint_violation': np.asarray(met.get_episode_constraint_violation_steps()).mean(),
            'constraint_violation_std': np.asarray(met.get_episode_constraint_violation_steps()).std(),
            'constraint_violation': np.asarray(met.get_episode_constraint_violation_steps()) if len(met.get_episode_constraint_violation_steps()) > 1 else met.get_episode_constraint_violation_steps()[0],
            # others ???
        }
        return metrics

    def reset(self):
        '''Resets the environments, controller, and safety filter to prepare for training or evaluation. '''

        self.env.reset()
        self.env.clear_data()
        self.ctrl.reset()

        if self.safety_filter is not None:
            self.safety_filter.reset()

        if self.train_env is not None:
            self.train_env.reset()
            self.train_env.clear_data()

    def close(self):
        '''Closes the environments, controller, and safety filter. '''

        self.env.close()
        self.ctrl.close()

        if self.safety_filter is not None:
            self.safety_filter.close()

        if self.train_env is not None:
            self.train_env.close()

    def load(self, ctrl_path=None, safety_filter_path=None):
        '''Restores model of the controller and/or safety filter given checkpoint paths.

        Args:
            ctrl_path (str): The path used to load the controller's model.
            safety_filter_path (str): The path used to load the safety_filter's model.
        '''

        if ctrl_path is not None:
            self.ctrl.load(ctrl_path)
        if safety_filter_path is not None:
            self.safety_filter.load(safety_filter_path)

    def save(self, ctrl_path=None, safety_filter_path=None):
        '''Saves the model of the controller and/or safety filter given checkpoint paths.

        Args:
            ctrl_path (str): The path used to save the controller's model.
            safety_filter_path (str): The path used to save the safety_filter's model.
        '''

        if ctrl_path is not None:
            self.ctrl.save(ctrl_path)
        if safety_filter_path is not None:
            self.safety_filter.save(safety_filter_path)


class RecordDataWrapper(gym.Wrapper):
    '''A wrapper to standardizes logging for benchmark envs.

    currently saved info
    * obs, reward, done, info, action
    * env.state, env.current_physical_action,
    env.current_noisy_physical_action, env.current_clipped_action
    '''

    def __init__(self, env):
        super().__init__(env)
        self.episode_data = defaultdict(list)
        self.clear_data()

    def save_data(self):
        '''Saves the current self.episode_data to self.data and clears self.episode_data. '''
        if self.episode_data:
            # save to data container
            for key, ep_val in self.episode_data.items():
                if key == 'info':
                    self.data[key].append(np.array(deepcopy(ep_val), dtype=object))
                else:
                    self.data[key].append(np.array(deepcopy(ep_val)))
            # re-initialize episode data container
            self.episode_data = defaultdict(list)

    def clear_data(self):
        '''Clears all data in self.data and self.episode_data. '''
        self.data = defaultdict(list)
        self.episode_data = defaultdict(list)

    def reset(self, **kwargs):
        '''Wrapper for the gym.env reset function. '''

        if self.env.INFO_IN_RESET:
            obs, info = self.env.reset(**kwargs)
            if 'symbolic_model' in info:
                info.pop('symbolic_model')
            if 'symbolic_constraints' in info:
                info.pop('symbolic_constraints')
            step_data = dict(
                obs=obs, info=info, state=self.env.state
            )
            for key, val in step_data.items():
                self.episode_data[key].append(val)
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            step_data = dict(
                obs=obs, state=self.env.state
            )
            for key, val in step_data.items():
                self.episode_data[key].append(val)
            return obs

    def step(self, action):
        '''Wrapper for the gym.env step function. '''

        obs, reward, done, info = self.env.step(action)
        # save to episode data container
        step_data = dict(
            obs=obs,
            action=self.env.current_raw_action,
            done=float(done),
            info=info,
            reward=reward,
            length=1,
            state=self.env.state,
            current_physical_action=self.env.current_physical_action,
            current_noisy_physical_action=self.env.current_noisy_physical_action,
            current_clipped_action=self.env.current_clipped_action,
            timestamp=time(),
        )
        for key, val in step_data.items():
            self.episode_data[key].append(val)

        if done:
            self.save_data()

        return obs, reward, done, info


class MetricExtractor:
    '''A utility class that computes metrics given collected trajectory data.

    metrics that can be derived
    * episode lengths, episode total rewards/returns
    * RMSE (given the square error/mse is saved in info dict at each step)
    * episode occurrences of constraint violation
        (0/1 for each episode, failure rate = #occurrences/#episodes)
    * episode constraint violation steps
        (how many constraint violations happened in each episode)
    '''

    def __init__(self, data):
        '''Creates a class to extract metrics from standard trajectory data.

        Args:
            data (defaultdict(list)): The raw data from the executed runs, in standard form from the Experiment class.
        '''
        self.data = data

    def get_episode_data(self, key, postprocess_func=lambda x: x):
        '''Extract data field from recorded trajectory data, optionally postprocess each episode data (e.g. get sum).

        Args:
            key (str): The key of the data to retrieve.
            postprocess_func (lambda): A function to process the outgoing data.

        Returns:
            episode_data (list): The desired data.
        '''

        if key in self.data:
            episode_data = [postprocess_func(ep_val) for ep_val in self.data[key]]
        elif key in self.data['info'][0][-1]:
            # if the data field is contained in step info dict
            episode_data = [postprocess_func([info.get(key, 0.) for info in ep_info])
                            for ep_info in self.data['info']]
        else:
            raise KeyError(f'Given data key \'{key}\' does not exist in recorded trajectory data.')
        return episode_data

    def get_episode_lengths(self):
        '''Total length of episodes.

        Returns:
            episode_lengths (list): The lengths of each episode.
        '''
        return self.get_episode_data('length', postprocess_func=sum)

    def get_episode_returns(self):
        '''Total reward/return of episodes.

        Returns:
            episode_rewards (list): The total reward of each episode.
        '''
        return self.get_episode_data('reward', postprocess_func=sum)

    def get_episode_rmse(self):
        '''Root mean square error of episodes.

        Returns:
            episode_rmse (list): The total rmse of each episode.
        '''
        return self.get_episode_data('mse',
                                     postprocess_func=lambda x: float(np.sqrt(np.mean(x))))

    def get_episode_constraint_violations(self):
        '''Occurence of any violation in episodes.

        Returns:
            episode_violated (list): Whether each episode had a constraint violation.
        '''
        return self.get_episode_data('constraint_violation',
                                     postprocess_func=lambda x: float(any(x)))

    def get_episode_constraint_violation_steps(self):
        '''Total violation steps of episodes.

        Returns:
            episode_violations (list): The total number of constraint violations of each episode.
        '''
        return self.get_episode_data('constraint_violation',
                                     postprocess_func=sum)
