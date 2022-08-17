'''To standardize training/evaluation interface. '''

from time import time
from copy import deepcopy
from collections import defaultdict

import gym
import numpy as np
from termcolor import colored

from safe_control_gym.utils.utils import is_wrapped
from safe_control_gym.math_and_models.metrics.performance_metrics import compute_cvar


class Experiment:
    '''Generic Experiment Class. '''

    def __init__(self,
                 env,
                 ctrl,
                 train_env=None,
                 safety_filter=None,
                 ):
        '''Creates a generic experiment class to run evaluations and collect standard metrics.

        Args:
            env (gym.Env): the environment for the task.
            ctrl (BaseController): the controller for the task.
            train_env (gym.Env): the environment used for training.
            safety_filter (BaseSafetyFilter): the safety filter to filter the controller.
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

    def run_evaluation(self, training=False, n_episodes=None, n_steps=None, log_freq=None, **kwargs):
        '''Evaluate a trained controller.

        Args:
            training (bool): whether run_evaluation is being run as part of a training loop or not.
            n_episodes (int): number of runs to execute.
            n_steps (int): the number of steps to collect in total.
            log_freq (int): the frequency with which to log information.

        Returns:
            trajs_data (dict): the raw data from the executed runs.
            metrics (dict): the metrics calculated from the raw data.
        '''

        if not training:
            self.reset()
        trajs_data = self._execute_evaluations(log_freq=log_freq, n_episodes=n_episodes, n_steps=n_steps, **kwargs)
        metrics = self.compute_metrics(trajs_data)

        # terminal printouts
        for metric_key, metric_val in metrics.items():
            print('{}: {:.3f}'.format(colored(metric_key, 'yellow'), metric_val))
        print('Evaluation done.')
        return dict(trajs_data), metrics

    def _execute_evaluations(self, n_episodes=None, n_steps=None, log_freq=None):
        '''Runs the experiments and collects all the required data.

        Args:
            n_episodes (int): number of runs to execute.
            n_steps (int): the number of steps to collect in total.
            log_freq (int): the frequency with which to log information.

        Returns:
            trajs_data (defaultdict(list)): the raw data from the executed runs.
        '''

        if n_episodes is None and n_steps is None:
            raise ValueError('One of n_episodes or n_steps must be defined.')
        elif n_episodes is not None and n_steps is not None:
            raise ValueError('Only one of n_episodes or n_steps can be defined.')

        # initialize
        sim_steps = log_freq // self.env.CTRL_FREQ if log_freq else 1
        steps, trajs = 0, 0
        obs, info = self._evaluation_reset(ctrl_data=None)
        ctrl_data = defaultdict(list)

        if n_episodes is not None:
            while trajs < n_episodes:
                action = self.ctrl.select_action(obs, info)
                # inner sim loop to accomodate different control frequencies
                for _ in range(sim_steps):
                    obs, _, done, info = self.env.step(action)
                    if done:
                        trajs += 1
                        obs, info = self._evaluation_reset(ctrl_data=ctrl_data)
                        break
        elif n_steps is not None:
            while steps < n_steps:
                action = self.ctrl.select_action(obs, info)
                # inner sim loop to accomodate different control frequencies
                for _ in range(sim_steps):
                    obs, _, done, info = self.env.step(action)
                    steps += 1
                    if steps >= n_steps:
                        self.env.save_data()
                        for data_key, data_val in self.ctrl.results_dict.items():
                            ctrl_data[data_key].append(data_val)
                        break
                    if done:
                        obs, info = self._evaluation_reset(ctrl_data=ctrl_data)
                        break

        trajs_data = self.env.data
        trajs_data['controller_data'].append(dict(ctrl_data))
        return trajs_data

    def _evaluation_reset(self, ctrl_data):
        '''Resets the evaluation between runs.

        Args:
            ctrl_data (defaultdict): the controller specific data collected during execution.

        Returns:
            obs (ndarray): the initial observation.
            info (dict): the initial info.
        '''
        if self.env.INFO_IN_RESET:
            obs, info = self.env.reset()
        else:
            obs = self.env.reset()
            info = None
        if ctrl_data is not None:
            for data_key, data_val in self.ctrl.results_dict.items():
                ctrl_data[data_key].append(data_val)
        self.ctrl.reset_before_run(obs, info, env=self.env)
        return obs, info

    def launch_training(self, **kwargs):
        '''Since the learning loop varies among controllers, can only delegate to its own `learn()` method. '''

        self.reset()
        self.ctrl.learn(env=self.train_env, **kwargs)

        if self.safety_filter:
            self.safety_filter.learn(env=self.train_env, **kwargs)

        print('Training done.')

        trajs_data = {}
        if self.train_env is not None:
            trajs_data = self.train_env.data
        return dict(trajs_data)

    def compute_metrics(self, trajs_data):
        '''Compute all standard metrics on the given trajectory data.

        Args:
            trajs_data (defaultdict(list)): the raw data from the executed runs.

        Returns:
            metrics (dict): the metrics calculated from the raw data.
        '''

        met = MetricExtractor(trajs_data)
        # collect & compute all sorts of metrics here
        metrics = {
            'average_length': np.asarray(met.get_episode_lengths()).mean(),
            'average_return': np.asarray(met.get_episode_returns()).mean(),
            'average_rmse': np.asarray(met.get_episode_rmse()).mean(),
            'rmse_std': np.asarray(met.get_episode_rmse()).std(),
            'worst_case_rmse_at_0.5': compute_cvar(np.asarray(met.get_episode_rmse()), 0.5, lower_range=False),
            'failure_rate':  np.asarray(met.get_episode_constraint_violations()).mean(),
            'average_constraint_violation': np.asarray(met.get_episode_constraint_violation_steps()).mean(),
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
            ctrl_path (str): the path used to load the controller's model.
            safety_filter_path (str): the path used to load the safety_filter's model.
        '''

        if ctrl_path is not None:
            self.ctrl.load(ctrl_path)
        if safety_filter_path is not None:
            self.safety_filter.load(safety_filter_path)

    def save(self, ctrl_path=None, safety_filter_path=None):
        '''Saves the model of the controller and/or safety filter given checkpoint paths.

        Args:
            ctrl_path (str): the path used to save the controller's model.
            safety_filter_path (str): the path used to save the safety_filter's model.
        '''

        if ctrl_path is not None:
            self.ctrl.save(ctrl_path)
        if safety_filter_path is not None:
            self.safety_filter.save(safety_filter_path)


class RecordDataWrapper(gym.Wrapper):
    '''A wrapper to standardizes logging for benchmark envs.

    currently saved info:
    * obs, reward, done, info, action
    * env.state, env.current_physical_action,
    * env.current_noisy_physical_action, env.current_clipped_action
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
                self.data[key].append(deepcopy(ep_val))
            # re-initialize episode data container
            self.episode_data = defaultdict(list)

    def clear_data(self):
        '''Clears all data in self.data and self.episode_data'''
        self.data = defaultdict(list)
        self.episode_data = defaultdict(list)

    def reset(self, **kwargs):
        '''Wrapper for the gym.env reset function. '''

        if self.env.INFO_IN_RESET:
            obs, info = self.env.reset(**kwargs)
            if 'symbolic_model' in info:
                info.pop('symbolic_model')
            step_data = dict(
                obs=obs, info=info, state=self.env.state
            )
            for key, val in step_data.items():
                self.episode_data[key].append(val)
            return obs, info
        else:
            obs = self.env.reset()
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
            action=action,
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
            data (defaultdict(list)): the raw data from the executed runs, in standard form from the Experiment class.
        '''
        self.data = data

    def get_episode_data(self, key, postprocess_func=lambda x: x):
        '''Extract data field from recorded trajectory data, optionally postprocess each episode data (e.g. get sum).

        Args:
            key (str): the key of the data to retrieve.
            postprocess_func (lambda): a function to process the outgoing data.

        Returns:
            episode_data (list): the desired data.
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
        '''Total length of episodes. '''
        return self.get_episode_data('length', postprocess_func=sum)

    def get_episode_returns(self):
        '''Total reward/return of episodes. '''
        return self.get_episode_data('reward', postprocess_func=sum)

    def get_episode_rmse(self):
        '''Root mean square error of episodes. '''
        return self.get_episode_data('mse',
                                     postprocess_func=lambda x: np.sqrt(np.mean(x)))

    def get_episode_constraint_violations(self):
        '''Occurence of any violation in episodes. '''
        return self.get_episode_data('constraint_violation',
                                     postprocess_func=lambda x: float(any(x)))

    def get_episode_constraint_violation_steps(self):
        '''Total violation steps of episodes. '''
        return self.get_episode_data('constraint_violation',
                                     postprocess_func=sum)
