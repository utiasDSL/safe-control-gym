'''Record Episode Statistics.'''

import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env import VecEnvWrapper


class RecordEpisodeStatistics(gym.Wrapper):
    '''Keep track of episode length and returns per instantiated env

       Based on OpenAI's Gym wrapper record_episode_statistics.py
    '''

    def __init__(self,
                 env,
                 deque_size=None,
                 **kwargs
                 ):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.deque_size = deque_size
        self.t0 = time.time()
        self.episode_return = 0.0
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        # Other tracked stats.
        self.episode_stats = {}
        self.accumulated_stats = {}
        self.queued_stats = {}

    def add_tracker(self,
                    name,
                    init_value,
                    mode='accumulate'
                    ):
        '''Adds a specific stat to be tracked (accumulate|queue).

        Modes to track stats
            * accumulate: rolling sum, e.g. total # of constraint violations during training.
            * queue: finite, individual storage, e.g. returns, lengths, constraint costs.
        '''
        self.episode_stats[name] = init_value
        if mode == 'accumulate':
            self.accumulated_stats[name] = init_value
        elif mode == 'queue':
            self.queued_stats[name] = deque(maxlen=self.deque_size)
        else:
            raise Exception('Tracker mode not implemented.')

    def reset(self,
              **kwargs
              ):
        self.episode_return = 0.0
        self.episode_length = 0
        # Reset other tracked stats.
        for key in self.episode_stats:
            self.episode_stats[key] *= 0
        return self.env.reset(**kwargs)

    def step(self,
             action
             ):
        observation, reward, done, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        # Add other tracked stats.
        for key in self.episode_stats:
            if key in info:
                self.episode_stats[key] += info[key]
        if done:
            info['episode'] = {'r': self.episode_return, 'l': self.episode_length, 't': round(time.time() - self.t0, 6)}
            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
            # Other tracked stats.
            for key in self.episode_stats:
                info['episode'][key] = deepcopy(self.episode_stats[key])
                if key in self.accumulated_stats:
                    self.accumulated_stats[key] += deepcopy(self.episode_stats[key])
                if key in self.queued_stats:
                    self.queued_stats[key].append(deepcopy(self.episode_stats[key]))
                self.episode_stats[key] *= 0
        return observation, reward, done, info


class VecRecordEpisodeStatistics(VecEnvWrapper):
    '''A vectorized wrapper that records episodic statistics.

    E.g. episode lengths, returns, constraint violations.
    '''

    def __init__(self,
                 venv,
                 deque_size=None,
                 **kwargs
                 ):
        super(VecRecordEpisodeStatistics, self).__init__(venv)
        self.deque_size = deque_size
        self.episode_return = np.zeros(self.num_envs)
        self.episode_length = np.zeros(self.num_envs)
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        # Other tracked stats.
        self.episode_stats = {}
        self.accumulated_stats = {}
        self.queued_stats = {}

    def add_tracker(self,
                    name,
                    init_value,
                    mode='accumulate'
                    ):
        '''Adds a specific stat to be tracked (accumulated).'''
        self.episode_stats[name] = [init_value for _ in range(self.num_envs)]
        if mode == 'accumulate':
            self.accumulated_stats[name] = init_value
        elif mode == 'queue':
            self.queued_stats[name] = deque(maxlen=self.deque_size)
        else:
            raise Exception('Tracker mode not implemented.')

    def reset(self,
              **kwargs
              ):
        self.episode_return = np.zeros(self.num_envs)
        self.episode_length = np.zeros(self.num_envs)
        # Reset other tracked stats.
        for key in self.episode_stats:
            for i in range(self.num_envs):
                self.episode_stats[key][i] *= 0
        return self.venv.reset(**kwargs)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        for i, (r, d) in enumerate(zip(reward, done)):
            self.episode_return[i] += r
            self.episode_length[i] += 1
            # Add other tracked stats.
            for key in self.episode_stats:
                if d:
                    inf = info['n'][i]['terminal_info']
                else:
                    inf = info['n'][i]
                if key in inf:
                    self.episode_stats[key][i] += inf[key]
            if d:
                info['n'][i]['episode'] = {'r': self.episode_return[i], 'l': self.episode_length[i]}
                self.return_queue.append(deepcopy(self.episode_return[i]))
                self.length_queue.append(deepcopy(self.episode_length[i]))
                self.episode_return[i] = 0
                self.episode_length[i] = 0
                # Other tracked stats.
                for key in self.episode_stats:
                    info['n'][i]['episode'][key] = deepcopy(self.episode_stats[key][i])
                    if key in self.accumulated_stats:
                        self.accumulated_stats[key] += deepcopy(self.episode_stats[key][i])
                    if key in self.queued_stats:
                        self.queued_stats[key].append(deepcopy(self.episode_stats[key][i]))
                    self.episode_stats[key][i] *= 0
        return obs, reward, done, info
