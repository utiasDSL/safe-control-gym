'''Perform normalization on inputs or rewards.'''

import numpy as np
import torch

from gym.spaces import Box


def normalize_angle(x):
    '''Wraps input angle to [-pi, pi].'''
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class RunningMeanStd():
    '''Calulates the running mean and std of a data stream.

    Attributes:
        mean (np.array): mean of data stream.
        var (np.array): variance of data stream.
        count (float): total count of data steam.
    '''

    def __init__(self, epsilon=1e-4, shape=()):
        '''Initializes containers for data mean and variance.

        Args:
            epsilon (float): helps with arithmetic issues.
            shape (tuple): the shape of the data stream's output.
        '''
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr):
        '''Update current stats with a new stream of data.

        Args:
            arr (np.array): 1D array of data, (batch_size, *shape).
        '''
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        '''Util function for `update` method.'''
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)
        new_count = batch_count + self.count
        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class BaseNormalizer(object):
    '''Template/default normalizer.

    Attributes:
        read_only (bool): if to freeze the current stats being tracked.
    '''

    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def __call__(self, x, *args, **kwargs):
        '''Invokes normalization on the given input.'''
        return x

    def state_dict(self):
        '''Returns snapshot of current stats.'''
        return {}

    def load_state_dict(self, _):
        '''Restores the stats from a snapshot.'''
        pass


class MeanStdNormalizer(BaseNormalizer):
    '''Normalize by the running average.'''

    def __init__(self, shape=(), read_only=False, clip=10.0, epsilon=1e-8):
        '''Initializes the data stream tracker.

        Args:
            shape (tuple): shape of data being tracked.
            read_only (bool): if to freeze the tracker.
            clip (float): bounds on the data.
            epsilon (float): offset to provide divide-by-zero.
        '''
        super().__init__(read_only)
        self.read_only = read_only
        self.rms = RunningMeanStd(shape=shape)
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        '''Update tracker given data, optionally normalize the data.'''
        x = np.asarray(x)
        if not self.read_only:
            self.rms.update(x)
        return np.clip(
            (x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
            -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean, 'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RewardStdNormalizer(MeanStdNormalizer):
    '''Reward normalization by running average of returns.

    Papers:
        * arxiv.org/pdf/1808.04355.pdf
        * arxiv.org/pdf/1810.12894.pdf

    Also see:
        * github.com/openai/baselines/issues/538
    '''

    def __init__(self, gamma=0.99, read_only=False, clip=10.0, epsilon=1e-8):
        '''Initializes the data stream tracker.

        Args:
            gamma (float): discount factor for rewards.
            read_only (bool): if to freeze the tracker.
            clip (float): bounds on the data.
            epsilon (float): offset to provide divide-by-zero.
        '''
        # Reward has default shape (1,) or just ().
        super().__init__((), read_only, clip, epsilon)
        self.gamma = gamma
        self.ret = None

    def __call__(self, x, dones):
        '''Update tracker given reward, optionally normalize the reward (only scaling).'''
        x = np.asarray(x)
        if not self.read_only:
            # Track running average of forward discounted returns.
            if self.ret is None:
                self.ret = np.zeros(x.shape[0])
            self.ret = self.ret * self.gamma + x
            self.rms.update(self.ret)
            # Prevent information leak from previous episodes.
            self.ret[dones.astype(np.long)] = 0
        return np.clip(x / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)


class RescaleNormalizer(BaseNormalizer):
    '''Apply constant scaling.'''

    def __init__(self, coef=1.0):
        '''Initializes with fixed scaling constant.

        Args:
            coef (float): scaling coefficient.
        '''
        super().__init__(self)
        self.coef = coef

    def __call__(self, x):
        '''Scale the input.'''
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    '''Scale image pixles from [0,255] to [0,1].'''

    def __init__(self):
        super().__init__(self, 1.0 / 255)


class ActionUnnormalizer(BaseNormalizer):
    '''Assumes policy output action is in [-1,1], unnormalize it for gym env.'''

    def __init__(self, action_space):
        '''Defines the mean and std for the bounded action space.'''
        super().__init__()
        assert isinstance(action_space, Box), 'action space must be gym.spaces.Box'
        low, high = action_space.low, action_space.high
        self.mean = (low + high) / 2.0
        self.std = (high - low) / 2.0

    def __call__(self, action):
        '''Unnormalizes given input action.'''
        x = np.asarray(action)
        return self.mean + x * self.std
