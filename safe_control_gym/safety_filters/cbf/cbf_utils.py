'''Utils for CBF and CBF_NN safety filters. '''

import torch
import numpy as np
import casadi as cs
from gymnasium.spaces import box, Box


def cbf_cartpole(X: cs.MX,
                 state_limits: list
                 ) -> cs.Function:
    '''Creates a control barrier function (CBF) candidate based on an ellipsoid function.

    Args:
        X (cs.MX): Symbolic state variable.
        state_limits (list): List of the limits on the state.

    Returns:
        cbf_func (cs.Function): Function object of the control barrier function (CBF).
    '''
    shrink_factor = 1.0
    x_max, v_max, theta_max, omega_max = state_limits
    cbf = shrink_factor - (X[0]/x_max)**2 - (X[1]/v_max)**2 - (X[2]/theta_max)**2 - (X[3]/omega_max)**2
    # cbf = 0.5 * (1 - (X[0]) ** 2 / (x_max) ** 2)
    # cbf = 0.5 * (1 - (X[2]) ** 2 / (theta_max) ** 2)
    # cbf = 0.5 * (1 - (X[0]) ** 2 / (x_max) ** 2 - (X[3]) ** 2 / (omega_max) ** 2)
    # cbf = 0.5 * (1 - (X[0]) ** 2 / (x_max) ** 2 - (X[1]) ** 2 / (v_max) ** 2)
    cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    return cbf_func

def linear_function(slope: float) -> cs.Function:
    '''Creates a one dimensional linear function.

    Args:
        slope (float): The slope of the linear function.

    Returns:
        linear_func (cs.Function): Function object of a linear function with desired slope.
    '''
    x = cs.MX.sym('x', 1)
    linear_func = cs.Function('linear', [x], [slope * x], ['x'], ['y'])
    return linear_func

def cartesian_product(*arrays: list) -> np.ndarray:
    '''Creates the cartesian product of a list of arrays from:
       https://stackoverflow.com/a/11146645

    Args:
        arrays (list): List of arrays.

    Returns:
        cartesian (np.ndarray): Cartesian product of all arrays in the list.
    '''
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class CBFBuffer(object):
    '''Storage for replay buffer during training.

    Attributes:
        max_size (int): Maximum size of the replay buffer.
        batch_size (int): Number of samples (steps) per batch.
        scheme (dict): Describes shape and other info of data to be stored.
        keys (list): Names of all data from scheme.
    '''

    def __init__(self,
                 obs_space: box.Box,
                 act_space: box.Box,
                 max_size: int,
                 device: str,
                 batch_size: int
                 ):
        '''
        A buffer for CBF_NN.

        Args:
            obs_space (gym.spaces.box.Box): The observation space.
            act_space (gym.spaces.box.Box): The action space.
            max_size (int): Maximum size of the replay buffer.
            device (str): The device being used.
            batch_size (int): Number of samples (steps) per batch.
        '''

        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size

        self.device=device

        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n

        N = max_size
        self.scheme = {
            'state': {
                'vshape': (N, *obs_dim)
            },
            'act': {
                'vshape': (N, act_dim)
            },
            'barrier_dot': {
                'vshape': (N, 1)
            },
            'barrier_dot_approx': {
                'vshape': (N, 1)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocate space for containers. '''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)

        self.pos = 0
        self.buffer_size = 0

    def __len__(self) -> int:
        '''Returns current size of the buffer.

        Returns:
            buffer_size (int): The buffer size.
        '''
        return self.buffer_size

    def state_dict(self) -> dict:
        '''Returns a snapshot of current buffer.

        Returns:
            state (dict): The state dictionary.
        '''
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self,
                        state: dict,
                        ):
        '''Restores buffer from previous state.

        Args:
            state (dict): The state dictionary.
        '''
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self,
             batch: dict,
             ):
        '''Inserts transition step data (as dict) to storage.

        Args:
            batch (dict):  The batch of data to push.
        '''
        # batch size
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            v_ = np.asarray(v, dtype=dtype).reshape((n,) + shape)

            if self.pos + n <= self.max_size:
                self.__dict__[k][self.pos:self.pos + n] = v_
            else:
                # wrap around
                remain_n = self.pos + n - self.max_size
                self.__dict__[k][self.pos:self.max_size] = v_[:-remain_n]
                self.__dict__[k][:remain_n] = v_[-remain_n:]

        self.pos = (self.pos + n) % self.max_size
        if self.buffer_size < self.max_size:
            self.buffer_size = min(self.max_size, self.pos + n)

    def sample(self,
               batch_size: int=None
               ) -> dict:
        '''Returns data batch.

        Args:
            batch_size (int): The size of batch to sample.

        Returns:
            batch (dict): The batch of data sampled.
        '''
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][1:]
            v = self.__dict__[k].reshape(-1, *shape)[indices]
            batch[k] = torch.as_tensor(v, device=self.device)
        return batch
