"""Control barrier function (CBF) utilities

"""
import numpy as np
import casadi as cs
import torch
from gym.spaces import Box


def cbf_cartpole(X, x_pos_max, x_vel_max, theta_max, theta_dot_max):
    """
    Creates a control barrier function (CBF) candidate based on an ellipsoid function

    Attributes:
        X (cs.sym.MX): Symbolic state variable
        x_pos_max (float): Semi axis of ellipsoid in the first dimension (not used)
        x_vel_max (float): Semi axis of ellipsoid in the second dimension (not used)
        theta_max (float): Semi axis of ellipsoid in the third dimension
        theta_dot_max (float): Semi axis of ellipsoid in the fourth dimension

    Returns:
        cbf_func (cs.Function): Function object of the control barrier function (CBF)

    """
    # cbf = 0.5 * (1 - (X[2]) ** 2 / (theta_max) ** 2 - (X[3]) ** 2 / (theta_dot_max) ** 2)
    # cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    # cbf = 0.5 * (1 - (X[0]) ** 2 / (x_pos_max) ** 2)
    # cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    # cbf = 0.5 * (1 - (X[2]) ** 2 / (theta_max) ** 2)
    # cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    # cbf = 0.5 * (1 - (X[0]) ** 2 / (x_pos_max) ** 2 - (X[3]) ** 2 / (theta_dot_max) ** 2)
    # cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    cbf = 0.5 * (1 - (X[0]) ** 2 / (x_pos_max) ** 2 - (X[1]) ** 2 / (x_vel_max) ** 2)
    cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    return cbf_func


def cbf_quadrotor():
    # TODO: Implementation missing
    return NotImplementedError


def linear_function(slope):
    """
    Creates a one dimensional linear function

    Attributes:
        slope (float): The slope of the linear function

    Returns:
        linear_func (cs.Function): Function object of a linear function with desired slope

    """
    x = cs.MX.sym("x", 1)
    linear_func = cs.Function('linear', [x], [slope * x], ["x"], ["y"])
    return linear_func


def cartesian_product(*arrays):
    """
    Creates the cartesian product of a list of arrays from:
    https://stackoverflow.com/a/11146645

    Attributes:
        arrays (list): List of arrays

    Returns:
        Cartesian product of all arrays in the list

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


class CBFBuffer(object):
    """Storage for replay buffer during training.

    Attributes:
        max_size (int): maximum size of the replay buffer.
        batch_size (int): number of samples (steps) per batch.
        scheme (dict): describs shape & other info of data to be stored.
        keys (list): names of all data from scheme.

    """

    def __init__(self, obs_space, act_space, max_size, batch_size=None):
        super().__init__()
        self.max_size = max_size
        self.batch_size = batch_size

        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n

        N = max_size
        self.scheme = {
            "state": {
                "vshape": (N, *obs_dim)
            },
            "act": {
                "vshape": (N, act_dim)
            },
            "barrier_dot": {
                "vshape": (N, 1)
            },
            "barrier_dot_approx": {
                "vshape": (N, 1)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        """Allocate space for containers.

        """
        for k, info in self.scheme.items():
            assert "vshape" in info, "Scheme must define vshape for {}".format(k)
            vshape = info["vshape"]
            dtype = info.get("dtype", np.float32)
            init = info.get("init", np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)

        self.pos = 0
        self.buffer_size = 0

    def __len__(self):
        """Returns current size of the buffer.

        """
        return self.buffer_size

    def state_dict(self):
        """Returns a snapshot of current buffer.

        """
        state = dict(
            pos=self.pos,
            buffer_size=self.buffer_size,
        )
        for k in self.scheme:
            v = self.__dict__[k]
            state[k] = v
        return state

    def load_state_dict(self, state):
        """Restores buffer from previous state.

        """
        for k, v in state.items():
            self.__dict__[k] = v

    def push(self, batch):
        """Inserts transition step data (as dict) to storage.

        """
        # batch size
        k = list(batch.keys())[0]
        n = batch[k].shape[0]

        for k, v in batch.items():
            shape = self.scheme[k]["vshape"][1:]
            dtype = self.scheme[k].get("dtype", np.float32)
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

    def sample(self, batch_size=None, device=None):
        """Returns data batch.

        """
        if not batch_size:
            batch_size = self.batch_size

        indices = np.random.randint(0, len(self), size=batch_size)
        batch = {}
        for k, info in self.scheme.items():
            shape = info["vshape"][1:]
            v = self.__dict__[k].reshape(-1, *shape)[indices]
            if device is None:
                batch[k] = torch.as_tensor(v)
            else:
                batch[k] = torch.as_tensor(v, device=device)
        return batch
