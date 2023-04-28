'''Methods to calculate various metrics.'''

import numpy as np


def compute_cvar(data, alpha, lower_range=True):
    '''CVaR as mean of the lower-alpha-percentile of data.
        adapted from https://github.com/nuria95/O-RAAC/blob/57347bc682798ff9f5600131c606517832efe864/oraaclib/util/utilities.py

    Args:
        data (ndarray): the trajectory RMSE collected by the Experiment class
        alpha (float): the percentile upper bound to use
        lower_range (bool): ???

    Returns:
        cvar (float): the resulting CVaR
    '''

    data = np.atleast_2d(data)
    _, N = data.shape
    sorted_data = np.sort(data)

    # NOTE: what does it do?
    # if alpha == 1 or alpha <= 0.5:
    #     cvar = sorted_data[:, :int(alpha * N)].mean(1)
    # else:
    #     cvar = sorted_data[:, int(alpha * N)::].mean(1)
    if lower_range:
        cvar = sorted_data[:, :int(alpha * N)].mean()
    else:
        cvar = sorted_data[:, -int(alpha * N):].mean()
    if np.all(np.isnan(cvar)):
        raise ValueError(f'Not enough samples to compute {alpha} '
                         f'CVaR from {data}')
    else:
        return cvar
