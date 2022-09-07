"""Bayesian Barrier Linear Regression (BBLR) utilities

"""

import numpy as np
import casadi as cs

from gym.spaces import Box

def cbf_cartpole(X, x_pos_max, x_vel_max, theta_max, theta_dot_max, epsilon_norm=None, cbf_scale=1.0):
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

    if epsilon_norm == None:
        # create h(x), or standard control barrier function
        cbf = cbf_scale * (1 - (X[0]) ** 2 / (x_pos_max) ** 2 - (X[1]) ** 2 / (x_vel_max) ** 2)
        cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])
    else:
        # create h_bar(x), the more conservative cbf for pssf
        cbf = cbf_scale * (1 - (X[0]) ** 2 / (x_pos_max) ** 2 - (X[1]) ** 2 / (x_vel_max) ** 2) - epsilon_norm
        cbf_func = cs.Function('barrier', [X], [cbf], ['X'], ['cbf'])

    return cbf_func

def linear_function(slope):
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